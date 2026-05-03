import numpy as np

from crypto.ot_extension import ot_ext_send_bits, ot_ext_recv_bits


def _split_batch_triples(a_i, b_i, c_i, count):
    # 把批量生成的 a、b、c 拆成单个 triple。
    # 返回形式为 [(a_i, b_i, c_i), ...]，
    # 这样后续 bit_and 可以按需取出一个 triple 使用。
    triples = []

    for idx in range(count):
        triples.append((
            a_i[idx].astype(np.uint32),
            b_i[idx].astype(np.uint32),
            c_i[idx].astype(np.uint32)
        ))

    return triples


def generate_bit_triples_ot(conn, party_id, shape, count):
    """
    使用 OT Extension 批量生成 Boolean Beaver triples。

    对每个 triple，满足：

        a = a0 XOR a1
        b = b0 XOR b1
        c = c0 XOR c1 = a AND b

    返回：
        当前方的 triple share 列表：
        [(a_i, b_i, c_i), ...]

    Boolean Beaver triple 主要用于安全计算 bit AND。
    因为 XOR 可以本地完成，但 AND 会涉及两方秘密值的乘积，
    所以需要通过 OT Extension 预先生成可复用的随机 triple。
    """

    # shape 表示单个 triple 中 a、b、c 的形状。
    # count 表示一次生成多少个 triple。
    shape = tuple(shape)
    batch_shape = (count,) + shape

    # 每一方本地随机生成自己的 a_i 和 b_i。
    # 最终完整的 a、b 是两方 share 做 XOR 得到的。
    a_i = np.random.randint(0, 2, size=batch_shape, dtype=np.uint32)
    b_i = np.random.randint(0, 2, size=batch_shape, dtype=np.uint32)

    if party_id == 0:
        a0 = a_i
        b0 = b_i

        # 交叉项 1：a0 & b1
        # P0 是 sender，P1 用 b1 作为 choice

        # 如果 b1=0，P1 得到 r01；
        # 如果 b1=1，P1 得到 r01 XOR a0。
        # 这样两方合起来可以形成 a0 & b1 的分享。
        r01 = np.random.randint(0, 2, size=batch_shape, dtype=np.uint32)

        ot_ext_send_bits(
            conn=conn,
            m0_arr=r01,
            m1_arr=r01 ^ a0
        )

        # 交叉项 2：a1 & b0
        # P1 是 sender，P0 用 b0 作为 choice。

        # P0 作为 receiver，只能拿到和自己 choice 对应的消息，
        # 但不会知道 P1 的 a1。
        q10 = ot_ext_recv_bits(
            conn=conn,
            choice_arr=b0
        )

        # 展开：
        # c = (a0 XOR a1) & (b0 XOR b1)

        # 在布尔乘法中，对应项可以拆成：
        # a0&b0, a0&b1, a1&b0, a1&b1

        # 这里 c0 保存 P0 对这些项的 share。
        c0 = (a0 & b0) ^ r01 ^ q10

        return _split_batch_triples(a0, b0, c0, count)

    else:
        a1 = a_i
        b1 = b_i

        # 交叉项 1：a0 & b1
        # P0 是 sender，P1 是 receiver。

        # P1 用自己的 b1 作为 OT choice，
        # 得到 a0&b1 对应的一份分享。
        q01 = ot_ext_recv_bits(
            conn=conn,
            choice_arr=b1
        )

        # 交叉项 2：a1 & b0
        # P1 是 sender，P0 是 receiver。

        # 这里 P1 构造两条消息：
        # m0 = r10
        # m1 = r10 XOR a1
        # P0 根据 b0 选择其中一条。
        r10 = np.random.randint(0, 2, size=batch_shape, dtype=np.uint32)

        ot_ext_send_bits(
            conn=conn,
            m0_arr=r10,
            m1_arr=r10 ^ a1
        )

        # c1 保存 P1 对 c = a AND b 的分享。
        # 两方 c0 XOR c1 后，应该得到完整的 a AND b。
        c1 = (a1 & b1) ^ q01 ^ r10

        return _split_batch_triples(a1, b1, c1, count)


def generate_bit_triple_ot(conn, party_id, shape):
    """
    单个 Boolean triple 生成接口。
    保留这个函数是为了兼容之前的测试脚本。

    实际实现仍然调用批量生成函数，只是 count 固定为 1。
    这样可以避免重复维护两套 triple 生成逻辑。
    """
    triples = generate_bit_triples_ot(
        conn=conn,
        party_id=party_id,
        shape=shape,
        count=1
    )

    return triples[0]