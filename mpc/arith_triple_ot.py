import numpy as np

from mpc.share import MOD
from crypto.ot_extension_ring import ot_ext_send_uint32, ot_ext_recv_uint32


def _mod_u32(x):
    # 将输入统一压回 Z_(2^32) 环中。
    # 先转 uint64 是为了避免中间计算溢出，再转回 uint32 保存。
    return (x.astype(np.uint64) % MOD).astype(np.uint32)


def _split_batch_triples(a_i, b_i, c_i, count):
    # 将批量生成的 a、b、c 拆成一个个 triple。
    # 返回形式是 [(a_i, b_i, c_i), ...]，方便后续 secure_mul 逐个取用。
    triples = []

    for idx in range(count):
        triples.append((
            a_i[idx].astype(np.uint32),
            b_i[idx].astype(np.uint32),
            c_i[idx].astype(np.uint32)
        ))

    return triples


def _sum_mod(arr, axis=0):
    # 对指定维度求和，并在 Z_(2^32) 环上取模。
    # OT 中每一 bit 会产生一份 share，最后需要把 32 位贡献加起来。
    return (np.sum(arr.astype(np.uint64), axis=axis) % MOD).astype(np.uint32)


def _powers_for_batch(batch_shape):
    # 生成 [1, 2, 4, ..., 2^31]。
    # 这些权重用于把 bit 选择结果还原成 uint32 整数乘法中的贡献。
    powers = np.array([1 << i for i in range(32)], dtype=np.uint64)

    # reshape 成可以和 batch 数据广播的形状。
    reshape_dims = (32,) + (1,) * len(batch_shape)

    return powers.reshape(reshape_dims)


def generate_arith_triples_ot(conn, party_id, shape, count):
    """
    使用 OT Extension 批量生成 Arithmetic Beaver triples。

    对每个 triple：

        a = a0 + a1 mod 2^32
        b = b0 + b1 mod 2^32
        c = c0 + c1 mod 2^32 = a*b mod 2^32

    核心展开：

        c = (a0+a1)(b0+b1)
          = a0b0 + a0b1 + a1b0 + a1b1

    本地项：
        a0b0 或 a1b1

    交叉项：
        a0b1、a1b0 通过 OT Extension 生成 additive shares

    这个函数的目的就是生成安全乘法需要的 Beaver triple。
    后续 secure_mul 中可以使用这些 triple，在不泄露输入值的情况下完成乘法。
    """

    # shape 表示单个 triple 中 a、b、c 的张量形状。
    # count 表示一次批量生成多少个 triple。
    shape = tuple(shape)
    batch_shape = (count,) + shape

    # 每一方本地随机生成自己的 a_i 和 b_i。
    # 最终的 a 和 b 是两方 share 相加后的结果。
    a_i = np.random.randint(0, MOD, size=batch_shape, dtype=np.uint32)
    b_i = np.random.randint(0, MOD, size=batch_shape, dtype=np.uint32)

    # 预先准备 2^0 到 2^31，用于按 bit 展开乘法。
    powers = _powers_for_batch(batch_shape)

    if party_id == 0:
        a0 = a_i
        b0 = b_i

        # Cross term 1: a0 * b1
        # P0 sender: x = a0
        # P1 receiver: choice = bits(b1)

        # 这里要让 P1 在不知道 a0 的情况下，得到 a0*b1 的一份 additive share；
        # 同时 P0 也不能知道 b1。
        # 因此使用 OT Extension 按 b1 的每一位做选择。
        r01 = np.random.randint(0, MOD, size=(32,) + batch_shape, dtype=np.uint32)

        # a0_expand 用于和 powers 广播相乘。
        # 如果 b1 的某一位为 1，接收方得到 r + a0*2^i；
        # 如果为 0，接收方得到 r。
        a0_expand = a0.astype(np.uint64).reshape((1,) + batch_shape)
        msg0_01 = r01
        msg1_01 = (
            r01.astype(np.uint64)
            + a0_expand * powers
        ) % MOD
        msg1_01 = msg1_01.astype(np.uint32)

        # P0 作为 sender，发送每一位对应的两条消息。
        ot_ext_send_uint32(
            conn=conn,
            m0_arr=msg0_01,
            m1_arr=msg1_01
        )

        # P0 自己保留 -sum(r) 作为 share。
        # P1 那边会得到 sum(r + bit_i*a0*2^i)，
        # 两方相加后正好得到 a0*b1。
        share_a0b1_p0 = (-_sum_mod(r01, axis=0).astype(np.uint64)) % MOD
        share_a0b1_p0 = share_a0b1_p0.astype(np.uint32)

        # Cross term 2: a1 * b0
        # P1 sender: x = a1
        # P0 receiver: choice = bits(b0)

        # 这一部分和上面角色相反。
        # P0 作为 receiver，用自己 b0 的 bit 去选择消息，
        # 从而获得 a1*b0 的一份 share。
        choices_b0 = np.zeros((32,) + batch_shape, dtype=np.uint32)

        for i in range(32):
            choices_b0[i] = ((b0 >> i) & 1).astype(np.uint32)

        q10 = ot_ext_recv_uint32(
            conn=conn,
            choice_arr=choices_b0
        )

        # P0 收到的 32 位选择结果求和后，就是 a1*b0 的一份 share。
        share_a1b0_p0 = _sum_mod(q10, axis=0)

        # 本地项 a0*b0 可以由 P0 自己直接计算。
        local = (
            a0.astype(np.uint64) * b0.astype(np.uint64)
        ) % MOD
        local = local.astype(np.uint32)

        # c0 = 本地项 + 两个交叉项的 P0 share。
        c0 = (
            local.astype(np.uint64)
            + share_a0b1_p0.astype(np.uint64)
            + share_a1b0_p0.astype(np.uint64)
        ) % MOD
        c0 = c0.astype(np.uint32)

        return _split_batch_triples(a0, b0, c0, count)

    else:
        a1 = a_i
        b1 = b_i

        # Cross term 1: a0 * b1
        # P0 sender, P1 receiver

        # P1 用 b1 的每一位作为 OT choice，
        # 在不暴露 b1 的情况下，接收 a0*b1 的一份 additive share。
        choices_b1 = np.zeros((32,) + batch_shape, dtype=np.uint32)

        for i in range(32):
            choices_b1[i] = ((b1 >> i) & 1).astype(np.uint32)

        q01 = ot_ext_recv_uint32(
            conn=conn,
            choice_arr=choices_b1
        )

        # P1 收到的选择消息累加后，就是 a0*b1 的 P1 share。
        share_a0b1_p1 = _sum_mod(q01, axis=0)

        # Cross term 2: a1 * b0
        # P1 sender, P0 receiver

        # 这里 P1 作为 sender，按照 a1 和 2^i 构造 OT 消息。
        # P0 会用 b0 的 bit 作为 choice 来接收。
        r10 = np.random.randint(0, MOD, size=(32,) + batch_shape, dtype=np.uint32)

        a1_expand = a1.astype(np.uint64).reshape((1,) + batch_shape)
        msg0_10 = r10
        msg1_10 = (
            r10.astype(np.uint64)
            + a1_expand * powers
        ) % MOD
        msg1_10 = msg1_10.astype(np.uint32)

        ot_ext_send_uint32(
            conn=conn,
            m0_arr=msg0_10,
            m1_arr=msg1_10
        )

        # P1 保留 -sum(r) 作为 a1*b0 交叉项中的本方 share。
        share_a1b0_p1 = (-_sum_mod(r10, axis=0).astype(np.uint64)) % MOD
        share_a1b0_p1 = share_a1b0_p1.astype(np.uint32)

        # 本地项 a1*b1 可以由 P1 自己直接计算。
        local = (
            a1.astype(np.uint64) * b1.astype(np.uint64)
        ) % MOD
        local = local.astype(np.uint32)

        # c1 = 本地项 + 两个交叉项的 P1 share。
        c1 = (
            local.astype(np.uint64)
            + share_a0b1_p1.astype(np.uint64)
            + share_a1b0_p1.astype(np.uint64)
        ) % MOD
        c1 = c1.astype(np.uint32)

        return _split_batch_triples(a1, b1, c1, count)