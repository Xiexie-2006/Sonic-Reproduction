import numpy as np

from mpc.share import MOD
from mpc.triple import get_arith_triple
from mpc.mul import secure_mul
from net.profiler import inc, time_block


def b2a_secure(xb_i, conn, party_id):
    """
    Boolean share -> Arithmetic share

    若 x = xb0 XOR xb1，则：

        x = xb0 + xb1 - 2 * xb0 * xb1

    xb0 * xb1 用 arithmetic secure_mul 计算。

    这个函数的作用是把布尔分享结果转换回算术分享。
    在协议里，比较和符号位判断通常会得到 Boolean share，
    但后续如果还要参与乘法、线性层或选择更新，就需要转成 arithmetic share。
    """
    # 记录 B2A 调用次数，方便后面对协议开销做统计。
    inc("b2a_calls")

    # 统计 B2A 转换耗时。
    with time_block("b2a_time"):
        # 统一转成 uint32，保持和 Z_(2^32) 环上的数据类型一致。
        xb_i = xb_i.astype(np.uint32)

        # Boolean share 中：
        #   x = xb0 XOR xb1

        # 为了套用公式 x = xb0 + xb1 - 2*xb0*xb1，
        # 这里把 party0 的 share 放到 u_i，party1 的 share 放到 v_i。
        if party_id == 0:
            u_i = xb_i
            v_i = np.zeros_like(xb_i, dtype=np.uint32)
        else:
            u_i = np.zeros_like(xb_i, dtype=np.uint32)
            v_i = xb_i

        # 计算 xb0 * xb1 需要安全乘法，
        # 因此先获取一个 arithmetic Beaver triple。
        triple_i = get_arith_triple(conn, party_id, xb_i.shape)

        # 安全计算 u_i * v_i。
        # 两方合起来得到的结果就是 xb0 * xb1。
        prod_i = secure_mul(
            xi=u_i,
            yi=v_i,
            triple_i=triple_i,
            conn=conn,
            party_id=party_id
        )

        # 根据公式：
        #   x = xb0 + xb1 - 2 * xb0 * xb1

        # 当前方手里 xb_i 表示自己那份 xb0 或 xb1，
        # prod_i 表示乘积项的一份 arithmetic share。
        # 最终两方 out_i 相加后，就得到 x 的 arithmetic share。
        out_i = (xb_i.astype(np.uint64) - 2 * prod_i.astype(np.uint64)) % MOD

        return out_i.astype(np.uint32)