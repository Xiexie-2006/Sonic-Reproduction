import numpy as np
from mpc.share import MOD
from net.socket_utils import send_data, recv_data
from net.profiler import inc, time_block


def secure_mul(xi, yi, triple_i, conn, party_id):
    # 记录安全乘法调用次数。
    # 线性层、卷积、ReLU、MaxPool、Argmax 等模块都会大量调用 secure_mul。
    inc("secure_mul_calls")

    # 统计安全乘法耗时。
    with time_block("secure_mul_time"):
        # Beaver triple：
        #   a, b 是随机值
        #   c = a * b

        # 每一方只持有自己的 share：
        #   ai, bi, ci
        ai, bi, ci = triple_i

        # 打开 e = x - a，f = y - b。
        # 因为 a 和 b 是随机掩码，公开 e、f 不会直接泄露 x、y。
        ei = (xi - ai) % MOD
        fi = (yi - bi) % MOD

        # 两方交换自己的 e_i 和 f_i。
        send_data(conn, (ei, fi))
        ej, fj = recv_data(conn)

        # 重构公开的 e 和 f。
        e = (ei + ej) % MOD
        f = (fi + fj) % MOD

        # Beaver 乘法公式：

        # xy = c + e*b + f*a + e*f

        # 其中 e、f 是公开值，a、b、c 是 triple。
        # 为了保持 arithmetic sharing，公开项 e*f 只加到一方即可。
        if party_id == 0:
            zi = (
                ci.astype(np.uint64)
                + e.astype(np.uint64) * bi.astype(np.uint64)
                + f.astype(np.uint64) * ai.astype(np.uint64)
            ) % MOD
        else:
            zi = (
                ci.astype(np.uint64)
                + e.astype(np.uint64) * bi.astype(np.uint64)
                + f.astype(np.uint64) * ai.astype(np.uint64)
                + e.astype(np.uint64) * f.astype(np.uint64)
            ) % MOD

        return zi.astype(np.uint32)