import numpy as np

from net.socket_utils import send_data, recv_data
from net.profiler import inc, time_block
from mpc.triple import get_bit_triple


def bit_and(xi, yi, conn, party_id):
    # 记录 Boolean AND 协议调用次数。
    # A2B、MSB、比较等模块里会频繁用到 bit_and，所以这个统计比较重要。
    inc("bit_and_calls")

    # 统计 Boolean AND 的通信和计算耗时。
    with time_block("bit_and_time"):
        # 获取 Boolean Beaver triple：
        #   a = a0 XOR a1
        #   b = b0 XOR b1
        #   c = c0 XOR c1 = a AND b

        # 每一方只拿到自己的 ai、bi、ci。
        ai, bi, ci = get_bit_triple(conn, party_id, xi.shape)

        # 打开 e = x XOR a，f = y XOR b。
        # 这里 ei、fi 是当前方的本地 share。
        ei = xi ^ ai
        fi = yi ^ bi

        # 两方交换 ei、fi。
        # 交换之后双方都可以恢复公开的 e 和 f，
        # 但因为 a、b 是随机掩码，所以不会泄露 x、y 本身。
        send_data(conn, (ei, fi))
        ej, fj = recv_data(conn)

        # 恢复公开值 e 和 f。
        e = ei ^ ej
        f = fi ^ fj

        # Boolean Beaver 乘法公式：
        #   x & y = c XOR (e & b) XOR (f & a) XOR (e & f)

        # 其中 e 和 f 是公开值，a、b、c 是 triple 的分享。
        # 为了保持 XOR sharing，最后的公开项 e&f 只加到一方即可。
        if party_id == 0:
            zi = ci ^ (e & bi) ^ (f & ai)
        else:
            zi = ci ^ (e & bi) ^ (f & ai) ^ (e & f)

        return zi.astype(np.uint32)