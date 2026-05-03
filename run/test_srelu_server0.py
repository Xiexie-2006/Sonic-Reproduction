import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block
from mpc.share import share_arith, reconstruct_arith, MOD
from mpc.triple_pool import setup_triple_pool
from mpc.srelu import srelu


HOST = "127.0.0.1"
PORT = 9000


def to_ring(x):
    return (np.array(x, dtype=np.int64) % MOD).astype(np.uint32)


def decode_signed(x):
    return x.astype(np.uint32).view(np.int32)


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    # 测试正数、0、负数
    x_plain = np.array([[5, 0, -3, 12, -9]], dtype=np.int64)
    x = to_ring(x_plain)

    shape = x.shape

    # SReLU 会调用：
    # compare/MSB -> 大量 bit_and
    # B2A        -> secure_mul
    # final mul  -> secure_mul
    #
    # 所以这里预留数量要充足。
    arith_plan = [
        (shape, 100),
    ]

    bit_plan = [
        (shape, 400),
    ]

    with time_block("total_time"):

        send_data(s, (
            "SRELU_CONFIG",
            arith_plan,
            bit_plan
        ))

        with time_block("offline_time"):
            setup_triple_pool(
                conn=s,
                party_id=0,
                arith_plan=arith_plan,
                bit_plan=bit_plan
            )

        x0, x1 = share_arith(x)

        send_data(s, x1)

        with time_block("online_time"):
            y0 = srelu(
                xi=x0,
                conn=s,
                party_id=0
            )

            y1 = recv_data(s)

    y = reconstruct_arith(y0, y1)
    y_signed = decode_signed(y)

    expected = np.maximum(x_plain, 0)

    print("===== SReLU Test =====")
    print("x_plain =")
    print(x_plain)
    print("SReLU MPC =")
    print(y_signed)
    print("Expected =")
    print(expected)

    if np.array_equal(y_signed, expected):
        print("SReLU test PASSED ✅")
    else:
        print("SReLU test FAILED ❌")

    print_report("Party0 SReLU Profiler")

    s.close()


if __name__ == "__main__":
    main()