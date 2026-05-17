import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith, MOD
from mpc.triple_dealer import setup_triple_pool_by_dealer
from mpc.compare_ppa import build_ppa_bit_plan
from mpc.srelu_ppa import srelu_ppa


HOST = "127.0.0.1"
PORT = 9010


def to_ring(x):
    """
    将普通整数映射到 Z_(2^32) 环。
    """
    return (np.array(x, dtype=np.int64) % MOD).astype(np.uint32)


def decode_signed(x):
    """
    将 uint32 环元素解释成 int32，便于显示负数。
    """
    return x.astype(np.uint32).view(np.int32)


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    # 同时测试正数、0 和负数。
    x_plain = np.array(
        [[5, 0, -3, 12, -9]],
        dtype=np.int64,
    )

    x = to_ring(x_plain)
    shape = x.shape

    # PPA SReLU 需要：
    # 1. PPA secure MSB 的 bit triples；
    # 2. B2A 的 arithmetic triple；
    # 3. gate*x 的 arithmetic triple。
    arith_plan = [
        (shape, 20),
    ]

    bit_plan = build_ppa_bit_plan(shape)

    with time_block("total_time"):
        send_data(
            s,
            (
                "SRELU_PPA_CONFIG",
                arith_plan,
                bit_plan,
            ),
        )

        with time_block("offline_time"):
            setup_triple_pool_by_dealer(
                party_id=0,
                arith_plan=arith_plan,
                bit_plan=bit_plan,
            )

        x0, x1 = share_arith(x)

        send_data(s, x1)

        with time_block("online_time"):
            y0 = srelu_ppa(
                xi=x0,
                conn=s,
                party_id=0,
            )

            y1 = recv_data(s)

    y = reconstruct_arith(y0, y1)
    y_signed = decode_signed(y)

    expected = np.maximum(x_plain, 0)

    print("===== SReLU PPA Test =====")
    print("x_plain =")
    print(x_plain)
    print("SReLU PPA MPC =")
    print(y_signed)
    print("Expected =")
    print(expected)

    if np.array_equal(y_signed, expected):
        print("SReLU PPA test PASSED ✅")
    else:
        print("SReLU PPA test FAILED ❌")

    print_report("Party0 SReLU PPA Profiler")

    s.close()


if __name__ == "__main__":
    main()