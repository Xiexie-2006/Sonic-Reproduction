import os
import sys
import socket
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith, MOD
from mpc.triple_pool import setup_triple_pool
from mpc.maxpool2d_secret import maxpool2d_secret


HOST = "127.0.0.1"
PORT = 9000


def to_ring(x):
    return (np.array(x, dtype=np.int64) % MOD).astype(np.uint32)


def decode_signed(x):
    return x.astype(np.uint32).view(np.int32)


def plain_maxpool2d_nchw(x, kernel_size=2, stride=2):
    n, c, h, w = x.shape

    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1

    y = np.zeros((n, c, out_h, out_w), dtype=np.int64)

    for ni in range(n):
        for ci in range(c):
            for oh in range(out_h):
                for ow in range(out_w):
                    ih = oh * stride
                    iw = ow * stride

                    window = x[
                        ni,
                        ci,
                        ih:ih + kernel_size,
                        iw:iw + kernel_size
                    ]

                    y[ni, ci, oh, ow] = np.max(window)

    return y


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    X_plain = np.array(
        [
            [
                [
                    [1, -2],
                    [5, 3]
                ]
            ]
        ],
        dtype=np.int64
    )

    kernel_size = 2
    stride = 2

    Y_expected = plain_maxpool2d_nchw(
        X_plain,
        kernel_size=kernel_size,
        stride=stride
    )

    X = to_ring(X_plain)
    X0, X1 = share_arith(X)

    # MaxPool 2x2 需要 3 次 secure_max2
    # 每次 secure_max2:
    #   compare_zero -> bit_and
    #   B2A -> secure_mul
    #   select -> secure_mul
    pool_out_shape = (1, 1, 1, 1)

    arith_plan = [
        (pool_out_shape, 80),
    ]

    bit_plan = [
        (pool_out_shape, 300),
    ]

    with time_block("total_time"):

        send_data(s, (
            "MAXPOOL2D_CONFIG",
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

        send_data(s, (
            X1,
            kernel_size,
            stride
        ))

        with time_block("online_time"):
            Y0 = maxpool2d_secret(
                x_i=X0,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                conn=s,
                party_id=0
            )

            Y1 = recv_data(s)

    Y = reconstruct_arith(Y0, Y1)
    Y_signed = decode_signed(Y)

    print("===== Secure MaxPool2D Test =====")
    print("X_plain =")
    print(X_plain)
    print("Y_expected =")
    print(Y_expected)
    print("Y_mpc =")
    print(Y_signed)

    if np.array_equal(Y_signed, Y_expected):
        print("Secure MaxPool2D test PASSED ✅")
    else:
        print("Secure MaxPool2D test FAILED ❌")

    print_report("Party0 Secure MaxPool2D Profiler")

    s.close()


if __name__ == "__main__":
    main()