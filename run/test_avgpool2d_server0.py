import os
import sys
import socket
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith
from mpc.fixed_point import encode_fixed, decode_fixed
from mpc.triple_pool import setup_triple_pool
from mpc.avgpool2d_secret import avgpool2d_secret


HOST = "127.0.0.1"
PORT = 9000


def plain_avgpool2d_nchw(x, kernel_size=2, stride=2):
    n, c, h, w = x.shape

    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1

    y = np.zeros((n, c, out_h, out_w), dtype=np.float64)

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

                    y[ni, ci, oh, ow] = np.mean(window)

    return y


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    scale_bits = 8
    kernel_size = 2
    stride = 2

    X_plain = np.array(
        [
            [
                [
                    [3.25, 4.25],
                    [6.25, 7.25]
                ]
            ]
        ],
        dtype=np.float64
    )

    Y_expected = plain_avgpool2d_nchw(
        X_plain,
        kernel_size=kernel_size,
        stride=stride
    )

    X = encode_fixed(X_plain, scale_bits)

    X0, X1 = share_arith(X)

    pool_out_shape = (1, 1, 1, 1)

    arith_plan = [
        (pool_out_shape, 80),
    ]

    bit_plan = [
        (pool_out_shape, 120),
    ]

    with time_block("total_time"):

        send_data(s, (
            "AVGPOOL2D_CONFIG",
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
            stride,
            scale_bits
        ))

        with time_block("online_time"):
            Y0 = avgpool2d_secret(
                x_i=X0,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                conn=s,
                party_id=0
            )

            Y1 = recv_data(s)

    Y_ring = reconstruct_arith(Y0, Y1)
    Y_mpc = decode_fixed(Y_ring, scale_bits)

    diff = np.abs(Y_mpc - Y_expected)

    print("===== Secure AvgPool2D Test =====")
    print("X_plain =")
    print(X_plain)
    print("Y_expected =")
    print(Y_expected)
    print("Y_mpc =")
    print(Y_mpc)
    print("abs_error =")
    print(diff)

    if np.all(diff <= 1.0 / (1 << scale_bits)):
        print("Secure AvgPool2D test PASSED ✅")
    else:
        print("Secure AvgPool2D test FAILED ❌")

    print_report("Party0 Secure AvgPool2D Profiler")

    s.close()


if __name__ == "__main__":
    main()