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
from mpc.sbn import sbn_fixed


HOST = "127.0.0.1"
PORT = 9000


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    scale_bits = 8

    # 模拟 Conv 输出特征，shape = [N, C, H, W]
    X_plain = np.array(
        [
            [
                [
                    [1.0, -2.0],
                    [3.5, 4.0]
                ],
                [
                    [-1.0, 2.0],
                    [0.5, -0.25]
                ]
            ]
        ],
        dtype=np.float64
    )

    # 每个通道一个 BN 参数
    # eps1 = gamma / delta
    # eps2 = beta - gamma * mu / delta
    eps1_plain = np.array([2.0, -1.5], dtype=np.float64)
    eps2_plain = np.array([0.25, 0.5], dtype=np.float64)

    # 明文期望：
    # z[:, c, :, :] = eps1[c] * x[:, c, :, :] + eps2[c]
    Y_expected = np.zeros_like(X_plain, dtype=np.float64)

    for c in range(X_plain.shape[1]):
        Y_expected[:, c, :, :] = eps1_plain[c] * X_plain[:, c, :, :] + eps2_plain[c]

    # fixed-point 编码
    X = encode_fixed(X_plain, scale_bits)

    # eps1 参与乘法，scale = 2^f
    eps1 = encode_fixed(eps1_plain, scale_bits)

    # eps2 加到乘法结果上，乘法结果 scale = 2^(2f)
    eps2 = encode_fixed(eps2_plain, scale_bits * 2)

    # secret share
    X0, X1 = share_arith(X)
    eps10, eps11 = share_arith(eps1)
    eps20, eps21 = share_arith(eps2)

    feature_shape = X.shape

    arith_plan = [
        (feature_shape, 260),
    ]

    bit_plan = [
        (feature_shape, 420),
    ]

    with time_block("total_time"):

        send_data(s, (
            "SBN_CONFIG",
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
            eps11,
            eps21,
            scale_bits
        ))

        with time_block("online_time"):
            Y0 = sbn_fixed(
                x_i=X0,
                eps1_i=eps10,
                eps2_i=eps20,
                scale_bits=scale_bits,
                conn=s,
                party_id=0
            )

            Y1 = recv_data(s)

    Y_ring = reconstruct_arith(Y0, Y1)
    Y_mpc = decode_fixed(Y_ring, scale_bits)

    diff = np.abs(Y_mpc - Y_expected)

    print("===== Secure BatchNorm / SBN Test =====")
    print("X_plain =")
    print(X_plain)
    print("eps1_plain =", eps1_plain)
    print("eps2_plain =", eps2_plain)
    print("Y_expected =")
    print(Y_expected)
    print("Y_mpc =")
    print(Y_mpc)
    print("abs_error =")
    print(diff)

    if np.all(diff <= 1.0 / (1 << scale_bits)):
        print("Secure BatchNorm / SBN test PASSED ✅")
    else:
        print("Secure BatchNorm / SBN test FAILED ❌")

    print_report("Party0 Secure BatchNorm / SBN Profiler")

    s.close()


if __name__ == "__main__":
    main()