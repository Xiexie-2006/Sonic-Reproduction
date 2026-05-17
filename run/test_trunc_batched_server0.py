import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith
from mpc.fixed_point import encode_fixed, decode_fixed
from mpc.triple_dealer import setup_triple_pool_by_dealer
from mpc.trunc_batched import secure_trunc_batched


HOST = "127.0.0.1"
PORT = 9030


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    scale_bits = 8

    # ------------------------------------------------------------
    # 1. 构造需要截断的 fixed-point 数据
    # ------------------------------------------------------------
    # 这里 x_plain 先按 2^(2f) 编码，
    # 再通过 secure_trunc_batched 右移 f 位，
    # 最后按 2^f 解码，应该恢复到原来的 x_plain。
    x_plain = np.array(
        [[1.25, -2.5, 0.5, -0.75]],
        dtype=np.float64,
    )

    x_encoded_high_scale = encode_fixed(
        x_plain,
        scale_bits * 2,
    )

    x0, x1 = share_arith(x_encoded_high_scale)

    value_shape = x_encoded_high_scale.shape

    # ------------------------------------------------------------
    # 2. 准备 triple plan
    # ------------------------------------------------------------
    # secure_trunc_batched 内部：
    # 1. A2B 需要 bit triples；
    # 2. 批量 B2A 需要 arithmetic triple，shape 是 (32, *value_shape)。
    #
    # bit_plan 这里给足一些，避免 A2B full-adder 中 bit_and 不够。
    arith_plan = [
        ((32,) + value_shape, 5),
    ]

    bit_plan = [
        (value_shape, 100),
    ]

    with time_block("total_time"):
        send_data(
            s,
            (
                "TRUNC_BATCHED_CONFIG",
                arith_plan,
                bit_plan,
                scale_bits,
            ),
        )

        with time_block("offline_time"):
            setup_triple_pool_by_dealer(
                party_id=0,
                arith_plan=arith_plan,
                bit_plan=bit_plan,
            )

        send_data(s, x1)

        with time_block("online_time"):
            y0 = secure_trunc_batched(
                xi=x0,
                shift_bits=scale_bits,
                conn=s,
                party_id=0,
            )

            y1 = recv_data(s)

    y_ring = reconstruct_arith(y0, y1)
    y_mpc = decode_fixed(y_ring, scale_bits)

    abs_error = np.abs(y_mpc - x_plain)

    print("===== Batched Secure Truncation Test =====")
    print("x_plain =")
    print(x_plain)
    print("Y_mpc after trunc =")
    print(y_mpc)
    print("abs_error =")
    print(abs_error)

    if np.all(abs_error <= 1.0 / (1 << scale_bits)):
        print("Batched secure truncation test PASSED ✅")
    else:
        print("Batched secure truncation test FAILED ❌")

    print_report("Party0 Batched Secure Truncation Profiler")

    s.close()


if __name__ == "__main__":
    main()