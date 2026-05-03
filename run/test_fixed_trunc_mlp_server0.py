import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith
from mpc.fixed_point import encode_fixed, decode_fixed
from mpc.triple_pool import setup_triple_pool
from mpc.secure_nn_fixed_trunc import secure_two_layer_mlp_secret_fixed_trunc


HOST = "127.0.0.1"
PORT = 9000


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    scale_bits = 8

    X_plain = np.array([[0.5, -0.25]], dtype=np.float64)

    W1_plain = np.array([
        [1.0, 1.0],
        [1.0, 3.0]
    ], dtype=np.float64)

    b1_plain = np.array([0.0, 0.0], dtype=np.float64)

    W2_plain = np.array([
        [4.0],
        [7.0]
    ], dtype=np.float64)

    b2_plain = np.array([0.0], dtype=np.float64)

    H_expected = X_plain @ W1_plain + b1_plain
    A_expected = np.maximum(H_expected, 0)
    Y_expected = A_expected @ W2_plain + b2_plain

    X = encode_fixed(X_plain, scale_bits)
    W1 = encode_fixed(W1_plain, scale_bits)
    W2 = encode_fixed(W2_plain, scale_bits)

    b1 = encode_fixed(b1_plain, scale_bits * 2)
    b2 = encode_fixed(b2_plain, scale_bits * 2)

    X0, X1 = share_arith(X)
    W10, W11 = share_arith(W1)
    b10, b11 = share_arith(b1)
    W20, W21 = share_arith(W2)
    b20, b21 = share_arith(b2)

    arith_plan = [
        ((1, 2), 100),
        ((1, 1), 100),
    ]

    bit_plan = [
        ((1, 2), 220),
        ((1, 1), 140),
    ]

    with time_block("total_time"):

        send_data(s, (
            "FIXED_TRUNC_MLP_CONFIG",
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
            W11,
            b11,
            W21,
            b21,
            scale_bits
        ))

        with time_block("online_time"):
            Y0 = secure_two_layer_mlp_secret_fixed_trunc(
                x_i=X0,
                W1_i=W10,
                b1_i=b10,
                W2_i=W20,
                b2_i=b20,
                scale_bits=scale_bits,
                conn=s,
                party_id=0
            )

            Y1 = recv_data(s)

    Y_ring = reconstruct_arith(Y0, Y1)
    Y_mpc = decode_fixed(Y_ring, scale_bits)

    diff = np.abs(Y_mpc - Y_expected)

    print("===== Fixed-point + Secure Truncation MLP Test =====")
    print("X_plain =", X_plain)
    print("H_expected =", H_expected)
    print("ReLU_expected =", A_expected)
    print("Y_expected =", Y_expected)
    print("Y_mpc =", Y_mpc)
    print("abs_error =", diff)
    print("scale_bits =", scale_bits)

    if np.all(diff <= 1.0 / (1 << scale_bits)):
        print("Fixed-point Trunc MLP test PASSED ✅")
    else:
        print("Fixed-point Trunc MLP test FAILED ❌")

    print_report("Party0 Fixed-point Trunc MLP Profiler")

    s.close()


if __name__ == "__main__":
    main()