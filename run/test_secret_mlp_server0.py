import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith, MOD
from mpc.triple_pool import setup_triple_pool
from mpc.secure_nn_secret import secure_two_layer_mlp_secret


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

    X_plain = np.array([[5, -3]], dtype=np.int64)

    W1_plain = np.array([
        [1, 1],
        [1, 3]
    ], dtype=np.int64)

    b1_plain = np.array([0, 0], dtype=np.int64)

    W2_plain = np.array([
        [4],
        [7]
    ], dtype=np.int64)

    b2_plain = np.array([0], dtype=np.int64)

    X = to_ring(X_plain)
    W1 = to_ring(W1_plain)
    b1 = to_ring(b1_plain)
    W2 = to_ring(W2_plain)
    b2 = to_ring(b2_plain)

    X0, X1 = share_arith(X)
    W10, W11 = share_arith(W1)
    b10, b11 = share_arith(b1)
    W20, W21 = share_arith(W2)
    b20, b21 = share_arith(b2)

    arith_plan = [
        ((1, 2), 40),
        ((1, 1), 40),
    ]

    bit_plan = [
        ((1, 2), 120),
    ]

    with time_block("total_time"):

        send_data(s, (
            "SECRET_MLP_CONFIG",
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

        send_data(s, (X1, W11, b11, W21, b21))

        with time_block("online_time"):
            Y0 = secure_two_layer_mlp_secret(
                x_i=X0,
                W1_i=W10,
                b1_i=b10,
                W2_i=W20,
                b2_i=b20,
                conn=s,
                party_id=0
            )

            Y1 = recv_data(s)

    Y = reconstruct_arith(Y0, Y1)
    Y_signed = decode_signed(Y)

    H_expected = X_plain @ W1_plain + b1_plain
    A_expected = np.maximum(H_expected, 0)
    Y_expected = A_expected @ W2_plain + b2_plain

    print("===== Secret MLP Test =====")
    print("X_plain =", X_plain)
    print("H_expected =", H_expected)
    print("ReLU_expected =", A_expected)
    print("Y =", Y_signed)
    print("Expected =", Y_expected)

    if np.array_equal(Y_signed, Y_expected):
        print("Secret MLP test PASSED ✅")
    else:
        print("Secret MLP test FAILED ❌")

    print_report("Party0 Secret MLP Profiler")

    s.close()


if __name__ == "__main__":
    main()