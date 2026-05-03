import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith, MOD
from mpc.triple_pool import setup_triple_pool
from mpc.linear_secret import linear_secret_weight


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

    X_plain = np.array([[1, 2]], dtype=np.int64)

    W_plain = np.array([
        [3, 4],
        [5, 6]
    ], dtype=np.int64)

    b_plain = np.array([7, 8], dtype=np.int64)

    X = to_ring(X_plain)
    W = to_ring(W_plain)
    b = to_ring(b_plain)

    X0, X1 = share_arith(X)
    W0, W1 = share_arith(W)
    b0, b1 = share_arith(b)

    arith_plan = [
        ((1, 2), 20),
    ]

    bit_plan = []

    with time_block("total_time"):

        send_data(s, (
            "SECRET_LINEAR_CONFIG",
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

        send_data(s, (X1, W1, b1))

        with time_block("online_time"):
            Y0 = linear_secret_weight(
                x_i=X0,
                w_i=W0,
                b_i=b0,
                conn=s,
                party_id=0
            )

            Y1 = recv_data(s)

    Y = reconstruct_arith(Y0, Y1)
    Y_signed = decode_signed(Y)

    expected = X_plain @ W_plain + b_plain

    print("===== Secret Linear Test =====")
    print("X_plain =", X_plain)
    print("W_plain =")
    print(W_plain)
    print("b_plain =", b_plain)
    print("Y =", Y_signed)
    print("Expected =", expected)

    if np.array_equal(Y_signed, expected):
        print("Secret Linear test PASSED ✅")
    else:
        print("Secret Linear test FAILED ❌")

    print_report("Party0 Secret Linear Profiler")

    s.close()


if __name__ == "__main__":
    main()