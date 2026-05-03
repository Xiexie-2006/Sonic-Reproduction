import socket

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block
from mpc.secure_nn_fixed_trunc import secure_two_layer_mlp_secret_fixed_trunc
from mpc.triple_pool import setup_triple_pool


HOST = "0.0.0.0"
PORT = 9000


def main():
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(5)

    print("Server1 等待连接...")

    while True:
        reset_stats()

        conn, _ = s.accept()

        batch = 3

        arith_plan = [
            ((batch, 2), 100),
            ((batch, 1), 100),
        ]

        bit_plan = [
            ((batch, 2), 200),
            ((batch, 1), 120),
        ]

        with time_block("total_time"):

            # Offline
            with time_block("offline_time"):
                setup_triple_pool(
                    conn=conn,
                    party_id=1,
                    arith_plan=arith_plan,
                    bit_plan=bit_plan
                )

            (
                X1,
                W11,
                b11,
                W21,
                b21,
                scale_bits
            ) = recv_data(conn)

            # Online
            with time_block("online_time"):
                Y1 = secure_two_layer_mlp_secret_fixed_trunc(
                    x_i=X1,
                    W1_i=W11,
                    b1_i=b11,
                    W2_i=W21,
                    b2_i=b21,
                    scale_bits=scale_bits,
                    conn=conn,
                    party_id=1
                )

                send_data(conn, Y1)

        print_report("Party1 Batch Offline / Online Profiler")

        conn.close()


if __name__ == "__main__":
    main()