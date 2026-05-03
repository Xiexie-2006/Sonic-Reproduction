import socket

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.triple_pool import setup_triple_pool
from mpc.secure_nn_fixed_trunc import secure_two_layer_mlp_secret_fixed_trunc


HOST = "0.0.0.0"
PORT = 9000


def main():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)

    print("Server1 等待连接...")

    while True:
        reset_stats()

        conn, _ = s.accept()

        config = recv_data(conn)

        if config[0] != "FIXED_TRUNC_MLP_CONFIG":
            raise RuntimeError("Unexpected config tag")

        _, arith_plan, bit_plan = config

        with time_block("total_time"):

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

        print_report("Party1 Fixed-point Trunc MLP Profiler")

        conn.close()


if __name__ == "__main__":
    main()