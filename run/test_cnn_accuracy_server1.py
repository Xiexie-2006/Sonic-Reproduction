import socket

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.triple_pool import setup_triple_pool
from mpc.secure_cnn_fixed import secure_fixed_cnn


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

        if config[0] != "CNN_ACCURACY_CONFIG":
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
                conv_W1,
                conv_b1,
                fc_W1,
                fc_b1,
                scale_bits,
                stride,
                padding
            ) = recv_data(conn)

            with time_block("online_time"):
                logits1 = secure_fixed_cnn(
                    x_i=X1,
                    conv_w_i=conv_W1,
                    conv_b_i=conv_b1,
                    fc_w_i=fc_W1,
                    fc_b_i=fc_b1,
                    scale_bits=scale_bits,
                    conn=conn,
                    party_id=1,
                    stride=stride,
                    padding=padding
                )

                send_data(conn, logits1)

        print_report("Party1 CNN Accuracy Profiler")

        conn.close()


if __name__ == "__main__":
    main()