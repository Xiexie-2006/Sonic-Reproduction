import os
import sys
import socket


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.triple_pool import setup_triple_pool
from mpc.secure_cnn_pool_fixed import secure_fixed_cnn_with_maxpool


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

        if config[0] != "FIXED_CNN_POOL_CONFIG":
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
                conv_stride,
                conv_padding,
                pool_kernel_size,
                pool_stride
            ) = recv_data(conn)

            with time_block("online_time"):
                Y1 = secure_fixed_cnn_with_maxpool(
                    x_i=X1,
                    conv_w_i=conv_W1,
                    conv_b_i=conv_b1,
                    fc_w_i=fc_W1,
                    fc_b_i=fc_b1,
                    scale_bits=scale_bits,
                    conn=conn,
                    party_id=1,
                    conv_stride=conv_stride,
                    conv_padding=conv_padding,
                    pool_kernel_size=pool_kernel_size,
                    pool_stride=pool_stride
                )

                send_data(conn, Y1)

        print_report("Party1 Fixed-point Secret CNN with MaxPool2D Profiler")

        conn.close()


if __name__ == "__main__":
    main()