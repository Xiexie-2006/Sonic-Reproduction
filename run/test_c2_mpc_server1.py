import os
import sys
import socket


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.triple_dealer import setup_triple_pool_by_dealer
from mpc.secure_c2_fixed import secure_c2_fixed_inference


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

        if config[0] != "C2_MPC_CONFIG":
            raise RuntimeError("Unexpected config tag")

        _, arith_plan, bit_plan = config

        with time_block("total_time"):

            with time_block("offline_time"):
                setup_triple_pool_by_dealer(
                    party_id=1,
                    arith_plan=arith_plan,
                    bit_plan=bit_plan,
                    seed=202408
                )

            data = recv_data(conn)

            with time_block("online_time"):
                y1 = secure_c2_fixed_inference(
                    x_i=data["X1"],

                    conv1_w_i=data["conv1"],
                    conv1_b_i=data["conv1_b"],
                    bn1_eps1_i=data["bn1_eps1"],
                    bn1_eps2_i=data["bn1_eps2"],

                    conv2_w_i=data["conv2"],
                    conv2_b_i=data["conv2_b"],
                    bn2_eps1_i=data["bn2_eps1"],
                    bn2_eps2_i=data["bn2_eps2"],

                    conv3_w_i=data["conv3"],
                    conv3_b_i=data["conv3_b"],
                    bn3_eps1_i=data["bn3_eps1"],
                    bn3_eps2_i=data["bn3_eps2"],

                    conv4_w_i=data["conv4"],
                    conv4_b_i=data["conv4_b"],
                    bn4_eps1_i=data["bn4_eps1"],
                    bn4_eps2_i=data["bn4_eps2"],

                    conv5_w_i=data["conv5"],
                    conv5_b_i=data["conv5_b"],
                    bn5_eps1_i=data["bn5_eps1"],
                    bn5_eps2_i=data["bn5_eps2"],

                    conv6_w_i=data["conv6"],
                    conv6_b_i=data["conv6_b"],
                    bn6_eps1_i=data["bn6_eps1"],
                    bn6_eps2_i=data["bn6_eps2"],

                    conv7_w_i=data["conv7"],
                    conv7_b_i=data["conv7_b"],
                    bn7_eps1_i=data["bn7_eps1"],
                    bn7_eps2_i=data["bn7_eps2"],

                    conv8_w_i=data["conv8"],
                    conv8_b_i=data["conv8_b"],
                    bn8_eps1_i=data["bn8_eps1"],
                    bn8_eps2_i=data["bn8_eps2"],

                    conv9_w_i=data["conv9"],
                    conv9_b_i=data["conv9_b"],
                    bn9_eps1_i=data["bn9_eps1"],
                    bn9_eps2_i=data["bn9_eps2"],

                    fc1_w_i=data["fc1"],
                    fc1_b_i=data["fc1_b"],
                    bn10_eps1_i=data["bn10_eps1"],
                    bn10_eps2_i=data["bn10_eps2"],

                    scale_bits=data["scale_bits"],
                    conn=conn,
                    party_id=1
                )

                send_data(conn, y1)

        print_report("Party1 Sonic C2 MPC Profiler")

        conn.close()


if __name__ == "__main__":
    main()