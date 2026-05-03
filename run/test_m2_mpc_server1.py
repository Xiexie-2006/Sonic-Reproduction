import os
import sys
import socket


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.triple_dealer import setup_triple_pool_by_dealer
from mpc.secure_m2_fixed import secure_m2_fixed_inference


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

        if config[0] != "M2_MPC_CONFIG":
            raise RuntimeError("Unexpected config tag")

        _, arith_plan, bit_plan = config

        with time_block("total_time"):

            with time_block("offline_time"):
                setup_triple_pool_by_dealer(
                    party_id=1,
                    arith_plan=arith_plan,
                    bit_plan=bit_plan,
                    seed=202406
                )

            (
                X1,

                conv1_w1,
                conv1_b1,
                bn1_eps11,
                bn1_eps21,

                conv2_w1,
                conv2_b1,
                bn2_eps11,
                bn2_eps21,

                fc1_w1,
                fc1_b1,
                bn3_eps11,
                bn3_eps21,

                fc2_w1,
                fc2_b1,
                bn4_eps11,
                bn4_eps21,

                scale_bits
            ) = recv_data(conn)

            with time_block("online_time"):
                y1 = secure_m2_fixed_inference(
                    x_i=X1,

                    conv1_w_i=conv1_w1,
                    conv1_b_i=conv1_b1,
                    bn1_eps1_i=bn1_eps11,
                    bn1_eps2_i=bn1_eps21,

                    conv2_w_i=conv2_w1,
                    conv2_b_i=conv2_b1,
                    bn2_eps1_i=bn2_eps11,
                    bn2_eps2_i=bn2_eps21,

                    fc1_w_i=fc1_w1,
                    fc1_b_i=fc1_b1,
                    bn3_eps1_i=bn3_eps11,
                    bn3_eps2_i=bn3_eps21,

                    fc2_w_i=fc2_w1,
                    fc2_b_i=fc2_b1,
                    bn4_eps1_i=bn4_eps11,
                    bn4_eps2_i=bn4_eps21,

                    scale_bits=scale_bits,
                    conn=conn,
                    party_id=1
                )

                send_data(conn, y1)

        print_report("Party1 Sonic M2 MPC Profiler")

        conn.close()


if __name__ == "__main__":
    main()