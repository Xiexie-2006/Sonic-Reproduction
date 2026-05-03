import os
import sys
import socket


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.triple_pool import setup_triple_pool
from mpc.argmax_secure import secure_argmax


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

        if config[0] != "SECURE_ARGMAX_CONFIG":
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

            logits1 = recv_data(conn)

            with time_block("online_time"):
                idx1, max1 = secure_argmax(
                    logits_i=logits1,
                    conn=conn,
                    party_id=1,
                    strict=True
                )

                send_data(conn, (idx1, max1))

        print_report("Party1 Secure Argmax Profiler")

        conn.close()


if __name__ == "__main__":
    main()