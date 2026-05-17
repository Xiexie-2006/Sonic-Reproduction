import socket

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.triple_dealer import setup_triple_pool_by_dealer
from mpc.srelu_ppa import srelu_ppa


HOST = "0.0.0.0"
PORT = 9010


def main():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)

    print("Server1 等待 PPA SReLU 连接...")

    while True:
        reset_stats()

        conn, _ = s.accept()

        config = recv_data(conn)

        if config[0] != "SRELU_PPA_CONFIG":
            raise RuntimeError("Unexpected config tag")

        _, arith_plan, bit_plan = config

        with time_block("total_time"):
            with time_block("offline_time"):
                setup_triple_pool_by_dealer(
                    party_id=1,
                    arith_plan=arith_plan,
                    bit_plan=bit_plan,
                )

            x1 = recv_data(conn)

            with time_block("online_time"):
                y1 = srelu_ppa(
                    xi=x1,
                    conn=conn,
                    party_id=1,
                )

                send_data(conn, y1)

        print_report("Party1 SReLU PPA Profiler")

        conn.close()


if __name__ == "__main__":
    main()