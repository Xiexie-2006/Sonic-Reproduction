import socket

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.triple_dealer import setup_triple_pool_by_dealer
from mpc.trunc_batched import secure_trunc_batched


HOST = "0.0.0.0"
PORT = 9030


def main():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)

    print("Server1 等待 Batched Truncation 连接...")

    while True:
        reset_stats()

        conn, _ = s.accept()

        config = recv_data(conn)

        if config[0] != "TRUNC_BATCHED_CONFIG":
            raise RuntimeError("Unexpected config tag")

        _, arith_plan, bit_plan, scale_bits = config

        with time_block("total_time"):
            with time_block("offline_time"):
                setup_triple_pool_by_dealer(
                    party_id=1,
                    arith_plan=arith_plan,
                    bit_plan=bit_plan,
                )

            x1 = recv_data(conn)

            with time_block("online_time"):
                y1 = secure_trunc_batched(
                    xi=x1,
                    shift_bits=scale_bits,
                    conn=conn,
                    party_id=1,
                )

                send_data(conn, y1)

        print_report("Party1 Batched Secure Truncation Profiler")

        conn.close()


if __name__ == "__main__":
    main()