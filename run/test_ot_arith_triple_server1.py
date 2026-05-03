import socket

from net.socket_utils import send_data, recv_data
from mpc.arith_triple_ot import generate_arith_triples_ot


HOST = "0.0.0.0"
PORT = 9000


def main():
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(5)

    print("Server1 等待连接...")

    while True:
        conn, _ = s.accept()

        shape, count = recv_data(conn)

        triples1 = generate_arith_triples_ot(
            conn=conn,
            party_id=1,
            shape=shape,
            count=count
        )

        send_data(conn, triples1)

        conn.close()


if __name__ == "__main__":
    main()