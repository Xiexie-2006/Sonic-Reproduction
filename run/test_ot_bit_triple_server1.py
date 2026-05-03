import socket

from net.socket_utils import send_data, recv_data
from mpc.bit_triple_ot import generate_bit_triple_ot


HOST = "0.0.0.0"
PORT = 9000


def main():
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(5)

    print("Server1 等待连接...")

    while True:
        conn, _ = s.accept()

        shape = recv_data(conn)

        a1, b1, c1 = generate_bit_triple_ot(
            conn=conn,
            party_id=1,
            shape=shape
        )

        send_data(conn, (a1, b1, c1))

        conn.close()


if __name__ == "__main__":
    main()