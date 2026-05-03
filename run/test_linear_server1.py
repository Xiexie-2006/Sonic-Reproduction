import socket

from net.socket_utils import send_data, recv_data
from mpc.linear import linear_public_weight

HOST = "0.0.0.0"
PORT = 9000


def main():
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(5)

    print("Server1 等待连接...")

    while True:
        conn, _ = s.accept()

        X1, W, b = recv_data(conn)

        Y1 = linear_public_weight(X1, W, b=None, party_id=1)

        send_data(conn, Y1)

        conn.close()


if __name__ == "__main__":
    main()