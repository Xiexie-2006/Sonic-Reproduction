import socket
from net.socket_utils import send_data, recv_data
from mpc.relu import relu

HOST = "0.0.0.0"
PORT = 9000


def main():
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(5)

    print("Server1 等待连接...")

    while True:
        conn, _ = s.accept()

        x1 = recv_data(conn)

        z1 = relu(x1, conn, 1)

        send_data(conn, z1)

        conn.close()


if __name__ == "__main__":
    main()