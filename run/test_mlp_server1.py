import socket

from net.socket_utils import send_data, recv_data
from mpc.secure_nn import secure_two_layer_mlp


HOST = "0.0.0.0"
PORT = 9000


def main():
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(5)

    print("Server1 等待连接...")

    while True:
        conn, _ = s.accept()

        X1, W1, b1, W2, b2 = recv_data(conn)

        Y1 = secure_two_layer_mlp(
            xi=X1,
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2,
            conn=conn,
            party_id=1
        )

        send_data(conn, Y1)

        conn.close()


if __name__ == "__main__":
    main()