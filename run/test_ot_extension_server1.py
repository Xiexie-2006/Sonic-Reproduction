import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from crypto.ot_extension import ot_ext_recv_bits


HOST = "0.0.0.0"
PORT = 9000


def main():
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(5)

    print("Server1 等待连接...")

    while True:
        conn, _ = s.accept()

        m0, m1 = recv_data(conn)

        choices = np.array([[0, 1, 1, 0, 1, 0, 0, 1]], dtype=np.uint32)

        result = ot_ext_recv_bits(
            conn=conn,
            choice_arr=choices
        )

        send_data(conn, (result, choices))

        conn.close()


if __name__ == "__main__":
    main()