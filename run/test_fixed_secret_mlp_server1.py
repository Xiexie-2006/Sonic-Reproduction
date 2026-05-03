import socket

from net.socket_utils import send_data, recv_data
from mpc.secure_nn_fixed import secure_two_layer_mlp_secret_fixed


HOST = "0.0.0.0"
PORT = 9000


def main():
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(5)

    print("Server1 等待连接...")

    while True:
        conn, _ = s.accept()

        (
            X1,
            W11,
            b11,
            W21,
            b21,
            scale_x_bits,
            scale_w1_bits,
            scale_w2_bits
        ) = recv_data(conn)

        Y1, out_scale_bits = secure_two_layer_mlp_secret_fixed(
            x_i=X1,
            W1_i=W11,
            b1_i=b11,
            W2_i=W21,
            b2_i=b21,
            scale_x_bits=scale_x_bits,
            scale_w1_bits=scale_w1_bits,
            scale_w2_bits=scale_w2_bits,
            conn=conn,
            party_id=1
        )

        send_data(conn, Y1)

        conn.close()


if __name__ == "__main__":
    main()