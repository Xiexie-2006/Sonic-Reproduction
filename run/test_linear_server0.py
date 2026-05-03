import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from mpc.share import reconstruct_arith
from mpc.linear import linear_public_weight

HOST = "127.0.0.1"
PORT = 9000


def main():
    s = socket.socket()
    s.connect((HOST, PORT))

    # 明文输入：1 x 3
    X = np.array([[1, 2, 3]], dtype=np.uint32)

    # 公开权重：3 x 2
    W = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ], dtype=np.uint32)

    # 公开偏置：1 x 2
    b = np.array([7, 8], dtype=np.uint32)

    # arithmetic share
    X0 = np.random.randint(0, 2**32, size=X.shape, dtype=np.uint32)
    X1 = (X - X0) % (2**32)

    send_data(s, (X1, W, b))

    Y0 = linear_public_weight(X0, W, b, party_id=0)
    Y1 = recv_data(s)

    Y = reconstruct_arith(Y0, Y1)

    expected = (X.astype(np.uint64) @ W.astype(np.uint64) + b.astype(np.uint64)) % (2**32)

    print("X =", X)
    print("Y =", Y)
    print("Expected =", expected.astype(np.uint32))

    s.close()


if __name__ == "__main__":
    main()