import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from mpc.share import reconstruct_arith, MOD
from mpc.secure_nn import secure_two_layer_mlp


HOST = "127.0.0.1"
PORT = 9000


def to_ring(x):
    return (np.array(x, dtype=np.int64) % MOD).astype(np.uint32)


def decode_signed(x):
    return x.astype(np.uint32).view(np.int32)


def share_arith_local(x):
    x0 = np.random.randint(0, MOD, size=x.shape, dtype=np.uint32)
    x1 = ((x.astype(np.int64) - x0.astype(np.int64)) % MOD).astype(np.uint32)
    return x0, x1


def main():
    s = socket.socket()
    s.connect((HOST, PORT))

    # 明文输入，包含负数，测试 ReLU 是否真的截断负值
    X_plain = np.array([[5, -3]], dtype=np.int64)

    # 第一层：
    # H = XW1
    # H[0] = 5*1 + (-3)*1 = 2
    # H[1] = 5*1 + (-3)*3 = -4
    # ReLU(H) = [2, 0]
    W1_plain = np.array([
        [1, 1],
        [1, 3]
    ], dtype=np.int64)

    b1_plain = np.array([0, 0], dtype=np.int64)

    # 第二层：
    # Y = [2,0] @ [[4],[7]] = 8
    W2_plain = np.array([
        [4],
        [7]
    ], dtype=np.int64)

    b2_plain = np.array([0], dtype=np.int64)

    X = to_ring(X_plain)
    W1 = to_ring(W1_plain)
    b1 = to_ring(b1_plain)
    W2 = to_ring(W2_plain)
    b2 = to_ring(b2_plain)

    X0, X1 = share_arith_local(X)

    send_data(s, (X1, W1, b1, W2, b2))

    Y0 = secure_two_layer_mlp(
        xi=X0,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        conn=s,
        party_id=0
    )

    Y1 = recv_data(s)

    Y = reconstruct_arith(Y0, Y1)

    H_expected = X_plain @ W1_plain + b1_plain
    A_expected = np.maximum(H_expected, 0)
    Y_expected = A_expected @ W2_plain + b2_plain

    print("X_plain =", X_plain)
    print("H_expected =", H_expected)
    print("ReLU_expected =", A_expected)
    print("Y =", decode_signed(Y))
    print("Expected =", Y_expected)

    s.close()


if __name__ == "__main__":
    main()