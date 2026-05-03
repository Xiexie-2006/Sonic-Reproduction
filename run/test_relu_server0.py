import socket
import numpy as np
from net.socket_utils import send_data, recv_data
from mpc.relu import relu
from mpc.share import reconstruct_arith

HOST = "127.0.0.1"
PORT = 9000


def test_case(value):
    s = socket.socket()
    s.connect((HOST, PORT))

    x = np.array([value], dtype=np.uint32)

    x0 = np.random.randint(0, 2**32, size=x.shape, dtype=np.uint32)
    x1 = (x - x0) % (2**32)

    send_data(s, x1)

    z0 = relu(x0, s, 0)
    z1 = recv_data(s)

    z = reconstruct_arith(z0, z1)

    print(f"x={int(x[0])}, ReLU={int(z[0])}")

    s.close()


if __name__ == "__main__":
    print("===== ReLU 测试 =====")
    test_case(5)
    test_case(0)
    test_case((-3) & 0xFFFFFFFF)