import socket
import numpy as np
from net.socket_utils import send_data, recv_data
from mpc.relu import relu
from mpc.share import reconstruct_arith

HOST = '127.0.0.1'
PORT = 9000

def main():
    s = socket.socket()
    s.connect((HOST, PORT))

    x = np.array([5], dtype=np.uint32)

    x0 = np.random.randint(0, 2**32, size=x.shape, dtype=np.uint32)
    x1 = (x - x0) % (2**32)

    send_data(s, x1)

    z0 = relu(x0, s, 0)
    z1 = recv_data(s)

    print("ReLU:", reconstruct_arith(z0, z1))

if __name__ == "__main__":
    main()