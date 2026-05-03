import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from mpc.bit_triple_ot import generate_bit_triple_ot


HOST = "127.0.0.1"
PORT = 9000


def main():
    s = socket.socket()
    s.connect((HOST, PORT))

    shape = (1, 4)

    send_data(s, shape)

    a0, b0, c0 = generate_bit_triple_ot(
        conn=s,
        party_id=0,
        shape=shape
    )

    a1, b1, c1 = recv_data(s)

    a = a0 ^ a1
    b = b0 ^ b1
    c = c0 ^ c1

    expected = a & b

    print("a =", a)
    print("b =", b)
    print("c =", c)
    print("expected =", expected)

    if np.array_equal(c, expected):
        print("OT Boolean triple test PASSED ✅")
    else:
        print("OT Boolean triple test FAILED ❌")

    s.close()


if __name__ == "__main__":
    main()