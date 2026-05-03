import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from crypto.ot_extension import ot_ext_send_bits


HOST = "127.0.0.1"
PORT = 9000


def main():
    s = socket.socket()
    s.connect((HOST, PORT))

    m0 = np.array([[0, 0, 1, 1, 0, 1, 0, 1]], dtype=np.uint32)
    m1 = np.array([[1, 0, 0, 1, 1, 1, 0, 0]], dtype=np.uint32)

    send_data(s, (m0, m1))

    ot_ext_send_bits(
        conn=s,
        m0_arr=m0,
        m1_arr=m1
    )

    result, choices = recv_data(s)

    expected = np.where(choices == 0, m0, m1).astype(np.uint32)

    print("m0       =", m0)
    print("m1       =", m1)
    print("choices  =", choices)
    print("result   =", result)
    print("expected =", expected)

    if np.array_equal(result, expected):
        print("OT Extension test PASSED ✅")
    else:
        print("OT Extension test FAILED ❌")

    s.close()


if __name__ == "__main__":
    main()