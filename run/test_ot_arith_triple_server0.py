import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from mpc.share import MOD
from mpc.arith_triple_ot import generate_arith_triples_ot


HOST = "127.0.0.1"
PORT = 9000


def main():
    s = socket.socket()
    s.connect((HOST, PORT))

    shape = (1, 3)
    count = 2

    send_data(s, (shape, count))

    triples0 = generate_arith_triples_ot(
        conn=s,
        party_id=0,
        shape=shape,
        count=count
    )

    triples1 = recv_data(s)

    ok = True

    for idx, (t0, t1) in enumerate(zip(triples0, triples1)):
        a0, b0, c0 = t0
        a1, b1, c1 = t1

        a = (a0.astype(np.uint64) + a1.astype(np.uint64)) % MOD
        b = (b0.astype(np.uint64) + b1.astype(np.uint64)) % MOD
        c = (c0.astype(np.uint64) + c1.astype(np.uint64)) % MOD

        expected = (a * b) % MOD

        print(f"\nTriple {idx}")
        print("a =", a.astype(np.uint32))
        print("b =", b.astype(np.uint32))
        print("c =", c.astype(np.uint32))
        print("expected =", expected.astype(np.uint32))

        if not np.array_equal(c.astype(np.uint32), expected.astype(np.uint32)):
            ok = False

    if ok:
        print("\nOT Arithmetic triple test PASSED ✅")
    else:
        print("\nOT Arithmetic triple test FAILED ❌")

    s.close()


if __name__ == "__main__":
    main()