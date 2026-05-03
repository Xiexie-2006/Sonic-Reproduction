import os
import sys
import socket
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith, MOD
from mpc.triple_pool import setup_triple_pool
from mpc.argmax_secure import secure_argmax


HOST = "127.0.0.1"
PORT = 9000


def to_ring(x):
    return (np.array(x, dtype=np.int64) % MOD).astype(np.uint32)


def decode_signed(x):
    return x.astype(np.uint32).view(np.int32)


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    logits_plain = np.array(
        [
            [10, 3, 7],
            [-5, -2, -9],
            [1, 8, 4],
            [6, 2, 11]
        ],
        dtype=np.int64
    )

    expected_idx = np.argmax(logits_plain, axis=1)
    expected_max = np.max(logits_plain, axis=1)

    logits = to_ring(logits_plain)

    logits0, logits1 = share_arith(logits)

    batch_size = logits_plain.shape[0]
    num_classes = logits_plain.shape[1]

    arg_shape = (batch_size,)

    # secure_argmax 对每个后续 class 做一次比较。
    # num_classes=3，所以比较 2 次。
    #
    # 每次比较包含：
    #   compare_zero -> 大量 bit_and
    #   b2a -> secure_mul
    #   max update -> secure_mul
    #   idx update -> secure_mul
    #
    # 这里预留充足 triple。
    arith_plan = [
        (arg_shape, 120),
    ]

    bit_plan = [
        (arg_shape, 300),
    ]

    with time_block("total_time"):

        send_data(s, (
            "SECURE_ARGMAX_CONFIG",
            arith_plan,
            bit_plan
        ))

        with time_block("offline_time"):
            setup_triple_pool(
                conn=s,
                party_id=0,
                arith_plan=arith_plan,
                bit_plan=bit_plan
            )

        send_data(s, logits1)

        with time_block("online_time"):
            idx0, max0 = secure_argmax(
                logits_i=logits0,
                conn=s,
                party_id=0,
                strict=True
            )

            idx1, max1 = recv_data(s)

    idx_ring = reconstruct_arith(idx0, idx1)
    max_ring = reconstruct_arith(max0, max1)

    idx_signed = decode_signed(idx_ring)
    max_signed = decode_signed(max_ring)

    print("===== Secure Argmax Test =====")
    print("logits_plain =")
    print(logits_plain)

    print("expected_idx =", expected_idx)
    print("mpc_idx      =", idx_signed)

    print("expected_max =", expected_max)
    print("mpc_max      =", max_signed)

    ok_idx = np.array_equal(idx_signed, expected_idx)
    ok_max = np.array_equal(max_signed, expected_max)

    if ok_idx and ok_max:
        print("Secure Argmax test PASSED ✅")
    else:
        print("Secure Argmax test FAILED ❌")

    print_report("Party0 Secure Argmax Profiler")

    s.close()


if __name__ == "__main__":
    main()