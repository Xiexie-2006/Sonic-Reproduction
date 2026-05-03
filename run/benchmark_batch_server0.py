import os
import csv
import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, get_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith
from mpc.fixed_point import encode_fixed, decode_fixed
from mpc.secure_nn_fixed_trunc import secure_two_layer_mlp_secret_fixed_trunc
from mpc.triple_pool import setup_triple_pool


HOST = "127.0.0.1"
PORT = 9000

RESULT_DIR = "results"
RESULT_FILE = os.path.join(RESULT_DIR, "benchmark.csv")


def build_batch_input(batch_size):
    """
    构造 batch 输入。
    每行都是 2 维输入，用于测试不同 batch 下的安全推理。
    """

    base = np.array([
        [0.5, -0.25],
        [1.0, 0.5],
        [-0.5, 0.25],
        [2.0, -1.0],
        [-1.0, 0.75],
        [0.25, 0.25],
        [1.5, -0.5],
        [-0.25, -0.25],
    ], dtype=np.float64)

    reps = (batch_size + len(base) - 1) // len(base)
    X = np.vstack([base for _ in range(reps)])

    return X[:batch_size]


def run_one_batch(batch_size, scale_bits=8):
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    # 当前网络结构：
    # X shape: [batch, 2]
    # hidden shape: [batch, 2]
    # output shape: [batch, 1]
    arith_plan = [
        ((batch_size, 2), 120),
        ((batch_size, 1), 120),
    ]

    bit_plan = [
        ((batch_size, 2), 240),
        ((batch_size, 1), 160),
    ]

    with time_block("total_time"):

        # 先把 batch 配置发给 server1
        send_data(s, (
            "BENCH_CONFIG",
            batch_size,
            scale_bits,
            arith_plan,
            bit_plan
        ))

        # Offline phase
        with time_block("offline_time"):
            setup_triple_pool(
                conn=s,
                party_id=0,
                arith_plan=arith_plan,
                bit_plan=bit_plan
            )

        # 明文输入
        X_plain = build_batch_input(batch_size)

        W1_plain = np.array([
            [1.0, 1.0],
            [1.0, 3.0]
        ], dtype=np.float64)

        b1_plain = np.array([0.0, 0.0], dtype=np.float64)

        W2_plain = np.array([
            [4.0],
            [7.0]
        ], dtype=np.float64)

        b2_plain = np.array([0.0], dtype=np.float64)

        # 明文期望
        H_expected = X_plain @ W1_plain + b1_plain
        A_expected = np.maximum(H_expected, 0)
        Y_expected = A_expected @ W2_plain + b2_plain

        # fixed-point 编码
        X = encode_fixed(X_plain, scale_bits)
        W1 = encode_fixed(W1_plain, scale_bits)
        W2 = encode_fixed(W2_plain, scale_bits)

        # 线性层输出先是 2f scale，所以 bias 编码到 2f
        b1 = encode_fixed(b1_plain, scale_bits * 2)
        b2 = encode_fixed(b2_plain, scale_bits * 2)

        # 全部 secret share
        X0, X1 = share_arith(X)
        W10, W11 = share_arith(W1)
        b10, b11 = share_arith(b1)
        W20, W21 = share_arith(W2)
        b20, b21 = share_arith(b2)

        # 发给 Party1
        send_data(s, (
            X1,
            W11,
            b11,
            W21,
            b21,
            scale_bits
        ))

        # Online phase
        with time_block("online_time"):
            Y0 = secure_two_layer_mlp_secret_fixed_trunc(
                x_i=X0,
                W1_i=W10,
                b1_i=b10,
                W2_i=W20,
                b2_i=b20,
                scale_bits=scale_bits,
                conn=s,
                party_id=0
            )

            Y1 = recv_data(s)

        Y_ring = reconstruct_arith(Y0, Y1)
        Y_mpc = decode_fixed(Y_ring, scale_bits)

    diff = np.abs(Y_mpc - Y_expected)
    max_abs_error = float(np.max(diff))

    ok = bool(np.all(diff <= 1.0 / (1 << scale_bits)))

    print("\n===== Batch Benchmark Result =====")
    print("batch_size =", batch_size)
    print("Y_expected =")
    print(Y_expected)
    print("Y_mpc =")
    print(Y_mpc)
    print("max_abs_error =", max_abs_error)
    print("PASSED ✅" if ok else "FAILED ❌")

    print_report(f"Party0 Benchmark batch={batch_size}")

    stats = get_stats()

    row = {
        "batch_size": batch_size,
        "passed": ok,
        "max_abs_error": max_abs_error,

        "bytes_sent": stats.get("bytes_sent", 0),
        "bytes_recv": stats.get("bytes_recv", 0),
        "total_comm_bytes": stats.get("bytes_sent", 0) + stats.get("bytes_recv", 0),

        "send_messages": stats.get("send_messages", 0),
        "recv_messages": stats.get("recv_messages", 0),
        "total_messages": stats.get("send_messages", 0) + stats.get("recv_messages", 0),

        "offline_arith_triples": stats.get("offline_arith_triples", 0),
        "offline_bit_triples": stats.get("offline_bit_triples", 0),

        "secure_mul_calls": stats.get("secure_mul_calls", 0),
        "bit_and_calls": stats.get("bit_and_calls", 0),
        "b2a_calls": stats.get("b2a_calls", 0),
        "a2b_calls": stats.get("a2b_calls", 0),
        "trunc_calls": stats.get("trunc_calls", 0),
        "relu_calls": stats.get("relu_calls", 0),
        "linear_calls": stats.get("linear_secret_calls", 0),

        "offline_time": stats.get("offline_time", 0.0),
        "online_time": stats.get("online_time", 0.0),
        "total_time": stats.get("total_time", 0.0),
        "secure_mul_time": stats.get("secure_mul_time", 0.0),
        "bit_and_time": stats.get("bit_and_time", 0.0),
        "b2a_time": stats.get("b2a_time", 0.0),
        "a2b_time": stats.get("a2b_time", 0.0),
        "trunc_time": stats.get("trunc_time", 0.0),
        "relu_time": stats.get("relu_time", 0.0),
        "linear_time": stats.get("linear_secret_time", 0.0),
    }

    s.close()

    return row


def write_csv(rows):
    os.makedirs(RESULT_DIR, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(RESULT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBenchmark CSV saved to: {RESULT_FILE}")


def main():
    batch_sizes = [1, 2, 4, 8]

    rows = []

    for batch_size in batch_sizes:
        row = run_one_batch(batch_size=batch_size, scale_bits=8)
        rows.append(row)

    write_csv(rows)

    print("\n===== Benchmark Summary =====")
    for row in rows:
        print(
            f"batch={row['batch_size']}, "
            f"passed={row['passed']}, "
            f"offline={row['offline_time']:.4f}s, "
            f"online={row['online_time']:.4f}s, "
            f"comm={row['total_comm_bytes']} bytes"
        )


if __name__ == "__main__":
    main()