import os
import csv
import socket
import numpy as np
import torch

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, get_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith
from mpc.fixed_point import encode_fixed, decode_fixed
from mpc.triple_pool import setup_triple_pool
from mpc.secure_cnn_fixed import secure_fixed_cnn

from models.simple_cnn import SimpleCNN, export_numpy_params


HOST = "127.0.0.1"
PORT = 9000

RESULT_DIR = "results"
RESULT_FILE = os.path.join(RESULT_DIR, "cnn_benchmark.csv")


def build_batch_input(batch_size):
    base_samples = []

    sample1 = np.array([
        [
            [
                [0.5, 1.0, 1.5],
                [2.0, 2.5, 3.0],
                [3.5, 4.0, 4.5]
            ]
        ]
    ], dtype=np.float64)

    sample2 = np.array([
        [
            [
                [-1.0, -0.5, 0.0],
                [0.5, 1.0, 1.5],
                [2.0, 2.5, 3.0]
            ]
        ]
    ], dtype=np.float64)

    sample3 = np.array([
        [
            [
                [1.0, 0.0, -1.0],
                [2.0, 0.5, -2.0],
                [3.0, 1.0, -3.0]
            ]
        ]
    ], dtype=np.float64)

    sample4 = np.array([
        [
            [
                [0.25, 0.5, 0.75],
                [1.0, 1.25, 1.5],
                [1.75, 2.0, 2.25]
            ]
        ]
    ], dtype=np.float64)

    sample5 = np.array([
        [
            [
                [-0.25, -0.5, -0.75],
                [1.0, 0.0, -1.0],
                [2.0, 1.0, 0.0]
            ]
        ]
    ], dtype=np.float64)

    sample6 = np.array([
        [
            [
                [2.0, 1.0, 0.0],
                [1.5, 0.5, -0.5],
                [1.0, 0.0, -1.0]
            ]
        ]
    ], dtype=np.float64)

    sample7 = np.array([
        [
            [
                [3.0, 2.0, 1.0],
                [0.0, -1.0, -2.0],
                [1.0, 2.0, 3.0]
            ]
        ]
    ], dtype=np.float64)

    sample8 = np.array([
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0]
            ]
        ]
    ], dtype=np.float64)

    base_samples = [
        sample1,
        sample2,
        sample3,
        sample4,
        sample5,
        sample6,
        sample7,
        sample8
    ]

    reps = (batch_size + len(base_samples) - 1) // len(base_samples)

    all_samples = []
    for _ in range(reps):
        all_samples.extend(base_samples)

    x = np.concatenate(all_samples[:batch_size], axis=0)

    return x


def run_one_batch(batch_size, scale_bits=8):
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    model = SimpleCNN()
    model.eval()

    X_plain = build_batch_input(batch_size)

    with torch.no_grad():
        x_torch = torch.tensor(X_plain, dtype=torch.float32)
        y_torch = model(x_torch).detach().cpu().numpy().astype(np.float64)

    conv_W_plain, conv_b_plain, fc_W_plain, fc_b_plain = export_numpy_params(model)

    X = encode_fixed(X_plain, scale_bits)
    conv_W = encode_fixed(conv_W_plain, scale_bits)
    fc_W = encode_fixed(fc_W_plain, scale_bits)

    conv_b = encode_fixed(conv_b_plain, scale_bits * 2)
    fc_b = encode_fixed(fc_b_plain, scale_bits * 2)

    X0, X1 = share_arith(X)
    conv_W0, conv_W1 = share_arith(conv_W)
    conv_b0, conv_b1 = share_arith(conv_b)
    fc_W0, fc_W1 = share_arith(fc_W)
    fc_b0, fc_b1 = share_arith(fc_b)

    stride = 1
    padding = 0

    # CNN 结构：
    # input: [batch, 1, 3, 3]
    # conv output: [batch, 1, 2, 2]
    # flatten: [batch, 4]
    # fc output: [batch, 1]
    conv_mul_shape = (batch_size * 4, 1)
    conv_tensor_shape = (batch_size, 1, 2, 2)
    fc_mul_shape = (batch_size, 1)
    fc_out_shape = (batch_size, 1)

    arith_plan = [
        (conv_mul_shape, 60),
        (conv_tensor_shape, 160),
        (fc_mul_shape, 120),
        (fc_out_shape, 120),
    ]

    bit_plan = [
        (conv_tensor_shape, 300),
        (fc_out_shape, 180),
    ]

    with time_block("total_time"):

        send_data(s, (
            "TORCH_CNN_BENCH_CONFIG",
            batch_size,
            scale_bits,
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

        send_data(s, (
            X1,
            conv_W1,
            conv_b1,
            fc_W1,
            fc_b1,
            scale_bits,
            stride,
            padding
        ))

        with time_block("online_time"):
            Y0 = secure_fixed_cnn(
                x_i=X0,
                conv_w_i=conv_W0,
                conv_b_i=conv_b0,
                fc_w_i=fc_W0,
                fc_b_i=fc_b0,
                scale_bits=scale_bits,
                conn=s,
                party_id=0,
                stride=stride,
                padding=padding
            )

            Y1 = recv_data(s)

    Y_ring = reconstruct_arith(Y0, Y1)
    Y_mpc = decode_fixed(Y_ring, scale_bits)

    diff = np.abs(Y_mpc - y_torch)
    max_abs_error = float(np.max(diff))
    passed = bool(np.all(diff <= 1.0 / (1 << scale_bits)))

    print("\n===== PyTorch CNN Batch Benchmark =====")
    print("batch_size =", batch_size)
    print("PyTorch Y =")
    print(y_torch)
    print("MPC Y =")
    print(Y_mpc)
    print("max_abs_error =", max_abs_error)
    print("PASSED ✅" if passed else "FAILED ❌")

    print_report(f"Party0 PyTorch CNN Benchmark batch={batch_size}")

    stats = get_stats()

    row = {
        "batch_size": batch_size,
        "passed": passed,
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

    print(f"\nCNN benchmark CSV saved to: {RESULT_FILE}")


def main():
    batch_sizes = [1, 2, 4]

    rows = []

    for batch_size in batch_sizes:
        row = run_one_batch(
            batch_size=batch_size,
            scale_bits=8
        )
        rows.append(row)

    write_csv(rows)

    print("\n===== CNN Benchmark Summary =====")
    for row in rows:
        print(
            f"batch={row['batch_size']}, "
            f"passed={row['passed']}, "
            f"offline={row['offline_time']:.4f}s, "
            f"online={row['online_time']:.4f}s, "
            f"comm={row['total_comm_bytes']} bytes, "
            f"messages={row['total_messages']}"
        )


if __name__ == "__main__":
    main()