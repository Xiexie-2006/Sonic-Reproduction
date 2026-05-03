import os
import sys
import socket
import numpy as np
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from data_loader import load_npz_dataset
from mpc.share import share_arith, reconstruct_arith
from mpc.fixed_point import encode_fixed, decode_fixed
from mpc.triple_pool import setup_triple_pool
from mpc.secure_cnn_fixed import secure_fixed_cnn

from models.simple_cnn_classifier import SimpleCNNClassifier, export_numpy_params


HOST = "127.0.0.1"
PORT = 9000

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "toy_cnn_dataset.npz")
MAX_SAMPLES = None


def main():
    reset_stats()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\n找不到数据集文件:\n{DATA_PATH}\n\n"
            f"请先运行:\npython run/make_toy_dataset.py\n"
        )

    s = socket.socket()
    s.connect((HOST, PORT))

    scale_bits = 8
    stride = 1
    padding = 0

    model = SimpleCNNClassifier()
    model.eval()

    X_plain, y_true = load_npz_dataset(DATA_PATH, max_samples=MAX_SAMPLES)
    batch_size = X_plain.shape[0]

    with torch.no_grad():
        x_torch = torch.tensor(X_plain, dtype=torch.float32)
        logits_torch = model(x_torch).detach().cpu().numpy().astype(np.float64)

    pred_torch = np.argmax(logits_torch, axis=1)
    acc_torch = float(np.mean(pred_torch == y_true))

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

    # 当前 SimpleCNNClassifier 结构：
    # input: [batch, 1, 3, 3]
    # conv:  [batch, 2, 2, 2]
    # flat:  [batch, 8]
    # fc:    [batch, 3]
    conv_out_h = 2
    conv_out_w = 2
    conv_out_channels = 2
    fc_out_dim = 3

    conv_mul_shape = (batch_size * conv_out_h * conv_out_w, conv_out_channels)
    conv_tensor_shape = (batch_size, conv_out_channels, conv_out_h, conv_out_w)
    fc_mul_shape = (batch_size, fc_out_dim)
    fc_out_shape = (batch_size, fc_out_dim)

    arith_plan = [
        (conv_mul_shape, 80),
        (conv_tensor_shape, 180),
        (fc_mul_shape, 160),
        (fc_out_shape, 160),
    ]

    bit_plan = [
        (conv_tensor_shape, 360),
        (fc_out_shape, 240),
    ]

    with time_block("total_time"):

        send_data(s, (
            "CNN_ACCURACY_DATASET_CONFIG",
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
            logits0 = secure_fixed_cnn(
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

            logits1 = recv_data(s)

    logits_ring = reconstruct_arith(logits0, logits1)
    logits_mpc = decode_fixed(logits_ring, scale_bits)

    pred_mpc = np.argmax(logits_mpc, axis=1)
    acc_mpc = float(np.mean(pred_mpc == y_true))

    diff = np.abs(logits_mpc - logits_torch)
    max_abs_error = float(np.max(diff))

    same_pred = np.array_equal(pred_torch, pred_mpc)
    close_logits = np.all(diff <= 1.0 / (1 << scale_bits))

    print("===== CNN Dataset Accuracy Test: PyTorch vs MPC =====")
    print("DATA_PATH =", DATA_PATH)
    print("X_test shape =", X_plain.shape)
    print("y_true =", y_true)

    print("\nPyTorch logits =")
    print(logits_torch)
    print("PyTorch pred =", pred_torch)
    print("PyTorch acc =", acc_torch)

    print("\nMPC logits =")
    print(logits_mpc)
    print("MPC pred =", pred_mpc)
    print("MPC acc =", acc_mpc)

    print("\nabs_error =")
    print(diff)
    print("max_abs_error =", max_abs_error)

    if same_pred and close_logits:
        print("CNN dataset accuracy test PASSED ✅")
    else:
        print("CNN dataset accuracy test FAILED ❌")

    print_report("Party0 CNN Dataset Accuracy Profiler")

    s.close()


if __name__ == "__main__":
    main()