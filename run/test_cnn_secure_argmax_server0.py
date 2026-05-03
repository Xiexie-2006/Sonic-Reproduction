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
from mpc.argmax_secure import secure_argmax

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

    # SimpleCNNClassifier:
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

    arg_shape = (batch_size,)

    arith_plan = [
        # CNN inference
        (conv_mul_shape, 80),
        (conv_tensor_shape, 180),
        (fc_mul_shape, 160),
        (fc_out_shape, 160),

        # secure_argmax
        (arg_shape, 160),
    ]

    bit_plan = [
        # CNN inference
        (conv_tensor_shape, 360),
        (fc_out_shape, 240),

        # secure_argmax
        (arg_shape, 360),
    ]

    with time_block("total_time"):

        send_data(s, (
            "CNN_SECURE_ARGMAX_CONFIG",
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

            pred0, max0 = secure_argmax(
                logits_i=logits0,
                conn=s,
                party_id=0,
                strict=True
            )

            pred1, max1 = recv_data(s)

    pred_ring = reconstruct_arith(pred0, pred1)
    pred_mpc = pred_ring.astype(np.uint32).view(np.int32)

    max_ring = reconstruct_arith(max0, max1)
    max_mpc = decode_fixed(max_ring, scale_bits)

    acc_mpc = float(np.mean(pred_mpc == y_true))

    # 为了测试数值一致性，这里仍然重构 logits 不是必须的。
    # secure prediction 场景下，最终只需要公开 pred_mpc。
    expected_max = np.max(logits_torch, axis=1)

    pred_same = np.array_equal(pred_torch, pred_mpc)
    max_diff = np.abs(max_mpc - expected_max)
    max_close = np.all(max_diff <= 1.0 / (1 << scale_bits))

    print("===== CNN Secure Argmax Accuracy Test =====")
    print("DATA_PATH =", DATA_PATH)
    print("X_test shape =", X_plain.shape)
    print("y_true =", y_true)

    print("\nPyTorch logits =")
    print(logits_torch)
    print("PyTorch pred =", pred_torch)
    print("PyTorch acc =", acc_torch)

    print("\nMPC secure pred =", pred_mpc)
    print("MPC secure acc =", acc_mpc)

    print("\nPyTorch max logits =", expected_max)
    print("MPC secure max     =", max_mpc)
    print("max_abs_error      =", max_diff)

    if pred_same and max_close:
        print("CNN secure argmax accuracy test PASSED ✅")
    else:
        print("CNN secure argmax accuracy test FAILED ❌")

    print_report("Party0 CNN Secure Argmax Accuracy Profiler")

    s.close()


if __name__ == "__main__":
    main()