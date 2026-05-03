import socket
import numpy as np
import torch

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith
from mpc.fixed_point import encode_fixed, decode_fixed
from mpc.triple_pool import setup_triple_pool
from mpc.secure_cnn_fixed import secure_fixed_cnn

from models.simple_cnn_classifier import SimpleCNNClassifier, export_numpy_params


HOST = "127.0.0.1"
PORT = 9000


def build_toy_dataset():
    """
    小型分类测试集。

    这里先用固定小样本验证：
        PyTorch 分类预测
        MPC 分类预测
        accuracy 是否一致

    后续接论文数据集时，只需要替换这里的数据加载部分。
    """

    X = np.array(
        [
            [
                [
                    [0.5, 1.0, 1.5],
                    [2.0, 2.5, 3.0],
                    [3.5, 4.0, 4.5]
                ]
            ],
            [
                [
                    [-1.0, -0.5, 0.0],
                    [0.5, 1.0, 1.5],
                    [2.0, 2.5, 3.0]
                ]
            ],
            [
                [
                    [1.0, 0.0, -1.0],
                    [2.0, 0.5, -2.0],
                    [3.0, 1.0, -3.0]
                ]
            ],
            [
                [
                    [0.25, 0.5, 0.75],
                    [1.0, 1.25, 1.5],
                    [1.75, 2.0, 2.25]
                ]
            ],
        ],
        dtype=np.float64
    )

    # 这里的 label 用于 accuracy 测试。
    # 为了验证 MPC pipeline 是否和 PyTorch 完全一致，
    # 我们会同时输出 PyTorch accuracy 和 MPC accuracy。
    y = np.array([0, 1, 2, 0], dtype=np.int64)

    return X, y


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    scale_bits = 8
    stride = 1
    padding = 0

    model = SimpleCNNClassifier()
    model.eval()

    X_plain, y_true = build_toy_dataset()
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

    # bias 在乘法累加后加入，当前层输出 scale 是 2f，所以 bias 编码到 2f
    conv_b = encode_fixed(conv_b_plain, scale_bits * 2)
    fc_b = encode_fixed(fc_b_plain, scale_bits * 2)

    X0, X1 = share_arith(X)
    conv_W0, conv_W1 = share_arith(conv_W)
    conv_b0, conv_b1 = share_arith(conv_b)
    fc_W0, fc_W1 = share_arith(fc_W)
    fc_b0, fc_b1 = share_arith(fc_b)

    # 模型结构：
    # input: [batch, 1, 3, 3]
    # conv output: [batch, 2, 2, 2]
    # flatten: [batch, 8]
    # fc output: [batch, 3]
    conv_mul_shape = (batch_size * 4, 2)
    conv_tensor_shape = (batch_size, 2, 2, 2)
    fc_mul_shape = (batch_size, 3)
    fc_out_shape = (batch_size, 3)

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
            "CNN_ACCURACY_CONFIG",
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

    print("===== CNN Accuracy Test: PyTorch vs MPC =====")
    print("X_plain shape =", X_plain.shape)
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

    same_pred = np.array_equal(pred_torch, pred_mpc)
    close_logits = np.all(diff <= 1.0 / (1 << scale_bits))

    if same_pred and close_logits:
        print("CNN accuracy pipeline test PASSED ✅")
    else:
        print("CNN accuracy pipeline test FAILED ❌")

    print_report("Party0 CNN Accuracy Profiler")

    s.close()


if __name__ == "__main__":
    main()