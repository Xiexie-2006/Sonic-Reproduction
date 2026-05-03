import socket
import numpy as np
import torch

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith
from mpc.fixed_point import encode_fixed, decode_fixed
from mpc.triple_pool import setup_triple_pool
from mpc.secure_cnn_fixed import secure_fixed_cnn

from models.simple_cnn import SimpleCNN, export_numpy_params


HOST = "127.0.0.1"
PORT = 9000


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    scale_bits = 8
    stride = 1
    padding = 0

    # 1. 构建 PyTorch CNN
    model = SimpleCNN()
    model.eval()

    # 2. 明文输入: N=1, C=1, H=3, W=3
    X_plain = np.array([
        [
            [
                [0.5, 1.0, 1.5],
                [2.0, 2.5, 3.0],
                [3.5, 4.0, 4.5]
            ]
        ]
    ], dtype=np.float64)

    with torch.no_grad():
        x_torch = torch.tensor(X_plain, dtype=torch.float32)
        y_torch = model(x_torch).detach().cpu().numpy().astype(np.float64)

    # 3. 导出 PyTorch 参数
    conv_W_plain, conv_b_plain, fc_W_plain, fc_b_plain = export_numpy_params(model)

    # 4. fixed-point 编码
    X = encode_fixed(X_plain, scale_bits)
    conv_W = encode_fixed(conv_W_plain, scale_bits)
    fc_W = encode_fixed(fc_W_plain, scale_bits)

    # bias 加在乘法累加之后，所以编码到 2f scale
    conv_b = encode_fixed(conv_b_plain, scale_bits * 2)
    fc_b = encode_fixed(fc_b_plain, scale_bits * 2)

    # 5. secret share
    X0, X1 = share_arith(X)
    conv_W0, conv_W1 = share_arith(conv_W)
    conv_b0, conv_b1 = share_arith(conv_b)
    fc_W0, fc_W1 = share_arith(fc_W)
    fc_b0, fc_b1 = share_arith(fc_b)

    # 当前 CNN 结构：
    # Conv:
    #   input  [1,1,3,3]
    #   kernel [1,1,2,2]
    #   output [1,1,2,2]
    #
    # im2col 后 secure_mul shape = (4,1)
    # Conv trunc shape = (1,1,2,2)
    # SReLU shape = (1,1,2,2)
    # FC secure_mul shape = (1,1)
    # FC trunc shape = (1,1)
    conv_mul_shape = (4, 1)
    conv_tensor_shape = (1, 1, 2, 2)
    fc_mul_shape = (1, 1)
    fc_out_shape = (1, 1)

    arith_plan = [
        (conv_mul_shape, 40),
        (conv_tensor_shape, 120),
        (fc_mul_shape, 80),
        (fc_out_shape, 80),
    ]

    bit_plan = [
        (conv_tensor_shape, 240),
        (fc_out_shape, 120),
    ]

    with time_block("total_time"):

        send_data(s, (
            "TORCH_CNN_CONFIG",
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

    print("===== PyTorch CNN vs MPC CNN Test =====")
    print("X_plain =")
    print(X_plain)

    print("\n===== PyTorch Parameters =====")
    print("conv_W_plain =")
    print(conv_W_plain)
    print("conv_b_plain =", conv_b_plain)
    print("fc_W_plain =")
    print(fc_W_plain)
    print("fc_b_plain =", fc_b_plain)

    print("\n===== Output Comparison =====")
    print("PyTorch Y =")
    print(y_torch)
    print("MPC Y =")
    print(Y_mpc)
    print("abs_error =")
    print(diff)

    if np.all(diff <= 1.0 / (1 << scale_bits)):
        print("PyTorch CNN vs MPC CNN test PASSED ✅")
    else:
        print("PyTorch CNN vs MPC CNN test FAILED ❌")

    print_report("Party0 PyTorch CNN Profiler")

    s.close()


if __name__ == "__main__":
    main()