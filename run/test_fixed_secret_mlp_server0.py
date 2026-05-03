import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from mpc.share import share_arith, reconstruct_arith
from mpc.fixed_point import encode_fixed, decode_fixed, encode_bias_for_output_scale
from mpc.secure_nn_fixed import secure_two_layer_mlp_secret_fixed


HOST = "127.0.0.1"
PORT = 9000


def main():
    s = socket.socket()
    s.connect((HOST, PORT))

    # 固定点精度
    scale_x_bits = 8
    scale_w1_bits = 8
    scale_w2_bits = 8

    # 浮点输入
    X_plain = np.array([[0.5, -0.25]], dtype=np.float64)

    # 第一层：
    # H[0] = 0.5*1 + (-0.25)*1 = 0.25
    # H[1] = 0.5*1 + (-0.25)*3 = -0.25
    # ReLU(H) = [0.25, 0]
    W1_plain = np.array([
        [1.0, 1.0],
        [1.0, 3.0]
    ], dtype=np.float64)

    b1_plain = np.array([0.0, 0.0], dtype=np.float64)

    # 第二层：
    # Y = [0.25, 0] @ [[4], [7]] = 1.0
    W2_plain = np.array([
        [4.0],
        [7.0]
    ], dtype=np.float64)

    b2_plain = np.array([0.0], dtype=np.float64)

    # 明文期望
    H_expected = X_plain @ W1_plain + b1_plain
    A_expected = np.maximum(H_expected, 0)
    Y_expected = A_expected @ W2_plain + b2_plain

    # 编码
    X = encode_fixed(X_plain, scale_x_bits)
    W1 = encode_fixed(W1_plain, scale_w1_bits)

    # 第一层输出 scale = x_scale + w1_scale
    h_scale_bits = scale_x_bits + scale_w1_bits
    b1 = encode_bias_for_output_scale(b1_plain, h_scale_bits)

    W2 = encode_fixed(W2_plain, scale_w2_bits)

    # 第二层输出 scale = h_scale + w2_scale
    y_scale_bits = h_scale_bits + scale_w2_bits
    b2 = encode_bias_for_output_scale(b2_plain, y_scale_bits)

    # 全部秘密分享
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
        scale_x_bits,
        scale_w1_bits,
        scale_w2_bits
    ))

    # Party0 MPC 推理
    Y0, out_scale_bits = secure_two_layer_mlp_secret_fixed(
        x_i=X0,
        W1_i=W10,
        b1_i=b10,
        W2_i=W20,
        b2_i=b20,
        scale_x_bits=scale_x_bits,
        scale_w1_bits=scale_w1_bits,
        scale_w2_bits=scale_w2_bits,
        conn=s,
        party_id=0
    )

    Y1 = recv_data(s)

    Y_ring = reconstruct_arith(Y0, Y1)
    Y_mpc = decode_fixed(Y_ring, out_scale_bits)

    print("===== Fixed-point Secret MLP Test =====")
    print("X_plain =", X_plain)
    print("W1_plain =")
    print(W1_plain)
    print("H_expected =", H_expected)
    print("ReLU_expected =", A_expected)
    print("W2_plain =")
    print(W2_plain)
    print("Y_expected =", Y_expected)
    print("Y_mpc =", Y_mpc)
    print("output_scale_bits =", out_scale_bits)

    s.close()


if __name__ == "__main__":
    main()