import os
import sys
import socket
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith
from mpc.fixed_point import encode_fixed, decode_fixed
from mpc.triple_pool import setup_triple_pool
from mpc.secure_cnn_sbn_fixed import secure_fixed_cnn_with_sbn


HOST = "127.0.0.1"
PORT = 9000


def plain_conv2d_nchw(x, w, b=None, stride=1, padding=0):
    if isinstance(stride, int):
        stride = (stride, stride)

    if isinstance(padding, int):
        padding = (padding, padding)

    stride_h, stride_w = stride
    pad_h, pad_w = padding

    n, c_in, h, width = x.shape
    c_out, _, k_h, k_w = w.shape

    x_pad = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0.0
    )

    h_pad = h + 2 * pad_h
    w_pad = width + 2 * pad_w

    out_h = (h_pad - k_h) // stride_h + 1
    out_w = (w_pad - k_w) // stride_w + 1

    y = np.zeros((n, c_out, out_h, out_w), dtype=np.float64)

    for ni in range(n):
        for oc in range(c_out):
            for oh in range(out_h):
                for ow in range(out_w):
                    acc = 0.0

                    for ic in range(c_in):
                        for kh in range(k_h):
                            for kw in range(k_w):
                                ih = oh * stride_h + kh
                                iw = ow * stride_w + kw
                                acc += x_pad[ni, ic, ih, iw] * w[oc, ic, kh, kw]

                    if b is not None:
                        acc += b[oc]

                    y[ni, oc, oh, ow] = acc

    return y


def apply_bn_nchw(x, eps1, eps2):
    y = np.zeros_like(x, dtype=np.float64)

    for c in range(x.shape[1]):
        y[:, c, :, :] = eps1[c] * x[:, c, :, :] + eps2[c]

    return y


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    scale_bits = 8
    conv_stride = 1
    conv_padding = 0

    X_plain = np.array(
        [
            [
                [
                    [0.5, 1.0, 1.5],
                    [2.0, 2.5, 3.0],
                    [3.5, 4.0, 4.5]
                ]
            ]
        ],
        dtype=np.float64
    )

    conv_W_plain = np.array(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0]
                ]
            ]
        ],
        dtype=np.float64
    )

    conv_b_plain = np.array([0.25], dtype=np.float64)

    # SBN 参数：
    # z = eps1 * x + eps2
    bn_eps1_plain = np.array([2.0], dtype=np.float64)
    bn_eps2_plain = np.array([-1.0], dtype=np.float64)

    # Conv 输出:
    # [[3.25, 4.25],
    #  [6.25, 7.25]]
    #
    # SBN 后:
    # [[5.5, 7.5],
    #  [11.5, 13.5]]
    #
    # ReLU 后不变。
    #
    # FC:
    # [5.5, 7.5, 11.5, 13.5] @ [1, 0.5, -1, 2] + 0.125
    # = 5.5 + 3.75 - 11.5 + 27 + 0.125
    # = 24.875
    fc_W_plain = np.array(
        [
            [1.0],
            [0.5],
            [-1.0],
            [2.0]
        ],
        dtype=np.float64
    )

    fc_b_plain = np.array([0.125], dtype=np.float64)

    conv_expected = plain_conv2d_nchw(
        x=X_plain,
        w=conv_W_plain,
        b=conv_b_plain,
        stride=conv_stride,
        padding=conv_padding
    )

    bn_expected = apply_bn_nchw(
        x=conv_expected,
        eps1=bn_eps1_plain,
        eps2=bn_eps2_plain
    )

    act_expected = np.maximum(bn_expected, 0)

    flat_expected = act_expected.reshape(act_expected.shape[0], -1)

    Y_expected = flat_expected @ fc_W_plain + fc_b_plain

    X = encode_fixed(X_plain, scale_bits)
    conv_W = encode_fixed(conv_W_plain, scale_bits)
    fc_W = encode_fixed(fc_W_plain, scale_bits)

    # Conv bias 加到乘法累加结果上，所以编码到 2f
    conv_b = encode_fixed(conv_b_plain, scale_bits * 2)

    # SBN eps1 参与乘法，编码到 f
    bn_eps1 = encode_fixed(bn_eps1_plain, scale_bits)

    # SBN eps2 加到 eps1*x 的 2f 结果上，所以编码到 2f
    bn_eps2 = encode_fixed(bn_eps2_plain, scale_bits * 2)

    # FC bias 加到乘法累加结果上，所以编码到 2f
    fc_b = encode_fixed(fc_b_plain, scale_bits * 2)

    X0, X1 = share_arith(X)
    conv_W0, conv_W1 = share_arith(conv_W)
    conv_b0, conv_b1 = share_arith(conv_b)
    bn_eps10, bn_eps11 = share_arith(bn_eps1)
    bn_eps20, bn_eps21 = share_arith(bn_eps2)
    fc_W0, fc_W1 = share_arith(fc_W)
    fc_b0, fc_b1 = share_arith(fc_b)

    conv_mul_shape = (4, 1)
    feature_shape = (1, 1, 2, 2)
    fc_mul_shape = (1, 1)
    fc_out_shape = (1, 1)

    arith_plan = [
        (conv_mul_shape, 40),
        (feature_shape, 520),
        (fc_mul_shape, 80),
        (fc_out_shape, 120),
    ]

    bit_plan = [
        (feature_shape, 1000),
        (fc_out_shape, 180),
    ]

    with time_block("total_time"):

        send_data(s, (
            "FIXED_CNN_SBN_CONFIG",
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
            bn_eps11,
            bn_eps21,
            fc_W1,
            fc_b1,
            scale_bits,
            conv_stride,
            conv_padding
        ))

        with time_block("online_time"):
            Y0 = secure_fixed_cnn_with_sbn(
                x_i=X0,
                conv_w_i=conv_W0,
                conv_b_i=conv_b0,
                bn_eps1_i=bn_eps10,
                bn_eps2_i=bn_eps20,
                fc_w_i=fc_W0,
                fc_b_i=fc_b0,
                scale_bits=scale_bits,
                conn=s,
                party_id=0,
                conv_stride=conv_stride,
                conv_padding=conv_padding
            )

            Y1 = recv_data(s)

    Y_ring = reconstruct_arith(Y0, Y1)
    Y_mpc = decode_fixed(Y_ring, scale_bits)

    diff = np.abs(Y_mpc - Y_expected)

    print("===== Fixed-point Secret CNN with SBN Test =====")
    print("X_plain =")
    print(X_plain)
    print("conv_expected =")
    print(conv_expected)
    print("bn_expected =")
    print(bn_expected)
    print("act_expected =")
    print(act_expected)
    print("flat_expected =")
    print(flat_expected)
    print("Y_expected =")
    print(Y_expected)
    print("Y_mpc =")
    print(Y_mpc)
    print("abs_error =")
    print(diff)

    if np.all(diff <= 1.0 / (1 << scale_bits)):
        print("Fixed-point Secret CNN with SBN test PASSED ✅")
    else:
        print("Fixed-point Secret CNN with SBN test FAILED ❌")

    print_report("Party0 Fixed-point Secret CNN with SBN Profiler")

    s.close()


if __name__ == "__main__":
    main()