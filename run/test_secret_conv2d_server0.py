import socket
import numpy as np

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith, MOD
from mpc.triple_pool import setup_triple_pool
from mpc.conv2d_secret import conv2d_secret


HOST = "127.0.0.1"
PORT = 9000


def to_ring(x):
    return (np.array(x, dtype=np.int64) % MOD).astype(np.uint32)


def decode_signed(x):
    return x.astype(np.uint32).view(np.int32)


def plain_conv2d_nchw(x, w, b=None, stride=1, padding=0):
    """
    明文 Conv2D，用于对比测试。
    x: [N, C_in, H, W]
    w: [C_out, C_in, kH, kW]
    """

    if isinstance(stride, int):
        stride = (stride, stride)

    if isinstance(padding, int):
        padding = (padding, padding)

    stride_h, stride_w = stride
    pad_h, pad_w = padding

    n, c_in, h, ww = x.shape
    c_out, _, k_h, k_w = w.shape

    x_pad = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0
    )

    h_pad = h + 2 * pad_h
    w_pad = ww + 2 * pad_w

    out_h = (h_pad - k_h) // stride_h + 1
    out_w = (w_pad - k_w) // stride_w + 1

    y = np.zeros((n, c_out, out_h, out_w), dtype=np.int64)

    for ni in range(n):
        for oc in range(c_out):
            for oh in range(out_h):
                for ow in range(out_w):
                    acc = 0

                    for ic in range(c_in):
                        for kh in range(k_h):
                            for kw in range(k_w):
                                ih = oh * stride_h + kh
                                iw = ow * stride_w + kw

                                acc += int(x_pad[ni, ic, ih, iw]) * int(w[oc, ic, kh, kw])

                    if b is not None:
                        acc += int(b[oc])

                    y[ni, oc, oh, ow] = acc

    return y


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    # 输入: N=1, C=1, H=3, W=3
    X_plain = np.array([
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        ]
    ], dtype=np.int64)

    # 卷积核: C_out=1, C_in=1, kH=2, kW=2
    # kernel =
    # [[1, 0],
    #  [0, 1]]
    W_plain = np.array([
        [
            [
                [1, 0],
                [0, 1]
            ]
        ]
    ], dtype=np.int64)

    b_plain = np.array([1], dtype=np.int64)

    stride = 1
    padding = 0

    Y_expected = plain_conv2d_nchw(
        x=X_plain,
        w=W_plain,
        b=b_plain,
        stride=stride,
        padding=padding
    )

    X = to_ring(X_plain)
    W = to_ring(W_plain)
    b = to_ring(b_plain)

    X0, X1 = share_arith(X)
    W0, W1 = share_arith(W)
    b0, b1 = share_arith(b)

    # 当前 Conv2D 转换后：
    # im2col rows = 1 * 2 * 2 = 4
    # out_channels = 1
    # 每个 kernel 有 4 个乘法项
    # 所以 secure_mul shape 为 (4, 1)
    arith_plan = [
        ((4, 1), 30),
    ]

    bit_plan = []

    with time_block("total_time"):

        send_data(s, (
            "CONV2D_CONFIG",
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
            W1,
            b1,
            stride,
            padding
        ))

        with time_block("online_time"):
            Y0 = conv2d_secret(
                x_i=X0,
                w_i=W0,
                b_i=b0,
                conn=s,
                party_id=0,
                stride=stride,
                padding=padding
            )

            Y1 = recv_data(s)

    Y = reconstruct_arith(Y0, Y1)
    Y_signed = decode_signed(Y)

    print("===== Secret Conv2D Test =====")
    print("X_plain =")
    print(X_plain)
    print("W_plain =")
    print(W_plain)
    print("b_plain =", b_plain)
    print("Y_expected =")
    print(Y_expected)
    print("Y_mpc =")
    print(Y_signed)

    if np.array_equal(Y_signed, Y_expected):
        print("Secret Conv2D test PASSED ✅")
    else:
        print("Secret Conv2D test FAILED ❌")

    print_report("Party0 Secret Conv2D Profiler")

    s.close()


if __name__ == "__main__":
    main()