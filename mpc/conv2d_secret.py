import numpy as np

from mpc.linear_secret import linear_secret_weight


def _pair(v):
    # 将整数形式的 stride、padding 或 kernel_size 转成二维形式。
    # 例如 stride=1 会变成 (1, 1)，方便后续统一处理高和宽。
    if isinstance(v, tuple):
        return v
    return (v, v)


def im2col_nchw(x, kernel_size, stride=1, padding=0):
    """
    NCHW 格式 im2col。

    输入：
        x: [N, C, H, W]

    输出：
        cols: [N*out_h*out_w, C*kH*kW]

    说明：
        im2col 是线性重排操作。
        对 secret share 来说，每一方可以对自己的 share 本地做 im2col。

    im2col 的作用是把卷积窗口展开成矩阵行。
    这样卷积计算就可以转化为矩阵乘法，后面直接复用安全线性层即可。
    """

    # 统一整理 stride、padding、kernel_size 的格式。
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    k_h, k_w = _pair(kernel_size)

    n, c, h, w = x.shape

    # 对输入做 padding。
    # padding 是线性操作，对秘密分享来说可以本地完成。
    # 当前补的是公开的 0，不会引入额外交互。
    x_pad = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0
    )

    h_pad = h + 2 * pad_h
    w_pad = w + 2 * pad_w

    # 根据卷积输出尺寸公式计算输出高宽。
    out_h = (h_pad - k_h) // stride_h + 1
    out_w = (w_pad - k_w) // stride_w + 1

    cols = []

    for ni in range(n):
        for oh in range(out_h):
            for ow in range(out_w):
                patch = []

                # 按照 NCHW 顺序遍历当前卷积窗口。
                # 每个 patch 最后会变成 im2col 矩阵中的一行。
                for ci in range(c):
                    for kh in range(k_h):
                        for kw in range(k_w):
                            ih = oh * stride_h + kh
                            iw = ow * stride_w + kw
                            patch.append(x_pad[ni, ci, ih, iw])

                cols.append(patch)

    # cols 的每一行对应一个卷积窗口展开后的向量。
    return np.array(cols, dtype=np.uint32), out_h, out_w


def weight_to_matrix_nchw(w):
    """
    卷积核转矩阵。

    输入：
        w: [out_channels, in_channels, kH, kW]

    输出：
        W_mat: [in_channels*kH*kW, out_channels]

    这个函数把卷积核整理成矩阵乘法需要的形式。
    im2col 后的输入是 [窗口数量, in_channels*kH*kW]，
    所以权重矩阵需要变成 [in_channels*kH*kW, out_channels]。
    """

    out_c, in_c, k_h, k_w = w.shape

    rows = []

    for oc in range(out_c):
        kernel = []

        # 按照和 im2col 相同的通道、卷积核顺序展开权重。
        # 这样输入窗口向量和权重向量才能一一对应。
        for ic in range(in_c):
            for kh in range(k_h):
                for kw in range(k_w):
                    kernel.append(w[oc, ic, kh, kw])

        rows.append(kernel)

    # 当前 rows 是 [out_c, in_c*kH*kW]
    # 线性层需要 [in_dim, out_dim]
    return np.array(rows, dtype=np.uint32).T


def conv2d_secret(
    x_i,
    w_i,
    b_i,
    conn,
    party_id,
    stride=1,
    padding=0
):
    """
    Secret Conv2D:

        Y = Conv2D(X, W) + b

    输入：
        x_i: 当前方持有的输入 share
             shape = [N, C_in, H, W]

        w_i: 当前方持有的卷积核 share
             shape = [C_out, C_in, kH, kW]

        b_i: 当前方持有的 bias share
             shape = [C_out]

    输出：
        y_i: 当前方持有的输出 share
             shape = [N, C_out, out_h, out_w]

    实现思路：
        1. 对输入 share 做 im2col
        2. 把卷积核 share 展开成矩阵
        3. 调用安全线性层完成矩阵乘法
        4. 再把输出 reshape 回 NCHW 格式
    """

    out_c, in_c, k_h, k_w = w_i.shape

    # 将输入特征图展开成二维矩阵。
    # 每一行对应一个卷积窗口。
    x_col_i, out_h, out_w = im2col_nchw(
        x=x_i,
        kernel_size=(k_h, k_w),
        stride=stride,
        padding=padding
    )

    # 将卷积核展开成矩阵。
    # 展开顺序必须和 im2col 中 patch 的展开顺序一致。
    w_mat_i = weight_to_matrix_nchw(w_i)

    # [N*out_h*out_w, C_out]
    #
    # 卷积此时已经被转化为矩阵乘法，
    # 因此可以直接复用 linear_secret_weight。
    y_col_i = linear_secret_weight(
        x_i=x_col_i,
        w_i=w_mat_i,
        b_i=b_i,
        conn=conn,
        party_id=party_id
    )

    n = x_i.shape[0]

    # 先恢复成 [N, out_h, out_w, C_out]。
    # 这是因为 im2col 展开时输出位置在前，输出通道在后。
    y_nhwc = y_col_i.reshape(n, out_h, out_w, out_c)

    # 转回 NCHW: [N, C_out, out_h, out_w]
    # 这样可以和 PyTorch Conv2d 的输出格式保持一致。
    y_i = np.transpose(y_nhwc, (0, 3, 1, 2)).astype(np.uint32)

    return y_i