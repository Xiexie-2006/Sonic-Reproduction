def _to_pair(x):
    """
    将 stride / padding 统一转换成二元组。
    """
    if isinstance(x, int):
        return x, x

    if isinstance(x, tuple) or isinstance(x, list):
        if len(x) != 2:
            raise ValueError(f"Expected pair, got {x}")
        return int(x[0]), int(x[1])

    raise TypeError(f"Unsupported value type: {type(x)}")


def build_linear_matmul_plan(x_shape, w_shape, count=1):
    """
    根据线性层矩阵乘法形状生成 matmul_plan。

    对于安全全连接层：
        Y = X @ W

    需要矩阵 triple：
        A.shape = X.shape
        B.shape = W.shape
        C = A @ B

    参数：
        x_shape: X 的 shape，例如 (batch, in_dim)
        w_shape: W 的 shape，例如 (in_dim, out_dim)
        count  : 需要几个矩阵 triple

    返回：
        [((batch, in_dim), (in_dim, out_dim), count)]
    """
    x_shape = tuple(x_shape)
    w_shape = tuple(w_shape)

    if len(x_shape) != 2 or len(w_shape) != 2:
        raise ValueError(
            f"linear matmul expects 2D shapes, got x_shape={x_shape}, w_shape={w_shape}"
        )

    if x_shape[1] != w_shape[0]:
        raise ValueError(
            f"invalid linear matmul shapes: x_shape={x_shape}, w_shape={w_shape}"
        )

    return [
        (x_shape, w_shape, count)
    ]


def conv2d_output_shape(x_shape, w_shape, stride=1, padding=0):
    """
    根据 NCHW 输入和卷积核形状计算 Conv2D 输出形状。

    x_shape:
        (N, C_in, H, W)

    w_shape:
        (C_out, C_in, kH, kW)

    返回：
        (N, C_out, out_h, out_w)
    """
    x_shape = tuple(x_shape)
    w_shape = tuple(w_shape)

    if len(x_shape) != 4:
        raise ValueError(f"x_shape must be NCHW 4D, got {x_shape}")

    if len(w_shape) != 4:
        raise ValueError(f"w_shape must be OIHW 4D, got {w_shape}")

    n, c_in, h, width = x_shape
    c_out, c_in_w, k_h, k_w = w_shape

    if c_in != c_in_w:
        raise ValueError(
            f"input channel mismatch: x_shape={x_shape}, w_shape={w_shape}"
        )

    stride_h, stride_w = _to_pair(stride)
    pad_h, pad_w = _to_pair(padding)

    h_pad = h + 2 * pad_h
    w_pad = width + 2 * pad_w

    out_h = (h_pad - k_h) // stride_h + 1
    out_w = (w_pad - k_w) // stride_w + 1

    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"invalid conv output size: out_h={out_h}, out_w={out_w}"
        )

    return n, c_out, out_h, out_w


def build_conv2d_matmul_plan(x_shape, w_shape, stride=1, padding=0, count=1):
    """
    根据 Conv2D 的输入形状和卷积核形状，生成 im2col 后的 matmul_plan。

    Conv2D 通过 im2col 会变成：
        Y_col = X_col @ W_mat

    其中：
        X_col.shape = (N * out_h * out_w, C_in * kH * kW)
        W_mat.shape = (C_in * kH * kW, C_out)

    所以需要矩阵 triple：
        A.shape = X_col.shape
        B.shape = W_mat.shape
        C = A @ B

    返回：
        [((N*out_h*out_w, C_in*kH*kW), (C_in*kH*kW, C_out), count)]
    """
    x_shape = tuple(x_shape)
    w_shape = tuple(w_shape)

    n, c_in, h, width = x_shape
    c_out, c_in_w, k_h, k_w = w_shape

    if c_in != c_in_w:
        raise ValueError(
            f"input channel mismatch: x_shape={x_shape}, w_shape={w_shape}"
        )

    _, _, out_h, out_w = conv2d_output_shape(
        x_shape=x_shape,
        w_shape=w_shape,
        stride=stride,
        padding=padding,
    )

    x_col_shape = (
        n * out_h * out_w,
        c_in * k_h * k_w,
    )

    w_mat_shape = (
        c_in * k_h * k_w,
        c_out,
    )

    return [
        (x_col_shape, w_mat_shape, count)
    ]


def merge_matmul_plans(*plans):
    """
    合并多个 matmul_plan。

    如果两个 plan 的 x_shape 和 w_shape 完全相同，则合并 count。
    """
    merged = {}

    for plan in plans:
        if plan is None:
            continue

        for x_shape, w_shape, count in plan:
            key = (tuple(x_shape), tuple(w_shape))
            merged[key] = merged.get(key, 0) + int(count)

    result = []

    for (x_shape, w_shape), count in merged.items():
        result.append((x_shape, w_shape, count))

    return result