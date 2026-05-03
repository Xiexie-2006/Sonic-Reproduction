import numpy as np

from mpc.share import MOD
from mpc.compare import secure_compare_zero
from mpc.b2a_secure import b2a_secure
from mpc.triple import get_arith_triple
from mpc.mul import secure_mul


def _pair(v):
    # 将 int 形式参数统一转成二维 tuple。
    # 例如 kernel_size=2 会变成 (2, 2)，方便同时处理高和宽。
    if isinstance(v, tuple):
        return v
    return (v, v)


def secure_max2(x_i, y_i, conn, party_id):
    """
    Secure max of two arithmetic shares.

        max(x, y) = y + gate * (x - y)

    gate = 1 if x >= y else 0

    输入:
        x_i, y_i:
            当前方持有的 arithmetic share

    输出:
        max(x, y) 的 arithmetic share

    这个函数用于在秘密分享状态下比较两个数，并返回其中较大的一个。
    MaxPool 本质上就是窗口内不断取最大值，所以这里先实现两个数的安全 max。
    """

    # diff = x - y。
    # 如果 diff >= 0，说明 x >= y；否则说明 y 更大。
    diff_i = (
        x_i.astype(np.uint64)
        - y_i.astype(np.uint64)
    ) % MOD
    diff_i = diff_i.astype(np.uint32)

    # 安全比较 diff 是否非负。
    # 返回的是 Boolean share：gate = 1 表示选择 x，gate = 0 表示选择 y。
    gate_bool_i = secure_compare_zero(
        xi=diff_i,
        conn=conn,
        party_id=party_id
    )

    # 后面 gate 要参与乘法，所以需要从 Boolean share 转成 Arithmetic share。
    gate_arith_i = b2a_secure(
        xb_i=gate_bool_i,
        conn=conn,
        party_id=party_id
    )

    # 计算 gate * diff 需要安全乘法，因此获取 Beaver triple。
    triple_i = get_arith_triple(
        conn=conn,
        party_id=party_id,
        shape=diff_i.shape
    )

    # selected_diff = gate * (x - y)。
    # gate=1 时 selected_diff = x-y；
    # gate=0 时 selected_diff = 0。
    selected_diff_i = secure_mul(
        xi=gate_arith_i,
        yi=diff_i,
        triple_i=triple_i,
        conn=conn,
        party_id=party_id
    )

    # max(x, y) = y + gate * (x - y)。
    # 如果 x 更大，结果变成 x；否则结果保持 y。
    out_i = (
        y_i.astype(np.uint64)
        + selected_diff_i.astype(np.uint64)
    ) % MOD

    return out_i.astype(np.uint32)


def maxpool2d_secret(
    x_i,
    kernel_size=2,
    stride=2,
    padding=0,
    conn=None,
    party_id=0
):
    """
    Secure MaxPool2D for NCHW tensor.

    输入:
        x_i: arithmetic share, shape = [N, C, H, W]

    输出:
        y_i: arithmetic share, shape = [N, C, out_h, out_w]

    当前版本:
        支持 padding = 0
        支持 kernel_size / stride 为 int 或 tuple

    说明:
        MaxPool 是非线性操作，需要 secure comparison。

    这里的实现方式比较直接：
        对池化窗口中的每个位置依次取 candidate，
        然后用 secure_max2 和当前最大值比较，
        最后得到每个窗口的最大值 share。
    """

    # 当前版本暂时不支持 padding。
    # 如果要支持 padding，需要额外考虑边界值在秘密分享下如何表示。
    if padding != 0:
        raise NotImplementedError(
            "Current secure MaxPool2D supports padding=0 only."
        )

    # 统一 kernel_size 和 stride 的二维形式。
    k_h, k_w = _pair(kernel_size)
    s_h, s_w = _pair(stride)

    # 输入为 NCHW 格式。
    n, c, h, w = x_i.shape

    # 按照普通池化输出尺寸公式计算输出高宽。
    out_h = (h - k_h) // s_h + 1
    out_w = (w - k_w) // s_w + 1

    # current 用来保存当前窗口内已经比较出的最大值。
    # 第一次遍历窗口元素时直接赋值，后续再逐个比较。
    current = None

    for kh in range(k_h):
        for kw in range(k_w):
            # candidate 保存当前窗口位置对应的所有输出点取值。
            # 形状和最终池化输出一致。
            candidate = np.zeros(
                (n, c, out_h, out_w),
                dtype=np.uint32
            )

            for oh in range(out_h):
                for ow in range(out_w):
                    # 当前输出位置 (oh, ow) 对应输入中的位置。
                    ih = oh * s_h + kh
                    iw = ow * s_w + kw
                    candidate[:, :, oh, ow] = x_i[:, :, ih, iw]

            # 第一个窗口位置直接作为初始最大值。
            if current is None:
                current = candidate
            else:
                # 后续窗口位置和 current 做安全比较，
                # 不断更新窗口最大值。
                current = secure_max2(
                    x_i=current,
                    y_i=candidate,
                    conn=conn,
                    party_id=party_id
                )

    return current.astype(np.uint32)