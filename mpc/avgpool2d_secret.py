import numpy as np

from mpc.share import MOD
from mpc.trunc import secure_trunc


def _pair(v):
    # 如果传入的是单个整数，就扩展成二维形式。
    # 例如 kernel_size=2 会变成 (2, 2)。
    if isinstance(v, tuple):
        return v
    return (v, v)


def _is_power_of_two(x):
    # 判断 x 是否为 2 的幂。
    # AvgPool 中如果窗口面积是 2 的幂，就可以用右移实现除法。
    return x > 0 and (x & (x - 1)) == 0


def _log2_int(x):
    # 计算整数 x 的 log2。
    # 这里默认 x 已经是 2 的幂，例如 4 -> 2，16 -> 4。
    r = 0
    while x > 1:
        x >>= 1
        r += 1
    return r


def avgpool2d_secret(
    x_i,
    kernel_size=2,
    stride=2,
    padding=0,
    conn=None,
    party_id=0
):
    """
    Secure AvgPool2D for NCHW tensor.

    输入:
        x_i: arithmetic share, shape = [N, C, H, W]

    输出:
        y_i: arithmetic share, shape = [N, C, out_h, out_w]

    当前版本:
        支持 padding = 0
        支持 kernel_area 为 2 的幂，例如 2x2, 4x4

    实现:
        1. 对窗口内元素本地求和
        2. 使用 secure_trunc 安全除以 kernel_area

    平均池化本质上是窗口求和再除以窗口面积。
    在 arithmetic share 下，加法可以直接本地完成；
    但除法需要谨慎处理，所以这里限制窗口面积必须是 2 的幂，
    这样可以通过 secure_trunc 来完成安全右移。
    """

    # 当前实现没有处理 padding。
    # 如果后续需要支持 padding，需要先在秘密分享状态下补零或处理边界。
    if padding != 0:
        raise NotImplementedError(
            "Current secure AvgPool2D supports padding=0 only."
        )

    # 将 kernel_size 和 stride 统一整理成二维形式。
    k_h, k_w = _pair(kernel_size)
    s_h, s_w = _pair(stride)

    # 池化窗口面积，例如 2x2 的面积为 4。
    kernel_area = k_h * k_w

    # 这里只支持窗口面积为 2 的幂。
    # 因为除以 4、8、16 这类数可以转换成右移操作，
    # 更容易和 fixed-point / secret truncation 对齐。
    if not _is_power_of_two(kernel_area):
        raise NotImplementedError(
            "Current AvgPool2D only supports power-of-two kernel area."
        )

    # 除以 kernel_area 等价于右移 shift_bits 位。
    shift_bits = _log2_int(kernel_area)

    # 输入张量格式为 NCHW。
    n, c, h, w = x_i.shape

    # 根据普通卷积/池化输出尺寸公式计算输出高宽。
    out_h = (h - k_h) // s_h + 1
    out_w = (w - k_w) // s_w + 1

    # 先保存每个池化窗口的求和结果。
    # 加法在 arithmetic share 下可以本地做，不需要交互。
    sum_i = np.zeros((n, c, out_h, out_w), dtype=np.uint32)

    for kh in range(k_h):
        for kw in range(k_w):
            for oh in range(out_h):
                for ow in range(out_w):
                    # 当前输出位置 (oh, ow) 对应到输入窗口中的实际坐标。
                    ih = oh * s_h + kh
                    iw = ow * s_w + kw

                    # 对窗口内元素做模 MOD 加法。
                    # 使用 uint64 是为了避免 uint32 加法中间溢出影响结果。
                    sum_i[:, :, oh, ow] = (
                        sum_i[:, :, oh, ow].astype(np.uint64)
                        + x_i[:, :, ih, iw].astype(np.uint64)
                    ) % MOD

    sum_i = sum_i.astype(np.uint32)

    # 对窗口和进行安全截断，相当于除以 kernel_area。
    # 例如 2x2 AvgPool 的 kernel_area=4，对应右移 2 位。
    y_i = secure_trunc(
        x_i=sum_i,
        shift_bits=shift_bits,
        conn=conn,
        party_id=party_id
    )

    return y_i