from mpc.conv2d_fixed import conv2d_secret_fixed
from mpc.srelu import srelu
from mpc.maxpool2d_secret import maxpool2d_secret
from mpc.linear_secret import linear_secret_weight
from mpc.trunc import secure_trunc


def flatten_nchw(x_i):
    """
    [N, C, H, W] -> [N, C*H*W]

    Flatten 只是改变张量形状，不涉及真实数值计算。
    对 secret share 来说，每一方可以直接在本地 reshape，不需要通信。
    """
    n = x_i.shape[0]
    return x_i.reshape(n, -1)


def secure_fixed_cnn_with_maxpool(
    x_i,
    conv_w_i,
    conv_b_i,
    fc_w_i,
    fc_b_i,
    scale_bits,
    conn,
    party_id,
    conv_stride=1,
    conv_padding=0,
    pool_kernel_size=2,
    pool_stride=2
):
    """
    Fixed-point Secret CNN with Secure MaxPool2D:

        Conv2D
        → Secure Truncation
        → SReLU
        → Secure MaxPool2D
        → Flatten
        → Secret Linear
        → Secure Truncation

    这个函数是在基础 fixed-point CNN 上加入 MaxPool 的版本。
    主要用于验证卷积、激活、池化、全连接这些模块能否在 MPC 中完整串起来。
    """

    # 卷积层。
    # conv2d_secret_fixed 内部会先做安全卷积，再做 secure_trunc，
    # 因此这里输出的 conv_y_i 已经恢复到 scale = 2^f。
    conv_y_i = conv2d_secret_fixed(
        x_i=x_i,
        w_i=conv_w_i,
        b_i=conv_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=conv_stride,
        padding=conv_padding
    )

    # 安全 ReLU。
    # SReLU 根据符号位决定是否保留原值，不会改变 fixed-point 的缩放尺度。
    act_i = srelu(
        xi=conv_y_i,
        conn=conn,
        party_id=party_id
    )

    # 安全最大池化。
    # MaxPool 是非线性操作，内部需要安全比较来选出窗口最大值。
    pool_i = maxpool2d_secret(
        x_i=act_i,
        kernel_size=pool_kernel_size,
        stride=pool_stride,
        padding=0,
        conn=conn,
        party_id=party_id
    )

    # 将池化后的卷积特征拉平成二维矩阵，准备接全连接层。
    flat_i = flatten_nchw(pool_i)

    # 安全全连接层。
    # 这里输入和权重都是 fixed-point 编码，乘法后 scale 会变成 2^(2f)。
    fc_raw_i = linear_secret_weight(
        x_i=flat_i,
        w_i=fc_w_i,
        b_i=fc_b_i,
        conn=conn,
        party_id=party_id
    )

    # 全连接层后做安全截断，把 scale 从 2^(2f) 恢复到 2^f。
    out_i = secure_trunc(
        x_i=fc_raw_i,
        shift_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    return out_i