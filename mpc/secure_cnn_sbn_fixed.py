from mpc.conv2d_fixed import conv2d_secret_fixed
from mpc.sbn import sbn_fixed
from mpc.srelu import srelu
from mpc.linear_secret import linear_secret_weight
from mpc.trunc import secure_trunc


def flatten_nchw(x_i):
    """
    [N, C, H, W] -> [N, C*H*W]

    这里用于把卷积层输出转成全连接层可以接收的二维输入。
    对秘密分享数据来说，reshape 是本地操作，不需要额外交互。
    """
    n = x_i.shape[0]
    return x_i.reshape(n, -1)


def secure_fixed_cnn_with_sbn(
    x_i,
    conv_w_i,
    conv_b_i,
    bn_eps1_i,
    bn_eps2_i,
    fc_w_i,
    fc_b_i,
    scale_bits,
    conn,
    party_id,
    conv_stride=1,
    conv_padding=0
):
    """
    Fixed-point Secret CNN with SBN:

        Conv2D
        → Secure Truncation
        → SBN
        → SReLU
        → Flatten
        → Secret Linear
        → Secure Truncation

    说明：
        conv2d_secret_fixed 输出 scale = 2^f
        sbn_fixed 输出 scale = 2^f
        srelu 不改变 scale
        linear_secret_weight 输出 scale = 2^(2f)
        最后 secure_trunc 恢复到 scale = 2^f

    这个版本相比基础 CNN 多了 SBN 层，
    用来验证安全卷积输出接安全 BN 时 fixed-point 尺度是否能正确保持。
    """

    # 安全定点卷积。
    # 卷积本身是乘加操作，所以内部已经做过一次 secure_trunc。
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

    # 安全 BatchNorm。
    # SBN 将推理阶段 BN 折叠成 z = eps1 * x + eps2 的形式。
    bn_y_i = sbn_fixed(
        x_i=conv_y_i,
        eps1_i=bn_eps1_i,
        eps2_i=bn_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # 安全激活函数。
    # 这里对 BN 输出做 SReLU，和普通 CNN 中 BN 后接 ReLU 的结构对应。
    act_i = srelu(
        xi=bn_y_i,
        conn=conn,
        party_id=party_id
    )

    # Flatten 后送入全连接层。
    flat_i = flatten_nchw(act_i)

    # 安全全连接。
    # 输入和权重都是 2^f 尺度，乘法结果会临时变成 2^(2f)。
    fc_raw_i = linear_secret_weight(
        x_i=flat_i,
        w_i=fc_w_i,
        b_i=fc_b_i,
        conn=conn,
        party_id=party_id
    )

    # 全连接层后截断，恢复 fixed-point 尺度。
    out_i = secure_trunc(
        x_i=fc_raw_i,
        shift_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    return out_i