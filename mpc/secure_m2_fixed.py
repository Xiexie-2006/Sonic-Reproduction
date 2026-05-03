from mpc.conv2d_fixed import conv2d_secret_fixed
from mpc.sbn import sbn_fixed
from mpc.maxpool2d_secret import maxpool2d_secret
from mpc.srelu import srelu
from mpc.linear_secret import linear_secret_weight
from mpc.trunc import secure_trunc


def flatten_nchw(x_i):
    """
    [N, C, H, W] -> [N, C*H*W]

    M2 是 CNN 结构，卷积和池化结束后需要接全连接层，
    所以这里把 NCHW 特征图拉平成二维矩阵。
    对 secret share 来说，reshape 是本地操作，不需要通信。
    """
    n = x_i.shape[0]
    return x_i.reshape(n, -1)


def secure_m2_fixed_inference(
    x_i,

    conv1_w_i,
    conv1_b_i,
    bn1_eps1_i,
    bn1_eps2_i,

    conv2_w_i,
    conv2_b_i,
    bn2_eps1_i,
    bn2_eps2_i,

    fc1_w_i,
    fc1_b_i,
    bn3_eps1_i,
    bn3_eps2_i,

    fc2_w_i,
    fc2_b_i,
    bn4_eps1_i,
    bn4_eps2_i,

    scale_bits,
    conn,
    party_id
):
    """
    Sonic paper M2 fixed-point MPC inference.

    M2 structure:

        Conv1
        → SBN1
        → SMP1
        → SReLU1

        Conv2
        → SBN2
        → SMP2
        → SReLU2

        Flatten
        → SFC1
        → SBN3
        → SReLU3

        → SFC2
        → SBN4

    使用 Sonic 优化顺序：
        ReLU → MaxPool
        优化为
        MaxPool → ReLU

    M2 是 Sonic 论文里 MNIST 上的卷积模型。
    相比 M1，它多了安全卷积和安全池化，因此更适合验证 CNN 类网络的 MPC 推理流程。
    """

    # ------------------------------------------------------------
    # Block 1:
    # Conv1: [N,1,28,28] -> [N,16,24,24]
    # SBN1
    # MaxPool: [N,16,24,24] -> [N,16,12,12]
    # SReLU1
    # ------------------------------------------------------------

    # 第一层安全定点卷积。
    # kernel=5、padding=0 时，MNIST 的 28x28 会变成 24x24。
    h1_conv_i = conv2d_secret_fixed(
        x_i=x_i,
        w_i=conv1_w_i,
        b_i=conv1_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=0
    )

    # 第一层安全 BN。
    # SBN 内部会计算 eps1*x + eps2，并通过 secure_trunc 恢复 scale。
    h1_bn_i = sbn_fixed(
        x_i=h1_conv_i,
        eps1_i=bn1_eps1_i,
        eps2_i=bn1_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # Sonic 优化顺序：先 MaxPool，再 SReLU。
    # 这样可以减少后续 SReLU 需要处理的元素数量。
    h1_pool_i = maxpool2d_secret(
        x_i=h1_bn_i,
        kernel_size=2,
        stride=2,
        padding=0,
        conn=conn,
        party_id=party_id
    )

    # 对池化后的结果做安全激活。
    h1_act_i = srelu(
        xi=h1_pool_i,
        conn=conn,
        party_id=party_id
    )

    # ------------------------------------------------------------
    # Block 2:
    # Conv2: [N,16,12,12] -> [N,16,8,8]
    # SBN2
    # MaxPool: [N,16,8,8] -> [N,16,4,4]
    # SReLU2
    # ------------------------------------------------------------

    # 第二层安全定点卷积。
    # 输入通道为 16，输出通道仍为 16，空间尺寸从 12x12 变成 8x8。
    h2_conv_i = conv2d_secret_fixed(
        x_i=h1_act_i,
        w_i=conv2_w_i,
        b_i=conv2_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=0
    )

    # 第二层安全 BN。
    h2_bn_i = sbn_fixed(
        x_i=h2_conv_i,
        eps1_i=bn2_eps1_i,
        eps2_i=bn2_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # 第二次安全池化。
    # 16x8x8 会变成 16x4x4。
    h2_pool_i = maxpool2d_secret(
        x_i=h2_bn_i,
        kernel_size=2,
        stride=2,
        padding=0,
        conn=conn,
        party_id=party_id
    )

    # 安全激活。
    h2_act_i = srelu(
        xi=h2_pool_i,
        conn=conn,
        party_id=party_id
    )

    # ------------------------------------------------------------
    # FC block:
    # Flatten: [N,16,4,4] -> [N,256]
    # FC1: 256 -> 100
    # SBN3
    # SReLU3
    # FC2: 100 -> 10
    # SBN4
    # ------------------------------------------------------------

    # 展平卷积特征，接入全连接层。
    flat_i = flatten_nchw(h2_act_i)

    # 第一层安全全连接：256 -> 100。
    # fixed-point 乘法后 scale 会临时变为 2^(2f)。
    fc1_raw_i = linear_secret_weight(
        x_i=flat_i,
        w_i=fc1_w_i,
        b_i=fc1_b_i,
        conn=conn,
        party_id=party_id
    )

    # FC1 后截断，恢复到 2^f。
    fc1_i = secure_trunc(
        x_i=fc1_raw_i,
        shift_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # SBN3。
    fc1_bn_i = sbn_fixed(
        x_i=fc1_i,
        eps1_i=bn3_eps1_i,
        eps2_i=bn3_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # SReLU3。
    fc1_act_i = srelu(
        xi=fc1_bn_i,
        conn=conn,
        party_id=party_id
    )

    # 第二层安全全连接：100 -> 10，输出分类 logits。
    fc2_raw_i = linear_secret_weight(
        x_i=fc1_act_i,
        w_i=fc2_w_i,
        b_i=fc2_b_i,
        conn=conn,
        party_id=party_id
    )

    # FC2 后截断。
    fc2_i = secure_trunc(
        x_i=fc2_raw_i,
        shift_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # 最后一层 SBN4，输出最终 logits share。
    out_i = sbn_fixed(
        x_i=fc2_i,
        eps1_i=bn4_eps1_i,
        eps2_i=bn4_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    return out_i