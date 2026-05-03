from mpc.conv2d_fixed import conv2d_secret_fixed
from mpc.sbn import sbn_fixed
from mpc.maxpool2d_secret import maxpool2d_secret
from mpc.srelu import srelu
from mpc.linear_secret import linear_secret_weight
from mpc.trunc import secure_trunc


def flatten_nchw(x_i):
    # NCHW -> [N, C*H*W]。
    # flatten 只是改变视图形状，不需要额外的安全协议。
    n = x_i.shape[0]
    return x_i.reshape(n, -1)


def secure_c2_fixed_inference(
    x_i,

    conv1_w_i,
    conv1_b_i,
    bn1_eps1_i,
    bn1_eps2_i,

    conv2_w_i,
    conv2_b_i,
    bn2_eps1_i,
    bn2_eps2_i,

    conv3_w_i,
    conv3_b_i,
    bn3_eps1_i,
    bn3_eps2_i,

    conv4_w_i,
    conv4_b_i,
    bn4_eps1_i,
    bn4_eps2_i,

    conv5_w_i,
    conv5_b_i,
    bn5_eps1_i,
    bn5_eps2_i,

    conv6_w_i,
    conv6_b_i,
    bn6_eps1_i,
    bn6_eps2_i,

    conv7_w_i,
    conv7_b_i,
    bn7_eps1_i,
    bn7_eps2_i,

    conv8_w_i,
    conv8_b_i,
    bn8_eps1_i,
    bn8_eps2_i,

    conv9_w_i,
    conv9_b_i,
    bn9_eps1_i,
    bn9_eps2_i,

    fc1_w_i,
    fc1_b_i,
    bn10_eps1_i,
    bn10_eps2_i,

    scale_bits,
    conn,
    party_id
):
    """
    Sonic C2 fixed-point MPC inference.

    C2:
        Conv1 -> SBN1 -> SReLU1
        Conv2 -> SBN2 -> SReLU2
        Conv3 -> SBN3 -> MaxPool1 -> SReLU3

        Conv4 -> SBN4 -> SReLU4
        Conv5 -> SBN5 -> SReLU5
        Conv6 -> SBN6 -> MaxPool2 -> SReLU6

        Conv7 -> SBN7 -> SReLU7
        Conv8 -> SBN8 -> SReLU8
        Conv9 -> SBN9 -> MaxPool3 -> SReLU9

        Flatten
        FC1 -> SBN10

    这个函数实现的是 Sonic 论文中 C2 网络的 fixed-point MPC 推理流程。
    C2 比 C1 更深，卷积层更多，因此主要用于验证多层 CNN 在安全推理下能否完整串联。
    """

    # Conv1: [N, 3, 32, 32] -> [N, 16, 32, 32]
    # padding=1 用来保持空间尺寸不变。
    h = conv2d_secret_fixed(
        x_i=x_i,
        w_i=conv1_w_i,
        b_i=conv1_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=1
    )

    h = sbn_fixed(
        x_i=h,
        eps1_i=bn1_eps1_i,
        eps2_i=bn1_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    h = srelu(
        xi=h,
        conn=conn,
        party_id=party_id
    )

    # Conv2: [N, 16, 32, 32] -> [N, 16, 32, 32]
    # 继续保持通道数和空间尺寸不变。
    h = conv2d_secret_fixed(
        x_i=h,
        w_i=conv2_w_i,
        b_i=conv2_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=1
    )

    h = sbn_fixed(
        x_i=h,
        eps1_i=bn2_eps1_i,
        eps2_i=bn2_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    h = srelu(
        xi=h,
        conn=conn,
        party_id=party_id
    )

    # Conv3: [N, 16, 32, 32] -> [N, 16, 32, 32]
    h = conv2d_secret_fixed(
        x_i=h,
        w_i=conv3_w_i,
        b_i=conv3_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=1
    )

    h = sbn_fixed(
        x_i=h,
        eps1_i=bn3_eps1_i,
        eps2_i=bn3_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # Sonic 优化顺序：MaxPool -> SReLU
    # 第一次池化将 32x32 降到 16x16。
    h = maxpool2d_secret(
        x_i=h,
        kernel_size=2,
        stride=2,
        padding=0,
        conn=conn,
        party_id=party_id
    )

    h = srelu(
        xi=h,
        conn=conn,
        party_id=party_id
    )

    # Conv4: [N, 16, 16, 16] -> [N, 32, 16, 16]
    # 这里通道数从 16 提升到 32，空间尺寸保持不变。
    h = conv2d_secret_fixed(
        x_i=h,
        w_i=conv4_w_i,
        b_i=conv4_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=1
    )

    h = sbn_fixed(
        x_i=h,
        eps1_i=bn4_eps1_i,
        eps2_i=bn4_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    h = srelu(
        xi=h,
        conn=conn,
        party_id=party_id
    )

    # Conv5: [N, 32, 16, 16] -> [N, 32, 16, 16]
    h = conv2d_secret_fixed(
        x_i=h,
        w_i=conv5_w_i,
        b_i=conv5_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=1
    )

    h = sbn_fixed(
        x_i=h,
        eps1_i=bn5_eps1_i,
        eps2_i=bn5_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    h = srelu(
        xi=h,
        conn=conn,
        party_id=party_id
    )

    # Conv6: [N, 32, 16, 16] -> [N, 32, 16, 16]
    h = conv2d_secret_fixed(
        x_i=h,
        w_i=conv6_w_i,
        b_i=conv6_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=1
    )

    h = sbn_fixed(
        x_i=h,
        eps1_i=bn6_eps1_i,
        eps2_i=bn6_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # Sonic 优化顺序：MaxPool -> SReLU
    # 第二次池化将 16x16 降到 8x8。
    h = maxpool2d_secret(
        x_i=h,
        kernel_size=2,
        stride=2,
        padding=0,
        conn=conn,
        party_id=party_id
    )

    h = srelu(
        xi=h,
        conn=conn,
        party_id=party_id
    )

    # Conv7: [N, 32, 8, 8] -> [N, 48, 6, 6]
    # 后半部分不再 padding，空间尺寸开始逐层缩小。
    h = conv2d_secret_fixed(
        x_i=h,
        w_i=conv7_w_i,
        b_i=conv7_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=0
    )

    h = sbn_fixed(
        x_i=h,
        eps1_i=bn7_eps1_i,
        eps2_i=bn7_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    h = srelu(
        xi=h,
        conn=conn,
        party_id=party_id
    )

    # Conv8: [N, 48, 6, 6] -> [N, 48, 4, 4]
    h = conv2d_secret_fixed(
        x_i=h,
        w_i=conv8_w_i,
        b_i=conv8_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=0
    )

    h = sbn_fixed(
        x_i=h,
        eps1_i=bn8_eps1_i,
        eps2_i=bn8_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    h = srelu(
        xi=h,
        conn=conn,
        party_id=party_id
    )

    # Conv9: [N, 48, 4, 4] -> [N, 64, 2, 2]
    h = conv2d_secret_fixed(
        x_i=h,
        w_i=conv9_w_i,
        b_i=conv9_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=0
    )

    h = sbn_fixed(
        x_i=h,
        eps1_i=bn9_eps1_i,
        eps2_i=bn9_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # Sonic 优化顺序：MaxPool -> SReLU
    # 第三次池化将 64x2x2 压缩成 64x1x1。
    h = maxpool2d_secret(
        x_i=h,
        kernel_size=2,
        stride=2,
        padding=0,
        conn=conn,
        party_id=party_id
    )

    h = srelu(
        xi=h,
        conn=conn,
        party_id=party_id
    )

    # Flatten: [N, 64, 1, 1] -> [N, 64]
    h = flatten_nchw(h)

    # FC1: 64 -> 10
    # 得到分类 logits 前的线性输出。
    h_raw = linear_secret_weight(
        x_i=h,
        w_i=fc1_w_i,
        b_i=fc1_b_i,
        conn=conn,
        party_id=party_id
    )

    # 线性层乘法后需要截断，恢复 fixed-point scale。
    h = secure_trunc(
        x_i=h_raw,
        shift_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # Final SBN10
    # 最后一层 BN 输出最终 logits share。
    out = sbn_fixed(
        x_i=h,
        eps1_i=bn10_eps1_i,
        eps2_i=bn10_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    return out