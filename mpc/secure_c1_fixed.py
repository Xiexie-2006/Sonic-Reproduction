from mpc.conv2d_fixed import conv2d_secret_fixed
from mpc.sbn import sbn_fixed
from mpc.maxpool2d_secret import maxpool2d_secret
from mpc.srelu import srelu
from mpc.linear_secret import linear_secret_weight
from mpc.trunc import secure_trunc


def flatten_nchw(x_i):
    # 将 NCHW 格式的卷积特征拉平成二维矩阵。
    # 对 secret share 来说，flatten 只是本地 reshape，不需要通信。
    n = x_i.shape[0]
    return x_i.reshape(n, -1)


def secure_c1_fixed_inference(
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

    fc1_w_i,
    fc1_b_i,
    bn8_eps1_i,
    bn8_eps2_i,

    scale_bits,
    conn,
    party_id
):
    """
    Sonic C1 fixed-point MPC inference.

    C1:
        Conv1 -> SBN1 -> SReLU1
        Conv2 -> SBN2 -> MaxPool1 -> SReLU2
        Conv3 -> SBN3 -> SReLU3
        Conv4 -> SBN4 -> MaxPool2 -> SReLU4
        Conv5 -> SBN5 -> SReLU5
        Conv6 -> SBN6 -> SReLU6
        Conv7 -> SBN7 -> SReLU7
        Flatten
        FC1   -> SBN8

    这个函数对应 Sonic 论文中 CIFAR-10 的 C1 网络结构。
    所有输入、卷积核、BN 参数和全连接参数都已经处于 arithmetic share 状态。
    整个流程保持 fixed-point 计算，每次乘法类操作后都要注意尺度恢复。
    """

    # Conv1: [N, 3, 32, 32] -> [N, 64, 30, 30]
    # 第一层卷积不使用 padding，空间尺寸从 32x32 变成 30x30。
    h = conv2d_secret_fixed(
        x_i=x_i,
        w_i=conv1_w_i,
        b_i=conv1_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=0
    )

    # SBN1: 对 Conv1 输出做安全 BN。
    h = sbn_fixed(
        x_i=h,
        eps1_i=bn1_eps1_i,
        eps2_i=bn1_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # SReLU1: 安全激活函数，保留非负值，过滤负值。
    h = srelu(
        xi=h,
        conn=conn,
        party_id=party_id
    )

    # Conv2: [N, 64, 30, 30] -> [N, 64, 28, 28]
    h = conv2d_secret_fixed(
        x_i=h,
        w_i=conv2_w_i,
        b_i=conv2_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=0
    )

    h = sbn_fixed(
        x_i=h,
        eps1_i=bn2_eps1_i,
        eps2_i=bn2_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # Sonic 优化顺序：MaxPool -> SReLU
    # 这里先池化再激活，和该复现版本中的结构定义保持一致。
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

    # Conv3: [N, 64, 14, 14] -> [N, 64, 12, 12]
    h = conv2d_secret_fixed(
        x_i=h,
        w_i=conv3_w_i,
        b_i=conv3_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=0
    )

    h = sbn_fixed(
        x_i=h,
        eps1_i=bn3_eps1_i,
        eps2_i=bn3_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    h = srelu(
        xi=h,
        conn=conn,
        party_id=party_id
    )

    # Conv4: [N, 64, 12, 12] -> [N, 64, 10, 10]
    h = conv2d_secret_fixed(
        x_i=h,
        w_i=conv4_w_i,
        b_i=conv4_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=0
    )

    h = sbn_fixed(
        x_i=h,
        eps1_i=bn4_eps1_i,
        eps2_i=bn4_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # Sonic 优化顺序：MaxPool -> SReLU
    # 第二次池化把 10x10 降到 5x5。
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

    # Conv5: [N, 64, 5, 5] -> [N, 64, 3, 3]
    h = conv2d_secret_fixed(
        x_i=h,
        w_i=conv5_w_i,
        b_i=conv5_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=0
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

    # Conv6: [N, 64, 3, 3] -> [N, 64, 3, 3]
    # 为保持 3x3 输出，这里 padding=1。
    # 这一步是为了和论文表格中的输出尺寸保持一致。
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

    h = srelu(
        xi=h,
        conn=conn,
        party_id=party_id
    )

    # Conv7: [N, 64, 3, 3] -> [N, 16, 3, 3]
    # 为保持 3x3 输出，这里 padding=1。
    # 输出通道降到 16，后面 flatten 后正好是 16*3*3=144。
    h = conv2d_secret_fixed(
        x_i=h,
        w_i=conv7_w_i,
        b_i=conv7_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=1,
        padding=1
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

    # Flatten: [N, 16, 3, 3] -> [N, 144]
    # reshape 本身不涉及数值计算，所以可以本地完成。
    h = flatten_nchw(h)

    # FC1: 144 -> 10
    # 全连接层中的乘法会让 scale 从 2^f 变成 2^(2f)。
    h_raw = linear_secret_weight(
        x_i=h,
        w_i=fc1_w_i,
        b_i=fc1_b_i,
        conn=conn,
        party_id=party_id
    )

    # FC 后做 secure_trunc，恢复 fixed-point scale。
    h = secure_trunc(
        x_i=h_raw,
        shift_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # Final SBN8
    # 最后一层 BN 后直接输出 logits share。
    out = sbn_fixed(
        x_i=h,
        eps1_i=bn8_eps1_i,
        eps2_i=bn8_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    return out