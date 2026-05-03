from mpc.conv2d_fixed import conv2d_secret_fixed
from mpc.sbn import sbn_fixed
from mpc.maxpool2d_secret import maxpool2d_secret
from mpc.srelu import srelu
from mpc.linear_secret import linear_secret_weight
from mpc.trunc import secure_trunc


def flatten_nchw(x_i):
    """
    [N, C, H, W] -> [N, C*H*W]

    Flatten 不改变数据值，只改变排列形状。
    因此在 MPC 中可以直接对本方 share 做本地 reshape。
    """
    n = x_i.shape[0]
    return x_i.reshape(n, -1)


def secure_fixed_cnn_sbn_pool_relu(
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
    conv_padding=0,
    pool_kernel_size=2,
    pool_stride=2
):
    """
    Sonic-style optimized Fixed-point Secret CNN:

        Conv2D
        → Secure Truncation
        → SBN
        → Secure MaxPool2D
        → SReLU
        → Flatten
        → Secret Linear
        → Secure Truncation

    论文优化思想：
        ReLU(MaxPool inputs) 与 MaxPool(ReLU inputs) 等价，
        因此可以先 MaxPool 再 ReLU，减少 ReLU 操作数量。

    这个函数体现的是 Sonic 中比较重要的优化思路：
    安全 ReLU 需要比较协议，代价比较高；
    如果把 MaxPool 放在 ReLU 前面，就可以减少需要执行 ReLU 的元素数量。
    """

    # 安全定点卷积。
    # 输出经过 secure_trunc 后仍保持 scale = 2^f。
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

    # SBN 层。
    # 这里的 BN 参数已经被整理成 eps1 和 eps2，
    # 因此计算形式是 eps1 * x + eps2。
    bn_y_i = sbn_fixed(
        x_i=conv_y_i,
        eps1_i=bn_eps1_i,
        eps2_i=bn_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # 先执行 MaxPool。
    # 由于 MaxPool 和 ReLU 在该场景下可以交换顺序，
    # 先池化可以减少后面 SReLU 需要处理的元素数量。
    pool_i = maxpool2d_secret(
        x_i=bn_y_i,
        kernel_size=pool_kernel_size,
        stride=pool_stride,
        padding=0,
        conn=conn,
        party_id=party_id
    )

    # 对池化后的结果再执行 SReLU。
    # 这样能降低安全比较和 B2A 转换的调用规模。
    act_i = srelu(
        xi=pool_i,
        conn=conn,
        party_id=party_id
    )

    # 将卷积特征展平，接入全连接层。
    flat_i = flatten_nchw(act_i)

    # 安全全连接层。
    # 这一层输出仍然需要截断，因为 fixed-point 乘法会放大尺度。
    fc_raw_i = linear_secret_weight(
        x_i=flat_i,
        w_i=fc_w_i,
        b_i=fc_b_i,
        conn=conn,
        party_id=party_id
    )

    # 恢复到 scale = 2^f，方便后续继续解码或接其他层。
    out_i = secure_trunc(
        x_i=fc_raw_i,
        shift_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    return out_i