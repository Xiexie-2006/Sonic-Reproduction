from mpc.conv2d_fixed import conv2d_secret_fixed
from mpc.srelu import srelu
from mpc.linear_secret import linear_secret_weight
from mpc.trunc import secure_trunc


def flatten_nchw(x_i):
    """
    NCHW tensor flatten:

        [N, C, H, W] -> [N, C*H*W]

    对 secret share 来说，flatten 只是本地重排，不需要通信。

    这个函数一般用于把卷积层输出接到全连接层前。
    因为没有数值运算，只是改变张量形状，所以不会影响秘密分享的正确性。
    """

    n = x_i.shape[0]
    return x_i.reshape(n, -1)


def secure_fixed_cnn(
    x_i,
    conv_w_i,
    conv_b_i,
    fc_w_i,
    fc_b_i,
    scale_bits,
    conn,
    party_id,
    stride=1,
    padding=0
):
    """
    Fixed-point Secret CNN:

        Conv2D
        → Truncation
        → SReLU
        → Flatten
        → Secret Linear
        → Truncation

    输入：
        x_i:
            输入 share，shape = [N, C, H, W]

        conv_w_i:
            卷积核 share，shape = [C_out, C_in, kH, kW]

        conv_b_i:
            卷积 bias share，shape = [C_out]

        fc_w_i:
            全连接层权重 share，shape = [flatten_dim, out_dim]

        fc_b_i:
            全连接层 bias share，shape = [out_dim]

    这是一个简化版的安全 CNN 推理流程。
    它通常用于先验证 Conv2D、SReLU、Flatten、Linear 这些模块能否正确串联，
    再逐步扩展到 Sonic 论文中的 M2、C1、C2 等完整模型。
    """

    # Conv2D + Truncation，输出 scale 仍然是 2^f。
    # conv2d_secret_fixed 内部已经完成了卷积后的 secure_trunc。
    conv_y_i = conv2d_secret_fixed(
        x_i=x_i,
        w_i=conv_w_i,
        b_i=conv_b_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id,
        stride=stride,
        padding=padding
    )

    # SReLU 不改变 fixed-point scale。
    # 它只根据符号判断保留或置零当前值。
    act_i = srelu(
        xi=conv_y_i,
        conn=conn,
        party_id=party_id
    )

    # Flatten。
    # 将卷积输出从 [N, C, H, W] 拉平成 [N, C*H*W]。
    flat_i = flatten_nchw(act_i)

    # Secret Linear，输出 scale = 2^(2f)。
    # 因为输入和权重都按 2^f 编码，相乘后尺度会扩大。
    fc_raw_i = linear_secret_weight(
        x_i=flat_i,
        w_i=fc_w_i,
        b_i=fc_b_i,
        conn=conn,
        party_id=party_id
    )

    # Linear 后截断，恢复到 scale = 2^f。
    # 这样输出才能继续接后面的 fixed-point 模块，或者解码成明文结果对比。
    out_i = secure_trunc(
        x_i=fc_raw_i,
        shift_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    return out_i