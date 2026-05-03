from mpc.linear_secret import linear_secret_weight
from mpc.relu import relu


def secure_two_layer_mlp_secret_fixed(
    x_i,
    W1_i,
    b1_i,
    W2_i,
    b2_i,
    scale_x_bits,
    scale_w1_bits,
    scale_w2_bits,
    conn,
    party_id
):
    """
    固定点两层 MLP：

        H = XW1 + b1
        A = ReLU(H)
        Y = AW2 + b2

    这里不做中间 truncation，而是跟踪 scale：

        X scale = 2^scale_x_bits
        W1 scale = 2^scale_w1_bits
        H scale = 2^(scale_x_bits + scale_w1_bits)

        ReLU 不改变 scale

        W2 scale = 2^scale_w2_bits
        Y scale = 2^(scale_x_bits + scale_w1_bits + scale_w2_bits)

    这个版本主要用于观察 fixed-point 乘法后 scale 是如何逐层累积的。
    它不在中间截断，所以更适合作为理解定点数尺度传播的测试版本。
    """

    # 第一层秘密权重线性计算。
    # X、W1、b1 都是 arithmetic share，所以使用 linear_secret_weight。
    h_i = linear_secret_weight(
        x_i=x_i,
        w_i=W1_i,
        b_i=b1_i,
        conn=conn,
        party_id=party_id
    )

    # 第一层输出 scale = 输入 scale + 第一层权重 scale。
    h_scale_bits = scale_x_bits + scale_w1_bits

    # ReLU 只做符号选择，不改变定点数缩放尺度。
    a_i = relu(
        xi=h_i,
        conn=conn,
        party_id=party_id
    )

    # 第二层秘密权重线性计算。
    # 输入 a_i 的 scale 仍然是 h_scale_bits。
    y_i = linear_secret_weight(
        x_i=a_i,
        w_i=W2_i,
        b_i=b2_i,
        conn=conn,
        party_id=party_id
    )

    # 第二层输出 scale 继续累加 W2 的 scale。
    y_scale_bits = h_scale_bits + scale_w2_bits

    # 返回结果 share 以及当前结果对应的 scale_bits。
    # 后续测试时可以根据 y_scale_bits 做解码。
    return y_i, y_scale_bits