from mpc.linear_secret import linear_secret_weight
from mpc.relu import relu
from mpc.trunc import secure_trunc


def secure_two_layer_mlp_secret_fixed_trunc(
    x_i,
    W1_i,
    b1_i,
    W2_i,
    b2_i,
    scale_bits,
    conn,
    party_id
):
    """
    固定点 + 安全截断版本两层 MLP：

        H_raw = XW1 + b1      scale = 2^(2f)
        H     = Trunc(H_raw) scale = 2^f

        A     = ReLU(H)      scale = 2^f

        Y_raw = AW2 + b2     scale = 2^(2f)
        Y     = Trunc(Y_raw) scale = 2^f

    这个版本更接近最终 fixed-point MPC 推理流程：
    每经过一次线性层，都通过 secure_trunc 把 scale 恢复到统一的 2^f。
    这样后续多层网络不会因为 scale 不断累积而难以解码或溢出。
    """

    # 第一层安全全连接。
    # 输入和权重都是 scale=2^f，因此乘法后 raw 输出 scale=2^(2f)。
    h_raw_i = linear_secret_weight(
        x_i=x_i,
        w_i=W1_i,
        b_i=b1_i,
        conn=conn,
        party_id=party_id
    )

    # 第一层截断，将 scale 从 2^(2f) 恢复到 2^f。
    h_i = secure_trunc(
        x_i=h_raw_i,
        shift_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # 安全 ReLU。
    # 这一层只根据符号选择保留或置零，不改变 scale。
    a_i = relu(
        xi=h_i,
        conn=conn,
        party_id=party_id
    )

    # 第二层安全全连接。
    # ReLU 输出和 W2 相乘后，scale 再次变成 2^(2f)。
    y_raw_i = linear_secret_weight(
        x_i=a_i,
        w_i=W2_i,
        b_i=b2_i,
        conn=conn,
        party_id=party_id
    )

    # 第二层截断，恢复到统一的 scale=2^f。
    y_i = secure_trunc(
        x_i=y_raw_i,
        shift_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    return y_i