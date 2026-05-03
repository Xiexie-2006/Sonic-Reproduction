from mpc.linear_secret import linear_secret_weight
from mpc.relu import relu


def secure_two_layer_mlp_secret(x_i, W1_i, b1_i, W2_i, b2_i, conn, party_id):
    """
    秘密模型参数版本的两层 MLP：

        H = XW1 + b1
        A = ReLU(H)
        Y = AW2 + b2

    其中：
        X 是 arithmetic share
        W1, b1, W2, b2 全部都是 arithmetic share

    这个版本和公开权重 MLP 不同。
    这里不仅输入 X 是秘密分享状态，模型参数 W 和 b 也是秘密分享状态，
    因此线性层中的乘法不能本地直接算，需要调用安全乘法协议。
    """

    # 第一层：Secret Linear。
    # 输入、权重和偏置都是 arithmetic share，
    # 所以这里使用 linear_secret_weight 完成安全矩阵乘法。
    h_i = linear_secret_weight(
        x_i=x_i,
        w_i=W1_i,
        b_i=b1_i,
        conn=conn,
        party_id=party_id
    )

    # 非线性层：Secure ReLU。
    # h_i 仍然是秘密分享状态，不能直接判断正负，
    # relu 内部会通过安全比较得到符号 gate，再计算 gate * h。
    a_i = relu(
        xi=h_i,
        conn=conn,
        party_id=party_id
    )

    # 第二层：Secret Linear。
    # 激活后的结果 a_i 继续作为秘密输入，
    # 和第二层秘密权重 W2_i、秘密偏置 b2_i 进行安全线性计算。
    y_i = linear_secret_weight(
        x_i=a_i,
        w_i=W2_i,
        b_i=b2_i,
        conn=conn,
        party_id=party_id
    )

    return y_i