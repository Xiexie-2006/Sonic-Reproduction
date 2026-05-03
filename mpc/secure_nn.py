from mpc.linear import linear_public_weight
from mpc.relu import relu


def secure_two_layer_mlp(xi, W1, b1, W2, b2, conn, party_id):
    """
    两层安全 MLP：

        H  = XW1 + b1
        A  = ReLU(H)
        Y  = AW2 + b2

    当前设定：
        X 是秘密分享输入
        W1, b1, W2, b2 是公开模型参数

    这个函数是最基础的安全神经网络推理流程。
    它主要用于早期验证：公开权重线性层 + 安全 ReLU + 公开权重线性层
    是否能在两方秘密分享状态下得到正确输出。
    """

    # 第一层线性层。
    # 因为 W1、b1 是公开参数，所以每一方可以直接用自己的输入 share 做本地矩阵乘法。
    h_i = linear_public_weight(
        xi=xi,
        W=W1,
        b=b1,
        party_id=party_id
    )

    # 安全 ReLU。
    # h_i 是 arithmetic share，不能直接判断正负，
    # 所以 relu 内部会调用安全比较和安全乘法。
    a_i = relu(
        xi=h_i,
        conn=conn,
        party_id=party_id
    )

    # 第二层线性层。
    # 激活结果仍然是秘密分享状态，继续和公开权重 W2 做本地线性计算。
    y_i = linear_public_weight(
        xi=a_i,
        W=W2,
        b=b2,
        party_id=party_id
    )

    return y_i