import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    明文 PyTorch 两层 MLP：

        H = XW1 + b1
        A = ReLU(H)
        Y = AW2 + b2

    为了和当前 MPC 环 Z_(2^32) 整数推理对齐，
    这里先使用整数权重。

    这个模型可以看作是最基础的 SFC + ReLU + SFC 结构，
    用来验证安全全连接层和安全激活函数在组合情况下是否能得到正确结果。
    """

    def __init__(self):
        super().__init__()

        # 第一层全连接：输入维度为 2，输出维度为 2。
        # 对应公式中的 H = XW1 + b1。
        self.fc1 = nn.Linear(2, 2, bias=True)

        # 明文 ReLU，用于和 MPC 中的安全 ReLU / SReLU 结果对齐。
        self.relu = nn.ReLU()

        # 第二层全连接：输入维度为 2，输出维度为 1。
        # 对应公式中的 Y = AW2 + b2。
        self.fc2 = nn.Linear(2, 1, bias=True)

        # 这里使用固定的整数权重，方便和 Z_(2^32) 环上的整数秘密分享推理对齐。
        self._init_integer_weights()

    def _init_integer_weights(self):
        # 参数初始化不需要梯度。
        # 当前模型主要用于推理验证，不参与训练。
        with torch.no_grad():
            # 注意 PyTorch Linear 权重形状是 [out_dim, in_dim]
            # 我们希望数学上 W1 是：
            # [[1, 1],
            #  [1, 3]]
            # 所以 PyTorch 里要转置成：
            # [[1, 1],
            #  [1, 3]]
            #
            # 这里刚好矩阵转置前后形式一致，但仍然保留这个说明，
            # 是为了强调 PyTorch Linear 和 MPC 矩阵乘法写法之间的差异。
            self.fc1.weight.copy_(torch.tensor([
                [1, 1],
                [1, 3]
            ], dtype=torch.float32))

            # 第一层 bias 设置为 0，方便先验证矩阵乘法主体逻辑。
            self.fc1.bias.copy_(torch.tensor([0, 0], dtype=torch.float32))

            # W2 数学上是：
            # [[4],
            #  [7]]
            # PyTorch Linear 里是 [1, 2]
            #
            # 因为 PyTorch 的 Linear 实际计算是：
            # y = x @ weight.T + bias
            # 所以这里写成一行 [4, 7]。
            self.fc2.weight.copy_(torch.tensor([
                [4, 7]
            ], dtype=torch.float32))

            # 第二层 bias 同样设置为 0，
            # 这样输出主要由前一层激活结果和 W2 决定，更便于手算检查。
            self.fc2.bias.copy_(torch.tensor([0], dtype=torch.float32))

    def forward(self, x):
        # 第一层线性变换，得到隐藏层结果 h。
        h = self.fc1(x)

        # 对隐藏层结果做 ReLU。
        # 在明文中这是普通 ReLU，在 MPC 中需要用安全比较逻辑替代。
        a = self.relu(h)

        # 第二层线性变换，得到最终输出 y。
        y = self.fc2(a)

        return y


def export_numpy_params(model):
    """
    导出为 MPC 使用的矩阵形式：

        PyTorch:
            Linear: y = x @ weight.T + bias

        MPC:
            y = x @ W + b

        所以：
            W = weight.T

    这个函数用于把明文 PyTorch MLP 的参数转成 MPC 侧更方便使用的矩阵形式。
    后续秘密分享推理时，会先对这些参数做整数环编码或直接作为整数参数使用。
    """

    # 第一层权重从 PyTorch 的 [out_dim, in_dim]
    # 转成 MPC 计算需要的 [in_dim, out_dim]。
    W1 = model.fc1.weight.detach().cpu().numpy().T.astype("int64")

    # 第一层 bias 直接导出为 int64。
    b1 = model.fc1.bias.detach().cpu().numpy().astype("int64")

    # 第二层权重同样需要转置，
    # 保证后续可以按照 x @ W2 + b2 的形式计算。
    W2 = model.fc2.weight.detach().cpu().numpy().T.astype("int64")

    # 第二层 bias 直接导出。
    b2 = model.fc2.bias.detach().cpu().numpy().astype("int64")

    # 返回两层全连接的权重和偏置。
    # 返回顺序和 forward 中的计算顺序一致，便于 MPC 测试脚本调用。
    return W1, b1, W2, b2