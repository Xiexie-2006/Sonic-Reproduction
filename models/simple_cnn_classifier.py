import torch
import torch.nn as nn
import numpy as np


class SimpleCNNClassifier(nn.Module):
    """
    一个小型 CNN 分类模型：

        Conv2D(1 -> 2)
        → ReLU
        → Flatten
        → Linear(8 -> 3)

    输入:
        [N, 1, 3, 3]

    输出:
        [N, 3] logits

    说明：
        这个模型用于验证 MPC 分类推理流程：
        PyTorch logits / predictions
        vs
        MPC logits / predictions

    这里的网络规模故意设计得比较小，主要目的是方便逐层检查。
    比如卷积输出、ReLU 输出、flatten 后的向量以及最后 logits，
    都可以和 MPC 版本中的 fixed-point secret 计算结果进行对照。
    """

    def __init__(self):
        super().__init__()

        # 卷积层输入通道为 1，输出通道为 2。
        # 输入通常是 [N, 1, 3, 3]，卷积核大小为 2x2，
        # 因此输出空间尺寸会变成 2x2。
        # 由于有 2 个输出通道，所以 flatten 后一共有 2 * 2 * 2 = 8 个特征。
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=2,
            stride=1,
            padding=0,
            bias=True
        )

        # 明文模型中直接使用 PyTorch 的 ReLU。
        # MPC 版本中这一层需要用安全比较、SReLU 或等价的安全激活逻辑来实现。
        self.relu = nn.ReLU()

        # 全连接层输入维度为 8，对应卷积输出展平后的长度。
        # 输出维度为 3，表示这是一个三分类任务，输出的是 3 个类别的 logits。
        self.fc = nn.Linear(
            in_features=8,
            out_features=3,
            bias=True
        )

        # 手动初始化参数，避免随机权重导致每次测试结果不同。
        # 这样 PyTorch 明文结果和 MPC 推理结果可以稳定对齐。
        self._init_weights()

    def _init_weights(self):
        # 初始化权重时不需要梯度记录。
        # 这里的模型只用于推理验证，不进行训练。
        with torch.no_grad():
            # Conv weight shape:
            # [out_channels, in_channels, kH, kW]
            #
            # 第一个卷积核取 2x2 区域的主对角线信息：
            # [[1, 0],
            #  [0, 1]]
            #
            # 第二个卷积核取 2x2 区域的副对角线信息：
            # [[0, 1],
            #  [1, 0]]
            #
            # 这样设计可以让两个输出通道关注不同位置，
            # 便于检查 MPC 卷积中多输出通道的计算是否正确。
            self.conv.weight.copy_(
                torch.tensor(
                    [
                        [
                            [
                                [1.0, 0.0],
                                [0.0, 1.0]
                            ]
                        ],
                        [
                            [
                                [0.0, 1.0],
                                [1.0, 0.0]
                            ]
                        ]
                    ],
                    dtype=torch.float32
                )
            )

            # 两个输出通道分别设置不同的 bias。
            # 其中包含正数和负数，可以顺便验证 fixed-point 编码对符号的处理。
            self.conv.bias.copy_(
                torch.tensor([0.25, -0.25], dtype=torch.float32)
            )

            # FC weight shape in PyTorch:
            # [out_dim, in_dim]
            #
            # 这里输出维度是 3，所以有 3 行权重；
            # 输入维度是 8，所以每一行有 8 个参数。
            # 权重中包含正数、负数和小数，主要是为了覆盖更多 fixed-point 场景。
            self.fc.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.5, -0.5, 1.0, 0.25, -0.25, 0.5, 1.0],
                        [-0.5, 1.0, 0.5, -0.25, 1.0, 0.5, -0.5, 0.25],
                        [0.25, -0.5, 1.0, 0.5, -0.25, 1.0, 0.25, -0.5]
                    ],
                    dtype=torch.float32
                )
            )

            # 三分类输出对应 3 个 bias。
            # 后续 MPC 版本会把这些 bias 编码成 fixed-point 后参与秘密计算。
            self.fc.bias.copy_(
                torch.tensor([0.125, -0.25, 0.5], dtype=torch.float32)
            )

    def forward(self, x):
        # 第一步：卷积计算。
        # 输入 x 的标准形状是 [batch_size, channel, height, width]。
        x = self.conv(x)

        # 第二步：ReLU 激活。
        # 负数位置会被置为 0，非负数保持原值。
        x = self.relu(x)

        # 第三步：展平。
        # start_dim=1 表示保留 batch 维度，把后面的通道和空间维度合并成一维。
        x = torch.flatten(x, start_dim=1)

        # 第四步：全连接层输出 3 个 logits。
        # logits 后续可以通过 argmax 得到预测类别。
        x = self.fc(x)

        return x


def export_numpy_params(model):
    """
    导出 PyTorch CNN 分类模型参数给 MPC 使用。

    Conv:
        PyTorch Conv2d 权重格式就是：
            [out_channels, in_channels, kH, kW]

    Linear:
        PyTorch:
            y = x @ weight.T + bias

        MPC:
            y = x @ W + b

        所以：
            W = weight.T

    这个函数主要负责把明文 PyTorch 模型中的参数取出来，
    转成 numpy 后交给 MPC 侧进行 fixed-point 编码和秘密分享。
    """

    # 卷积权重本身的维度顺序已经和卷积计算需要的格式一致，
    # 因此这里直接导出，不需要额外转置。
    conv_w = model.conv.weight.detach().cpu().numpy().astype(np.float64)

    # 卷积 bias 直接导出即可。
    conv_b = model.conv.bias.detach().cpu().numpy().astype(np.float64)

    # PyTorch Linear 的 weight 是 [out_features, in_features]。
    # 但 MPC 线性层通常按 x @ W + b 来写，
    # 所以需要转置成 [in_features, out_features]。
    fc_w = model.fc.weight.detach().cpu().numpy().T.astype(np.float64)

    # 全连接层 bias 不涉及矩阵方向，直接导出。
    fc_b = model.fc.bias.detach().cpu().numpy().astype(np.float64)

    # 返回卷积层和全连接层的全部参数，
    # 后续 MPC 分类推理脚本会按照这个顺序读取。
    return conv_w, conv_b, fc_w, fc_b