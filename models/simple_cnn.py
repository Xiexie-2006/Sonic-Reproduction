import torch
import torch.nn as nn
import numpy as np


class SimpleCNN(nn.Module):
    """
    明文 PyTorch CNN:

        Conv2D
        → ReLU
        → Flatten
        → Linear

    用于和 MPC fixed-point secret CNN 对齐。

    这里单独写一个很小的 CNN，主要不是为了追求模型复杂度，
    而是为了方便和后面的秘密分享版本逐层对照：
    明文卷积结果、ReLU 结果、展平结果、全连接结果都可以拿来做验证。
    """

    def __init__(self):
        super().__init__()

        # 输入只有 1 个通道，输出也设置为 1 个通道。
        # kernel_size=2 表示卷积核大小为 2x2，stride=1 表示每次滑动 1 格。
        # padding=0 保持最直接的卷积形式，方便和 MPC 版本手动对齐。
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            stride=1,
            padding=0,
            bias=True
        )

        # 明文 CNN 中直接使用 PyTorch 的 ReLU。
        # 在 MPC 版本中，这一层通常要替换成安全比较或 SReLU 等安全激活函数。
        self.relu = nn.ReLU()

        # 卷积后的特征图会被 flatten 成长度为 4 的向量，
        # 因此这里全连接层的输入维度设置为 4，输出维度为 1。
        self.fc = nn.Linear(
            in_features=4,
            out_features=1,
            bias=True
        )

        # 为了让测试结果稳定，这里不用随机初始化，
        # 而是手动指定卷积层和全连接层的参数。
        self._init_weights()

    def _init_weights(self):
        # 初始化参数时不需要记录梯度。
        # 这个模型主要用于推理和结果对照，不涉及训练过程。
        with torch.no_grad():
            # Conv kernel:
            # [[1, 0],
            #  [0, 1]]
            #
            # 这个卷积核相当于取 2x2 区域的主对角线元素相加，
            # 形式比较简单，便于检查 MPC 卷积实现是否正确。
            self.conv.weight.copy_(
                torch.tensor(
                    [[[[1.0, 0.0],
                       [0.0, 1.0]]]],
                    dtype=torch.float32
                )
            )

            # 卷积层偏置设置为 0.25。
            # 在 fixed-point 版本中，这个小数也需要编码到整数环中。
            self.conv.bias.copy_(
                torch.tensor([0.25], dtype=torch.float32)
            )

            # FC:
            # [1.0, 0.5, -1.0, 2.0]
            #
            # 全连接层权重中故意包含正数、小数和负数，
            # 这样可以同时测试 fixed-point 编码、负数表示和乘加逻辑。
            self.fc.weight.copy_(
                torch.tensor(
                    [[1.0, 0.5, -1.0, 2.0]],
                    dtype=torch.float32
                )
            )

            # 全连接层偏置设置为 0.125。
            # 这个值也会用于和 MPC 端的 fixed-point bias 对齐。
            self.fc.bias.copy_(
                torch.tensor([0.125], dtype=torch.float32)
            )

    def forward(self, x):
        # 第一步：明文卷积。
        # 输入 x 的形状一般为 [batch, channel, height, width]。
        x = self.conv(x)

        # 第二步：ReLU 激活。
        # 小于 0 的值会被置为 0，大于等于 0 的值保持不变。
        x = self.relu(x)

        # 第三步：展平。
        # start_dim=1 表示保留 batch 维度，只把后面的通道和空间维度拉平成一维。
        x = torch.flatten(x, start_dim=1)

        # 第四步：全连接层，得到最终输出。
        x = self.fc(x)

        return x


def export_numpy_params(model):
    """
    导出 PyTorch CNN 参数给 MPC 使用。

    Conv:
        PyTorch Conv2d 权重本身就是：
        [out_channels, in_channels, kH, kW]

    Linear:
        PyTorch Linear:
            y = x @ weight.T + bias

        MPC linear:
            y = x @ W + b

        所以：
            W = weight.T

    这个函数的作用是把 PyTorch 模型中的参数转成 numpy 格式，
    后续 MPC 版本可以直接读取这些参数，再进行 fixed-point 编码和秘密分享。
    """

    # 卷积权重从 PyTorch Tensor 转成 numpy。
    # detach() 表示脱离计算图，cpu() 保证数据在 CPU 上，
    # astype(np.float64) 是为了后续 fixed-point 编码时精度更稳定。
    conv_w = model.conv.weight.detach().cpu().numpy().astype(np.float64)

    # 卷积偏置同样导出为 numpy。
    conv_b = model.conv.bias.detach().cpu().numpy().astype(np.float64)

    # PyTorch 的 Linear 权重形状是 [out_features, in_features]。
    # 但当前 MPC linear 的计算形式通常写成 x @ W + b，
    # 所以这里需要转置成 [in_features, out_features]。
    fc_w = model.fc.weight.detach().cpu().numpy().T.astype(np.float64)

    # 全连接层 bias 不需要转置，直接导出即可。
    fc_b = model.fc.bias.detach().cpu().numpy().astype(np.float64)

    # 返回顺序保持为卷积参数在前，全连接参数在后，
    # 这样后续 secret CNN 测试脚本中读取会更清晰。
    return conv_w, conv_b, fc_w, fc_b