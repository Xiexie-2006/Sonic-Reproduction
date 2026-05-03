import torch
import torch.nn as nn
import numpy as np

from models.sonic_model_specs import (
    ModelSpec,
    LayerSpec,
    get_sonic_model_spec,
)


class SonicTorchModel(nn.Module):
    """
    根据 sonic_model_specs.py 中的 M1/M2/C1/C2 结构
    自动构建 PyTorch 明文模型。

    当前用途：
        1. 检查论文模型结构是否能 forward 跑通
        2. 检查每层 shape 是否正确
        3. 后续导出 Conv / FC / BN 参数给 MPC 使用

    注意：
        本文件只负责明文 PyTorch 模型结构。
        真正 MPC 安全推理仍由 mpc/ 下的安全协议实现。

    简单来说，这个类相当于是 Sonic 论文模型的明文版本。
    它不是安全计算本身，而是用来提供可对照的 PyTorch 结果和参数。
    """

    def __init__(self, spec, seed=1234, use_bias=False):
        super().__init__()

        # spec 可以直接传模型名，比如 "M1"、"M2"、"C1"、"C2"。
        # 如果传入的是字符串，就从规格表中取出对应 ModelSpec。
        if isinstance(spec, str):
            spec = get_sonic_model_spec(spec)

        # 这里做类型检查，避免后面构建网络时才出现不容易定位的问题。
        if not isinstance(spec, ModelSpec):
            raise TypeError("spec must be a model name string or ModelSpec")

        self.spec = spec
        self.seed = seed

        # 是否给 Conv / FC 使用 bias。
        # Sonic 论文结构中 BN 层较多，因此这里默认不使用 bias，
        # 可以减少参数重复，也更方便后续和 MPC 参数导出对齐。
        self.use_bias = use_bias

        # ModuleList 用来按顺序保存所有 PyTorch 层。
        # 普通 list 不会被 PyTorch 自动注册为子模块，所以这里必须用 ModuleList。
        self.layers = nn.ModuleList()

        # 单独保存层名称和层类型，方便 trace、参数导出和调试。
        self.layer_names = []
        self.layer_types = []

        # 根据 ModelSpec 创建真正的 PyTorch 层。
        self._build_layers()

        # 初始化参数。
        # 这里不是训练好的论文权重，只是为了让 forward 可以稳定跑通。
        self._init_parameters(seed=seed)

    def _build_layers(self):
        # 用于记录最近一个真正带参数或改变维度的层类型。
        # BN 层需要根据前一层是 conv 还是 fc，决定使用 BatchNorm2d 还是 BatchNorm1d。
        last_real_layer_type = None

        for layer in self.spec.layers:
            layer_type = layer.layer_type.lower()

            if layer_type == "conv":
                # 根据 LayerSpec 中记录的卷积参数创建 Conv2d。
                module = nn.Conv2d(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    bias=self.use_bias
                )

                last_real_layer_type = "conv"

            elif layer_type == "fc":
                # 根据 LayerSpec 中的输入输出维度创建全连接层。
                module = nn.Linear(
                    in_features=layer.in_features,
                    out_features=layer.out_features,
                    bias=self.use_bias
                )

                last_real_layer_type = "fc"

            elif layer_type == "bn":
                # 如果 BN 接在卷积后面，输入一般是 [N, C, H, W]，
                # 所以使用 BatchNorm2d。
                if last_real_layer_type == "conv":
                    module = nn.BatchNorm2d(
                        num_features=layer.num_features,
                        affine=True,
                        track_running_stats=True
                    )

                # 如果 BN 接在全连接后面，输入一般是 [N, F]，
                # 所以使用 BatchNorm1d。
                elif last_real_layer_type == "fc":
                    module = nn.BatchNorm1d(
                        num_features=layer.num_features,
                        affine=True,
                        track_running_stats=True
                    )

                else:
                    # BN 层必须依赖前面的 Conv 或 FC 输出。
                    # 如果结构表写错，这里会直接报错，便于定位。
                    raise RuntimeError(
                        f"BN layer {layer.name} appears before Conv/FC layer."
                    )

            elif layer_type == "relu":
                # 明文 PyTorch 中直接使用 ReLU。
                # 对应到 MPC 时，这里会替换成安全 ReLU 或 Sonic 中的 SReLU 类逻辑。
                module = nn.ReLU()

            elif layer_type == "maxpool":
                # 最大池化层，主要出现在 M2、C1、C2 这些 CNN 模型中。
                module = nn.MaxPool2d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride
                )

            elif layer_type == "flatten":
                # Flatten 用于连接卷积部分和全连接部分。
                # start_dim=1 表示保留 batch 维度，只展开后面的特征维度。
                module = nn.Flatten(start_dim=1)

            else:
                raise ValueError(f"Unknown layer_type: {layer.layer_type}")

            # 保存当前层对象，以及对应的名字和类型。
            # 后续 forward、trace 和 export 都依赖这三个列表顺序一致。
            self.layers.append(module)
            self.layer_names.append(layer.name)
            self.layer_types.append(layer_type)

    def _init_parameters(self, seed=1234):
        """
        初始化权重。

        这里不是训练模型，只是为了让 forward 可以稳定跑通。
        后续如果要接真实 accuracy，需要训练或加载论文模型权重。
        """

        # 固定随机种子，保证每次运行生成的参数一致。
        # 这样 PyTorch 明文输出和 MPC 输出对比时更稳定。
        torch.manual_seed(seed)

        for module in self.layers:
            if isinstance(module, nn.Conv2d):
                # 卷积权重使用较小范围的均匀分布初始化。
                # 数值范围不大，可以减少 fixed-point 编码后溢出或误差过大的情况。
                nn.init.uniform_(module.weight, a=-0.05, b=0.05)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                # 全连接权重同样使用较小范围的均匀分布。
                nn.init.uniform_(module.weight, a=-0.05, b=0.05)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # BN 初始设置成接近恒等变换：
                # gamma = 1, beta = 0, running_mean = 0, running_var = 1。
                # 这样在没有训练权重的情况下，BN 不会明显改变数据分布。
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

                module.running_mean.zero_()
                module.running_var.fill_(1.0)

    def forward(self, x):
        # 普通前向传播。
        # 按照 self.layers 中的顺序依次执行每一层。
        for module in self.layers:
            x = module(x)

        return x

    def forward_with_trace(self, x):
        """
        forward 时记录每一层输入输出 shape。

        这个函数主要用于调试模型结构。
        如果某一层输出 shape 和论文表格不一致，可以通过 trace 很快定位。
        """

        trace = []

        for name, layer_type, module in zip(
            self.layer_names,
            self.layer_types,
            self.layers
        ):
            # 记录当前层输入形状。
            in_shape = tuple(x.shape)

            # 执行当前层。
            x = module(x)

            # 记录当前层输出形状。
            out_shape = tuple(x.shape)

            trace.append(
                {
                    "name": name,
                    "type": layer_type,
                    "in_shape": in_shape,
                    "out_shape": out_shape,
                }
            )

        return x, trace

    def count_parameters(self):
        # 统计模型中需要训练的参数数量。
        # 虽然当前复现阶段主要做推理验证，但这个函数可以辅助检查模型规模。
        total = 0

        for p in self.parameters():
            if p.requires_grad:
                total += p.numel()

        return total


def build_sonic_torch_model(name, seed=1234, use_bias=False):
    """
    根据模型名构建 PyTorch 模型。

    name:
        M1 / M2 / C1 / C2

    这个函数相当于一个简单封装，方便测试脚本中直接通过模型名创建模型。
    """

    return SonicTorchModel(
        spec=name,
        seed=seed,
        use_bias=use_bias
    )


def extract_bn_eps_from_torch_bn(bn_module):
    """
    从 PyTorch BatchNorm 层提取 Sonic SBN 所需参数：

        z = eps1 * x + eps2

    PyTorch BN eval 模式:
        y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta

    因此：
        eps1 = gamma / sqrt(running_var + eps)
        eps2 = beta - gamma * running_mean / sqrt(running_var + eps)

    Sonic 中的 SBN 可以看成一种线性变换。
    所以这里提前把 PyTorch BN 的参数折叠成 eps1 和 eps2，
    后续 MPC 端只需要做乘法和加法即可。
    """

    if not isinstance(bn_module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        raise TypeError("bn_module must be BatchNorm1d or BatchNorm2d")

    # 提取 BN 中的可学习参数 gamma 和 beta。
    gamma = bn_module.weight.detach().cpu().numpy().astype(np.float64)
    beta = bn_module.bias.detach().cpu().numpy().astype(np.float64)

    # 提取 BN 在推理阶段使用的 running mean 和 running variance。
    running_mean = bn_module.running_mean.detach().cpu().numpy().astype(np.float64)
    running_var = bn_module.running_var.detach().cpu().numpy().astype(np.float64)

    # PyTorch BN 中用于数值稳定的小常数。
    eps = float(bn_module.eps)

    # 分母部分 sqrt(running_var + eps)。
    denom = np.sqrt(running_var + eps)

    # 将 BN 公式整理成 z = eps1 * x + eps2 的形式。
    eps1 = gamma / denom
    eps2 = beta - gamma * running_mean / denom

    return eps1, eps2


def export_sonic_model_params(model):
    """
    导出 PyTorch Sonic 模型参数。

    返回列表，每一项对应一个可导出的层：
        Conv2d:
            type, name, weight

        Linear:
            type, name, weight

        BatchNorm:
            type, name, eps1, eps2

    注意：
        PyTorch Linear 的 weight shape 是 [out_dim, in_dim]
        MPC linear 常用 W shape 是 [in_dim, out_dim]
        所以这里导出时转置成 [in_dim, out_dim]

    这个函数是明文模型和 MPC 模型之间的接口。
    它把 PyTorch 中的层参数整理成统一的字典格式，
    后续 fixed-point 编码、秘密分享和安全推理都可以基于这些参数进行。
    """

    exported = []

    for name, layer_type, module in zip(
        model.layer_names,
        model.layer_types,
        model.layers
    ):
        if isinstance(module, nn.Conv2d):
            # 卷积权重在 PyTorch 中本身就是
            # [out_channels, in_channels, kH, kW]，
            # 这个格式和后续卷积实现比较容易对应，因此直接导出。
            exported.append(
                {
                    "type": "conv",
                    "name": name,
                    "weight": module.weight.detach().cpu().numpy().astype(np.float64),
                    "bias": None if module.bias is None else module.bias.detach().cpu().numpy().astype(np.float64),
                    "stride": module.stride,
                    "padding": module.padding,
                }
            )

        elif isinstance(module, nn.Linear):
            # PyTorch Linear 计算是 y = x @ weight.T + bias。
            # MPC 线性层一般写成 y = x @ W + b，
            # 所以这里需要先把 weight 转置再导出。
            weight = module.weight.detach().cpu().numpy().T.astype(np.float64)

            exported.append(
                {
                    "type": "fc",
                    "name": name,
                    "weight": weight,
                    "bias": None if module.bias is None else module.bias.detach().cpu().numpy().astype(np.float64),
                }
            )

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # BN 层不直接导出 gamma、beta、mean、var，
            # 而是导出整理后的 eps1、eps2。
            # 这样后续 MPC 中只需要实现线性变换 z = eps1*x + eps2。
            eps1, eps2 = extract_bn_eps_from_torch_bn(module)

            exported.append(
                {
                    "type": "bn",
                    "name": name,
                    "eps1": eps1,
                    "eps2": eps2,
                }
            )

        elif isinstance(module, nn.ReLU):
            # ReLU 没有权重参数，但仍然需要导出层类型和名称。
            # MPC 推理时看到这一项，就知道这里要执行安全激活函数。
            exported.append(
                {
                    "type": "relu",
                    "name": name,
                }
            )

        elif isinstance(module, nn.MaxPool2d):
            # MaxPool 也没有可学习参数，只需要记录窗口大小和步长。
            # 后续安全推理时可以根据这些信息执行对应的池化逻辑。
            exported.append(
                {
                    "type": "maxpool",
                    "name": name,
                    "kernel_size": module.kernel_size,
                    "stride": module.stride,
                }
            )

        elif isinstance(module, nn.Flatten):
            # Flatten 没有参数，但它会改变张量形状。
            # 导出它是为了让 MPC 端保持和 PyTorch 前向传播完全一致的层顺序。
            exported.append(
                {
                    "type": "flatten",
                    "name": name,
                }
            )

    return exported