from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class LayerSpec:
    """
    Sonic paper model layer specification.

    layer_type:
        "fc"
        "conv"
        "bn"
        "relu"
        "maxpool"
        "flatten"

    For conv:
        in_channels
        out_channels
        kernel_size
        padding
        stride

    For fc:
        in_features
        out_features

    For maxpool:
        kernel_size
        stride

    For bn:
        num_features

    这个类主要用来描述单独一层的结构信息。
    这里并不真正执行神经网络计算，只是把论文表格中的每层参数整理成统一格式，
    后面构建 PyTorch 明文模型和 MPC 模型时都可以直接读取这些信息。
    """

    # 当前层的类型，例如 conv、fc、bn、relu 等
    layer_type: str

    # 卷积层相关参数
    # 如果当前层不是卷积层，这些字段就保持为 None
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None

    # 全连接层相关参数
    # 如果当前层不是全连接层，这些字段就不使用
    in_features: Optional[int] = None
    out_features: Optional[int] = None

    # BN 层的特征数量
    # 对卷积 BN 来说一般对应通道数，对全连接 BN 来说对应输出维度
    num_features: Optional[int] = None

    # 卷积和池化都会用到 kernel_size 和 stride
    # padding 主要用于卷积层
    kernel_size: Optional[int] = None
    padding: Optional[int] = None
    stride: Optional[int] = None

    # 记录输入输出 shape，主要用于和论文表格中的尺寸对齐
    # 这里的 shape 一般写成 (C, H, W)
    input_shape: Optional[Tuple[int, int, int]] = None
    output_shape: Optional[Tuple[int, int, int]] = None

    # 层名称，例如 conv1、bn1、relu1
    # 后续打印结构、导出参数时会用到
    name: str = ""


@dataclass
class ModelSpec:
    # 模型名称，例如 M1、M2、C1、C2
    name: str

    # 对应的数据集，例如 MNIST 或 CIFAR-10
    dataset: str

    # 输入图像的形状，格式为 (C, H, W)
    input_shape: Tuple[int, int, int]

    # 分类类别数
    num_classes: int

    # 按顺序保存模型的每一层结构
    layers: List[LayerSpec]


def fc(name, in_features, out_features):
    # 创建全连接层的结构描述。
    # 这里只记录输入输出维度，不创建真正的 nn.Linear。
    return LayerSpec(
        layer_type="fc",
        in_features=in_features,
        out_features=out_features,
        name=name
    )


def conv(name, in_channels, out_channels, kernel_size, padding, stride,
         input_shape=None, output_shape=None):
    # 创建卷积层的结构描述。
    # input_shape 和 output_shape 主要是为了和论文中的网络结构表逐项对应。
    return LayerSpec(
        layer_type="conv",
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        input_shape=input_shape,
        output_shape=output_shape,
        name=name
    )


def bn(name, num_features):
    # 创建 BN 层的结构描述。
    # 具体是 BatchNorm1d 还是 BatchNorm2d，会在构建 PyTorch 模型时根据前一层类型判断。
    return LayerSpec(
        layer_type="bn",
        num_features=num_features,
        name=name
    )


def relu(name):
    # 创建 ReLU 层描述。
    # 在明文模型中对应普通 ReLU，在 MPC 中会对应安全激活函数。
    return LayerSpec(
        layer_type="relu",
        name=name
    )


def maxpool(name, kernel_size=2, stride=2,
            input_shape=None, output_shape=None):
    # 创建最大池化层描述。
    # Sonic 论文中的 CNN 模型多次使用 2x2、stride=2 的池化来降低空间尺寸。
    return LayerSpec(
        layer_type="maxpool",
        kernel_size=kernel_size,
        stride=stride,
        input_shape=input_shape,
        output_shape=output_shape,
        name=name
    )


def flatten(name):
    # 创建 Flatten 层描述。
    # 通常用于把卷积输出从 [N, C, H, W] 拉平成 [N, C*H*W]，
    # 以便送入后面的全连接层。
    return LayerSpec(
        layer_type="flatten",
        name=name
    )


# ============================================================
# M1: MNIST
#
# Paper Table 13:
#   FC (784 -> 128) - BN - ReLU
#   FC (128 -> 128) - BN - ReLU
#   FC (128 -> 10)  - BN
#
# M1 是一个纯全连接网络。
# MNIST 输入是 1x28x28，flatten 后正好是 784 维。
# 这个模型比较适合先验证 SFC、SBN、ReLU 这些基础模块。
# ============================================================

M1_SPEC = ModelSpec(
    name="M1",
    dataset="MNIST",
    input_shape=(1, 28, 28),
    num_classes=10,
    layers=[
        flatten("flatten"),

        fc("fc1", 784, 128),
        bn("bn1", 128),
        relu("relu1"),

        fc("fc2", 128, 128),
        bn("bn2", 128),
        relu("relu2"),

        fc("fc3", 128, 10),
        bn("bn3", 10),
    ]
)


# ============================================================
# M2: MNIST
#
# Paper Table 14:
#   CONV 1x28x28 -> 16x24x24, kernel 1x16x5x5, stride 1
#   BN - ReLU
#   MP 16x24x24 -> 16x12x12, window 2x2, stride 2
#
#   CONV 16x12x12 -> 16x8x8, kernel 16x16x5x5, stride 1
#   BN - ReLU
#   MP 16x8x8 -> 16x4x4, window 2x2, stride 2
#
#   FC 256 -> 100 - BN - ReLU
#   FC 100 -> 10  - BN
#
# M2 是 MNIST 上的 CNN 结构。
# 它比 M1 多了卷积和池化，适合验证 SConv、MaxPool、Flatten 和 FC 的衔接。
# ============================================================

M2_SPEC = ModelSpec(
    name="M2",
    dataset="MNIST",
    input_shape=(1, 28, 28),
    num_classes=10,
    layers=[
        conv(
            "conv1",
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            padding=0,
            stride=1,
            input_shape=(1, 28, 28),
            output_shape=(16, 24, 24)
        ),
        bn("bn1", 16),
        relu("relu1"),

        maxpool(
            "mp1",
            kernel_size=2,
            stride=2,
            input_shape=(16, 24, 24),
            output_shape=(16, 12, 12)
        ),

        conv(
            "conv2",
            in_channels=16,
            out_channels=16,
            kernel_size=5,
            padding=0,
            stride=1,
            input_shape=(16, 12, 12),
            output_shape=(16, 8, 8)
        ),
        bn("bn2", 16),
        relu("relu2"),

        maxpool(
            "mp2",
            kernel_size=2,
            stride=2,
            input_shape=(16, 8, 8),
            output_shape=(16, 4, 4)
        ),

        flatten("flatten"),

        # 经过第二次池化后，特征图大小为 16x4x4，
        # flatten 后就是 256 维，因此 fc1 输入维度为 256。
        fc("fc1", 256, 100),
        bn("bn3", 100),
        relu("relu3"),

        fc("fc2", 100, 10),
        bn("bn4", 10),
    ]
)


# ============================================================
# C1: CIFAR-10
#
# Paper Table 15:
#   CONV 3x32x32   -> 64x30x30, kernel 3x64x3x3
#   CONV 64x30x30  -> 64x28x28, kernel 64x64x3x3
#   MP   64x28x28  -> 64x14x14
#   CONV 64x14x14  -> 64x12x12
#   CONV 64x12x12  -> 64x10x10
#   MP   64x10x10  -> 64x5x5
#   CONV 64x5x5    -> 64x3x3
#   CONV 64x3x3    -> 64x3x3, padding 0 in paper table means same-size handling here uses padding=1
#   CONV 64x3x3    -> 16x3x3, padding 0 in paper table means same-size handling here uses padding=1
#   FC   144 -> 10 - BN
#
# Note:
#   The paper table denotes padding as "0" for the last two 3x3 -> 3x3 convs.
#   In PyTorch, keeping 3x3 -> 3x3 with kernel=3 requires padding=1.
#   We use padding=1 to match the output shape stated in the paper.
#
# C1 是 CIFAR-10 上更深的卷积模型。
# 这里重点是按论文表格保持每一层的输入输出尺寸一致，
# 因为后续 MPC 复现时每层 shape 一旦不一致，参数导出和安全推理都会出错。
# ============================================================

C1_SPEC = ModelSpec(
    name="C1",
    dataset="CIFAR-10",
    input_shape=(3, 32, 32),
    num_classes=10,
    layers=[
        conv("conv1", 3, 64, 3, 0, 1, (3, 32, 32), (64, 30, 30)),
        bn("bn1", 64),
        relu("relu1"),

        conv("conv2", 64, 64, 3, 0, 1, (64, 30, 30), (64, 28, 28)),
        bn("bn2", 64),
        relu("relu2"),

        maxpool("mp1", 2, 2, (64, 28, 28), (64, 14, 14)),

        conv("conv3", 64, 64, 3, 0, 1, (64, 14, 14), (64, 12, 12)),
        bn("bn3", 64),
        relu("relu3"),

        conv("conv4", 64, 64, 3, 0, 1, (64, 12, 12), (64, 10, 10)),
        bn("bn4", 64),
        relu("relu4"),

        maxpool("mp2", 2, 2, (64, 10, 10), (64, 5, 5)),

        conv("conv5", 64, 64, 3, 0, 1, (64, 5, 5), (64, 3, 3)),
        bn("bn5", 64),
        relu("relu5"),

        # 这里使用 padding=1，是为了让 3x3 输入经过 3x3 卷积后仍保持 3x3 输出。
        # 这一步主要是为了和论文表格中给出的输出尺寸保持一致。
        conv("conv6", 64, 64, 3, 1, 1, (64, 3, 3), (64, 3, 3)),
        bn("bn6", 64),
        relu("relu6"),

        # 输出通道从 64 降到 16，但空间尺寸仍保持 3x3。
        # flatten 后得到 16*3*3 = 144 维，对应后面的 FC 输入。
        conv("conv7", 64, 16, 3, 1, 1, (64, 3, 3), (16, 3, 3)),
        bn("bn7", 16),
        relu("relu7"),

        flatten("flatten"),

        fc("fc1", 144, 10),
        bn("bn8", 10),
    ]
)


# ============================================================
# C2: CIFAR-10
#
# Paper Table 16:
#   CONV 3x32x32   -> 16x32x32
#   CONV 16x32x32  -> 16x32x32
#   CONV 16x32x32  -> 16x32x32
#   MP   16x32x32  -> 16x16x16
#
#   CONV 16x16x16  -> 32x16x16
#   CONV 32x16x16  -> 32x16x16
#   CONV 32x16x16  -> 32x16x16
#   MP   32x16x16  -> 32x8x8
#
#   CONV 32x8x8    -> 48x6x6
#   CONV 48x6x6    -> 48x4x4
#   CONV 48x4x4    -> 64x2x2
#   MP   64x2x2    -> 64x1x1
#   FC   64 -> 10 - BN
#
# C2 同样用于 CIFAR-10，但结构和 C1 不同。
# 前半部分通过 padding=1 保持空间尺寸，后半部分逐步缩小特征图，
# 最终池化到 64x1x1，再接全连接分类层。
# ============================================================

C2_SPEC = ModelSpec(
    name="C2",
    dataset="CIFAR-10",
    input_shape=(3, 32, 32),
    num_classes=10,
    layers=[
        conv("conv1", 3, 16, 3, 1, 1, (3, 32, 32), (16, 32, 32)),
        bn("bn1", 16),
        relu("relu1"),

        conv("conv2", 16, 16, 3, 1, 1, (16, 32, 32), (16, 32, 32)),
        bn("bn2", 16),
        relu("relu2"),

        conv("conv3", 16, 16, 3, 1, 1, (16, 32, 32), (16, 32, 32)),
        bn("bn3", 16),
        relu("relu3"),

        maxpool("mp1", 2, 2, (16, 32, 32), (16, 16, 16)),

        conv("conv4", 16, 32, 3, 1, 1, (16, 16, 16), (32, 16, 16)),
        bn("bn4", 32),
        relu("relu4"),

        conv("conv5", 32, 32, 3, 1, 1, (32, 16, 16), (32, 16, 16)),
        bn("bn5", 32),
        relu("relu5"),

        conv("conv6", 32, 32, 3, 1, 1, (32, 16, 16), (32, 16, 16)),
        bn("bn6", 32),
        relu("relu6"),

        maxpool("mp2", 2, 2, (32, 16, 16), (32, 8, 8)),

        # 从这里开始不再 padding，空间尺寸会逐层缩小。
        conv("conv7", 32, 48, 3, 0, 1, (32, 8, 8), (48, 6, 6)),
        bn("bn7", 48),
        relu("relu7"),

        conv("conv8", 48, 48, 3, 0, 1, (48, 6, 6), (48, 4, 4)),
        bn("bn8", 48),
        relu("relu8"),

        conv("conv9", 48, 64, 3, 0, 1, (48, 4, 4), (64, 2, 2)),
        bn("bn9", 64),
        relu("relu9"),

        # 64x2x2 经过 2x2 池化后变成 64x1x1。
        maxpool("mp3", 2, 2, (64, 2, 2), (64, 1, 1)),

        flatten("flatten"),

        fc("fc1", 64, 10),
        bn("bn10", 10),
    ]
)


# 统一管理所有 Sonic 论文模型规格。
# 后续通过字符串名称就可以取到对应结构，避免在多个文件中重复写模型配置。
SONIC_MODEL_SPECS = {
    "M1": M1_SPEC,
    "M2": M2_SPEC,
    "C1": C1_SPEC,
    "C2": C2_SPEC,
}


def get_sonic_model_spec(name):
    # 为了使用方便，模型名统一转成大写。
    # 这样传入 "m1" 或 "M1" 都可以正常匹配。
    name = name.upper()

    if name not in SONIC_MODEL_SPECS:
        raise KeyError(f"Unknown Sonic model name: {name}")

    return SONIC_MODEL_SPECS[name]


def print_model_spec(spec):
    # 打印模型结构，主要用于检查当前定义是否和论文表格一致。
    print("=" * 80)
    print(f"Model: {spec.name}")
    print(f"Dataset: {spec.dataset}")
    print(f"Input shape: {spec.input_shape}")
    print(f"Num classes: {spec.num_classes}")
    print("=" * 80)

    for idx, layer in enumerate(spec.layers, start=1):
        if layer.layer_type == "conv":
            print(
                f"{idx:02d}. {layer.name}: CONV "
                f"{layer.in_channels}->{layer.out_channels}, "
                f"k={layer.kernel_size}, pad={layer.padding}, stride={layer.stride}, "
                f"{layer.input_shape} -> {layer.output_shape}"
            )

        elif layer.layer_type == "fc":
            print(
                f"{idx:02d}. {layer.name}: FC "
                f"{layer.in_features}->{layer.out_features}"
            )

        elif layer.layer_type == "bn":
            print(
                f"{idx:02d}. {layer.name}: BN "
                f"num_features={layer.num_features}"
            )

        elif layer.layer_type == "relu":
            print(f"{idx:02d}. {layer.name}: ReLU")

        elif layer.layer_type == "maxpool":
            print(
                f"{idx:02d}. {layer.name}: MaxPool "
                f"k={layer.kernel_size}, stride={layer.stride}, "
                f"{layer.input_shape} -> {layer.output_shape}"
            )

        elif layer.layer_type == "flatten":
            print(f"{idx:02d}. {layer.name}: Flatten")

        else:
            # 兜底输出，防止后续扩展新层时打印函数直接报错。
            print(f"{idx:02d}. {layer.name}: {layer.layer_type}")

    print()