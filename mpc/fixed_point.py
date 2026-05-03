import numpy as np
from mpc.share import MOD


def encode_fixed(x, scale_bits=8):
    """
    float / int -> fixed-point ring element

    例如：
        scale_bits = 8
        1.5 -> round(1.5 * 2^8) = 384

    这个函数把浮点数编码到 Z_(2^32) 环中。
    在 MPC 协议里不能直接处理浮点数，所以需要先把小数放大成整数，
    再用 uint32 表示为环元素。
    """
    # scale 表示定点数的小数缩放倍数。
    # scale_bits=8 时，scale=256。
    scale = 1 << scale_bits

    # 先乘以 scale，再四舍五入成整数。
    # 使用 int64 作为中间类型，是为了避免编码时数值范围过早溢出。
    x_int = np.round(np.array(x, dtype=np.float64) * scale).astype(np.int64)

    # 映射到 Z_(2^32) 环中。
    # 负数会通过取模变成对应的二补码形式。
    return (x_int % MOD).astype(np.uint32)


def decode_fixed(x_ring, scale_bits=8):
    """
    fixed-point ring element -> float

    这个函数用于把环上的定点数结果解码回浮点数。
    一般在测试或对比 PyTorch 明文输出时使用。
    """
    # uint32 先 view 成 int32，可以恢复二补码下的有符号含义。
    # 例如环上的大数会被解释成负数。
    x_signed = x_ring.astype(np.uint32).view(np.int32).astype(np.float64)

    # 再除以 scale，恢复小数尺度。
    scale = 1 << scale_bits
    return x_signed / scale


def to_signed_int(x_ring):
    """
    ring element -> int32 signed

    将 Z_(2^32) 环元素解释为 int32 有符号整数。
    这个函数常用于调试中查看负数是否编码正确。
    """
    return x_ring.astype(np.uint32).view(np.int32)


def encode_bias_for_output_scale(bias_float, output_scale_bits):
    """
    bias 要编码到当前层输出尺度。

    如果：
        X scale = 2^8
        W scale = 2^8
    那么：
        XW scale = 2^16

    所以这一层 bias 也必须编码到 2^16。

    注意 bias 的尺度要和加法发生时的输出尺度一致。
    如果 bias 仍然按 2^8 编码，却和 2^16 的 XW 相加，
    最终结果会出现明显偏差。
    """
    return encode_fixed(bias_float, output_scale_bits)