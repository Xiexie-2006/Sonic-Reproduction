import numpy as np

from mpc.share import MOD
from mpc.triple import get_arith_triple
from mpc.mul import secure_mul
from mpc.trunc import secure_trunc
from net.profiler import inc, time_block


def _expand_param_like(param_i, target_shape):
    """
    将 BN 参数 share 扩展到输入张量形状。

    支持：
        Conv 特征:
            x shape      = [N, C, H, W]
            param shape  = [C]

        FC 特征:
            x shape      = [N, D]
            param shape  = [D]

        或者 param 本身已经和 x 同形状。

    这个函数主要是为了让 eps1、eps2 可以同时适配卷积层输出和全连接层输出。
    BN 参数通常只按通道或特征维度保存，但真正计算时需要和 x 的形状一致。
    """

    # 先统一成 uint32，保持和 arithmetic share 的环表示一致。
    param_i = np.asarray(param_i, dtype=np.uint32)

    # 如果参数本来就和目标张量同形状，直接复制返回即可。
    if param_i.shape == target_shape:
        return param_i.copy().astype(np.uint32)

    # 卷积特征一般是 [N, C, H, W]。
    # BN 参数如果是 [C]，需要扩展成 [1, C, 1, 1]，
    # 再广播到完整的 [N, C, H, W]。
    if len(target_shape) == 4 and param_i.ndim == 1:
        # [C] -> [1, C, 1, 1] -> [N, C, H, W]
        return np.broadcast_to(
            param_i.reshape(1, param_i.shape[0], 1, 1),
            target_shape
        ).copy().astype(np.uint32)

    # 全连接层输出一般是 [N, D]。
    # BN 参数如果是 [D]，需要扩展成 [1, D]，
    # 再广播到 [N, D]。
    if len(target_shape) == 2 and param_i.ndim == 1:
        # [D] -> [1, D] -> [N, D]
        return np.broadcast_to(
            param_i.reshape(1, param_i.shape[0]),
            target_shape
        ).copy().astype(np.uint32)

    # 其他可广播情况走 numpy 的 broadcast_to。
    # 如果无法广播，就抛出更明确的错误信息，方便定位模型参数 shape 问题。
    try:
        return np.broadcast_to(param_i, target_shape).copy().astype(np.uint32)
    except Exception as e:
        raise ValueError(
            f"Cannot broadcast BN parameter shape {param_i.shape} "
            f"to target shape {target_shape}"
        ) from e


def sbn_fixed(
    x_i,
    eps1_i,
    eps2_i,
    scale_bits,
    conn,
    party_id
):
    """
    Secure Batch Normalization:

        z = eps1 * x + eps2

    fixed-point 说明：
        x      scale = 2^f
        eps1   scale = 2^f
        eps2   scale = 2^(2f)

        eps1 * x 后 scale = 2^(2f)
        加 eps2 后仍为 2^(2f)
        最后 secure_trunc 恢复到 scale = 2^f

    输入：
        x_i:
            当前方持有的输入 arithmetic share

        eps1_i:
            当前方持有的 eps1 arithmetic share
            shape 可以是 [C]、[D] 或和 x_i 同形状

        eps2_i:
            当前方持有的 eps2 arithmetic share
            shape 可以是 [C]、[D] 或和 x_i 同形状

    输出：
        当前方持有的 SBN 输出 arithmetic share

    这里的 SBN 对应论文里的安全 BN 计算。
    推理阶段的 BatchNorm 可以提前整理成一次线性变换，
    所以 MPC 里只需要安全计算 eps1 * x，再加上 eps2。
    """

    # 记录 SBN 调用次数，方便统计整体推理中 BN 层的开销。
    inc("sbn_calls")

    # 统计 SBN 协议运行时间。
    with time_block("sbn_time"):
        target_shape = x_i.shape

        # 将 eps1、eps2 扩展成和 x_i 相同的形状。
        # 这样后面可以直接逐元素做安全乘法和加法。
        eps1_expand_i = _expand_param_like(eps1_i, target_shape)
        eps2_expand_i = _expand_param_like(eps2_i, target_shape)

        # eps1 * x 是秘密分享状态下的乘法，
        # 因此需要 Beaver triple。
        triple_i = get_arith_triple(
            conn=conn,
            party_id=party_id,
            shape=target_shape
        )

        # 安全计算 x * eps1。
        # 此时结果 scale 会从 2^f 变成 2^(2f)。
        prod_i = secure_mul(
            xi=x_i,
            yi=eps1_expand_i,
            triple_i=triple_i,
            conn=conn,
            party_id=party_id
        )

        # 加上 eps2。
        # eps2 已经按照 2^(2f) 编码，所以可以直接和 prod_i 相加。
        raw_i = (
            prod_i.astype(np.uint64)
            + eps2_expand_i.astype(np.uint64)
        ) % MOD
        raw_i = raw_i.astype(np.uint32)

        # 最后做 secure_trunc，把 scale 从 2^(2f) 恢复到 2^f。
        out_i = secure_trunc(
            x_i=raw_i,
            shift_bits=scale_bits,
            conn=conn,
            party_id=party_id
        )

        return out_i