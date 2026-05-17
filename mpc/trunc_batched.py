import numpy as np

from mpc.share import MOD
from mpc.a2b import a2b_bits
from mpc.b2a_secure import b2a_secure
from net.profiler import inc, time_block


def _ensure_bit_array(bits):
    """
    将 a2b_bits 的输出统一整理成 ndarray。

    兼容两种情况：
    1. bits 是 list，长度为 bit_len，每个元素 shape = value_shape；
    2. bits 已经是 ndarray，shape = [bit_len, *value_shape]。
    """
    if isinstance(bits, list):
        return np.stack(bits, axis=0).astype(np.uint32)

    return bits.astype(np.uint32)


def secure_trunc_batched(xi, shift_bits, conn, party_id, bit_len=32):
    """
    批量版 secure truncation。

    用途：
        fixed-point 乘法后，scale 从 2^(2f) 变成 2^f，
        因此需要右移 f 位。

    原始 trunc 的常见实现：
        1. A2B 得到 32 个 bit；
        2. 按 signed right shift 重新排列 bit；
        3. 对每个 bit 分别调用 B2A；
        4. 本地重组。

    优化思路：
        1. A2B 得到所有 bit；
        2. 一次性整理出 shift 后的 32 个 bit；
        3. 把 32 个 bit 作为一个张量一次性 B2A；
        4. 用 NumPy 向量化方式本地重组。

    这样可以减少通信切分：
        原来 32 次 B2A
        现在 1 次 batched B2A
    """
    inc("trunc_calls")

    with time_block("trunc_time"):
        xi = xi.astype(np.uint32)

        # --------------------------------------------------------
        # 1. Arithmetic share -> Boolean bit share
        # --------------------------------------------------------
        # bits[j] 表示 x 的第 j 位 Boolean share。
        bits = a2b_bits(
            xi=xi,
            conn=conn,
            party_id=party_id,
            bit_len=bit_len,
        )

        bits = _ensure_bit_array(bits)

        if bits.shape[0] != bit_len:
            raise ValueError(
                f"Expected bit array first dimension {bit_len}, got {bits.shape}"
            )

        # --------------------------------------------------------
        # 2. signed arithmetic right shift
        # --------------------------------------------------------
        # 二补码负数右移时，高位需要补符号位。
        # sign bit 是第 bit_len-1 位。
        shifted_bits = np.zeros_like(bits, dtype=np.uint32)
        sign_bit = bits[bit_len - 1]

        for j in range(bit_len):
            src = j + shift_bits

            if src < bit_len:
                shifted_bits[j] = bits[src]
            else:
                shifted_bits[j] = sign_bit

        # --------------------------------------------------------
        # 3. 一次性 B2A
        # --------------------------------------------------------
        # shifted_bits.shape = [32, *value_shape]
        # 这里一次性把所有 bit 从 Boolean share 转成 Arithmetic share。
        arith_bits = b2a_secure(
            xb_i=shifted_bits,
            conn=conn,
            party_id=party_id,
        ).astype(np.uint32)

        # --------------------------------------------------------
        # 4. NumPy 向量化重组
        # --------------------------------------------------------
        # value = sum_j bit_j * 2^j
        weights = (1 << np.arange(bit_len, dtype=np.uint64))

        reshape_dims = (bit_len,) + (1,) * xi.ndim
        weights = weights.reshape(reshape_dims)

        out = (
            np.sum(arith_bits.astype(np.uint64) * weights, axis=0)
            % MOD
        ).astype(np.uint32)

        return out