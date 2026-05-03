import numpy as np

from mpc.share import MOD
from mpc.a2b import a2b_bits
from mpc.b2a_secure import b2a_secure
from net.profiler import inc, time_block


def secure_trunc(x_i, shift_bits, conn, party_id):
    """
    精确安全截断：

        y = arithmetic_shift_right(x, shift_bits)

    路线：
        Arithmetic share
        -> A2B bit shares
        -> signed arithmetic right shift
        -> B2A
        -> 按位重组

    在 fixed-point 推理中，乘法会让 scale 放大。
    例如两个 scale=2^f 的数相乘后，scale 会变成 2^(2f)。
    secure_trunc 的作用就是在不公开 x 的情况下右移 shift_bits 位，
    把 fixed-point scale 恢复到目标尺度。
    """
    # 记录安全截断调用次数。
    inc("trunc_calls")

    # 统计安全截断耗时。
    with time_block("trunc_time"):
        # 保持输入为 uint32 环元素。
        x_i = x_i.astype(np.uint32)

        bit_len = 32

        # 第一步：A2B。
        # 将 arithmetic share 转成每一位的 Boolean share。
        bits = a2b_bits(x_i, conn, party_id, bit_len=bit_len)

        # 第 31 位是符号位。
        # 算术右移需要用符号位补高位，才能正确处理负数。
        sign_bit = bits[31]

        shifted_bits = []

        for j in range(bit_len):
            # 右移 shift_bits 位后，新第 j 位来自旧的第 j + shift_bits 位。
            src = j + shift_bits

            if src < bit_len:
                shifted_bits.append(bits[src])
            else:
                # 超出范围的高位用符号位填充。
                # 这对应 signed arithmetic right shift。
                shifted_bits.append(sign_bit)

        # 用于保存重组后的 arithmetic share。
        y_i = np.zeros_like(x_i, dtype=np.uint32)

        for j in range(bit_len):
            # 每一位现在仍然是 Boolean share。
            # 要重组成 arithmetic share，需要先做 B2A。
            bit_a_i = b2a_secure(shifted_bits[j], conn, party_id)

            # 当前 bit 的权重是 2^j。
            weight = np.uint64(1 << j)
            term = (bit_a_i.astype(np.uint64) * weight) % MOD

            # 累加每一位的贡献，得到最终右移后的 arithmetic share。
            y_i = (y_i.astype(np.uint64) + term) % MOD
            y_i = y_i.astype(np.uint32)

        return y_i