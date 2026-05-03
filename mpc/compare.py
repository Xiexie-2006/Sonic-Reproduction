import numpy as np
from mpc.bit_and import bit_and


def _bit(x, i):
    # 取出 x 的第 i 位。
    # 这里 x 是 uint32 数组，右移后取最低位即可。
    return ((x >> i) & 1).astype(np.uint32)


def secure_msb(xi, conn, party_id):
    """
    Compute MSB(x0 + x1 mod 2^32) as Boolean share.

    Party 0 owns x0.
    Party 1 owns x1.

    We build a Boolean full-adder circuit:
        sum_i = a_i xor b_i xor carry_i
        carry_{i+1} = (a_i & b_i) xor (carry_i & (a_i xor b_i))

    这个函数用于安全计算 x 的最高位，也就是符号位。
    在二补码表示下，最高位为 1 通常表示负数，
    因此它是 secure_compare_zero 和安全 ReLU 的基础。
    """
    # carry 表示逐位加法过程中的进位。
    # 从最低位开始加，初始进位为 0。
    carry = np.zeros_like(xi, dtype=np.uint32)

    # 最终保存第 31 位，也就是 uint32 的最高位。
    msb_share = None

    for i in range(32):
        # party0 持有 x0 的 bit，party1 持有 x1 的 bit。
        # 为了构造 x0 + x1 的逐位加法，
        # 当前方只取自己的 share，另一方位置用 0 占位。
        if party_id == 0:
            ai = _bit(xi, i)
            bi = np.zeros_like(ai, dtype=np.uint32)
        else:
            ai = np.zeros_like(xi, dtype=np.uint32)
            bi = _bit(xi, i)

        # 当前位不考虑进位时的 XOR 结果。
        axb = ai ^ bi

        # 当前位求和：
        #   sum_i = ai XOR bi XOR carry
        sum_i = axb ^ carry

        # 计算下一位进位：
        #   carry_{i+1} = (ai & bi) XOR (carry_i & (ai XOR bi))

        # 这里 AND 涉及秘密 bit，所以要调用安全 bit_and。
        t1 = bit_and(ai, bi, conn, party_id)
        t2 = bit_and(carry, axb, conn, party_id)
        carry = t1 ^ t2

        # 第 31 位就是 MSB。
        if i == 31:
            msb_share = sum_i

    return msb_share.astype(np.uint32)


def secure_compare_zero(xi, conn, party_id):
    """
    Return Boolean share of:
        b = 1 if x >= 0 else 0

    For ReLU, x=0 may return 1 because 1*0 = 0.

    这里的比较是基于二补码符号位完成的：
        MSB = 0 表示非负数
        MSB = 1 表示负数

    所以 x >= 0 的结果可以看作是 NOT(MSB)。
    """
    # 先安全计算 x 的最高位。
    msb_i = secure_msb(xi, conn, party_id)

    # Boolean share 下，对一个 bit 做取反时，
    # 只需要让其中一方 XOR 1 即可。

    # 因此 party0 返回 msb_i ^ 1，
    # party1 保持 msb_i 不变。
    # 两方 XOR 后得到的就是 1 - MSB。
    if party_id == 0:
        return (msb_i ^ 1).astype(np.uint32)
    else:
        return msb_i.astype(np.uint32)