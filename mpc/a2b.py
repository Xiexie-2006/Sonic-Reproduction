import numpy as np
from mpc.bit_and import bit_and
from net.profiler import inc, time_block


def _bit(x, i):
    # 取出 x 的第 i 位。
    # 这里 x 是 uint32 数组，右移 i 位后再和 1 做按位与，
    # 就可以得到对应 bit 位的值。
    return ((x >> i) & 1).astype(np.uint32)


def a2b_bits(xi, conn, party_id, bit_len=32):
    """
    Arithmetic share -> Boolean bit shares

    x = x0 + x1 mod 2^32

    输出：
        bits[i] 是 x 的第 i 位 Boolean share

    这个函数用于把算术秘密分享形式的数据转换成布尔 bit 分享。
    在 Sonic 这类安全推理协议中，线性层更适合用 arithmetic share，
    但比较、符号位判断、ReLU 等操作通常需要 bit 级表示，
    所以 A2B 是连接算术域和布尔域的重要步骤。
    """
    # 记录 A2B 调用次数，方便后续统计协议开销。
    inc("a2b_calls")

    # 统计 A2B 的运行时间。
    with time_block("a2b_time"):
        # carry 表示逐位加法过程中的进位。
        # 初始最低位没有进位，所以全为 0。
        carry = np.zeros_like(xi, dtype=np.uint32)

        # 用于保存每一位的 Boolean share。
        bits = []

        for i in range(bit_len):
            # 在 arithmetic sharing 中：
            # party0 持有 x0，party1 持有 x1。
            #
            # 这里为了做逐位加法，party0 取自己 share 的第 i 位作为 ai，
            # party1 取自己 share 的第 i 位作为 bi。
            if party_id == 0:
                ai = _bit(xi, i)
                bi = np.zeros_like(ai, dtype=np.uint32)
            else:
                ai = np.zeros_like(xi, dtype=np.uint32)
                bi = _bit(xi, i)

            # axb 表示 ai XOR bi。
            # 对单 bit 加法来说，sum = ai XOR bi XOR carry。
            axb = ai ^ bi

            # 当前位的和，也就是最终 x 的第 i 位对应的 Boolean share。
            sum_i = axb ^ carry

            # 计算新的进位：
            # carry_out = (ai & bi) ^ (carry & (ai ^ bi))
            #
            # 这里的 & 不能直接本地做，因为 ai、bi、carry 是秘密分享状态下的 bit。
            # 所以需要调用安全 bit_and 协议。
            t1 = bit_and(ai, bi, conn, party_id)
            t2 = bit_and(carry, axb, conn, party_id)

            # 更新进位，供下一位使用。
            carry = t1 ^ t2

            # 保存当前位结果。
            bits.append(sum_i.astype(np.uint32))

        return bits