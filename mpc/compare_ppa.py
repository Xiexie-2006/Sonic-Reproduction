import numpy as np

from mpc.bit_and import bit_and
from net.profiler import inc, time_block


def _bit_stack(x, bit_len=32):
    """
    将 uint32 环元素拆成 bit 平面。

    输入：
        x.shape = [N, ...]

    输出：
        bits.shape = [bit_len, N, ...]

    bits[i] 表示第 i 位。
    """
    x = x.astype(np.uint32)

    bits = [
        ((x >> i) & 1).astype(np.uint32)
        for i in range(bit_len)
    ]

    return np.stack(bits, axis=0).astype(np.uint32)


def _zero_bit_stack_like(x, bit_len=32):
    """
    生成与 _bit_stack(x) 相同形状的全 0 bit share。
    """
    return np.zeros(
        (bit_len,) + tuple(x.shape),
        dtype=np.uint32
    )


def _prefix_combine(g_high, p_high, g_low, p_low, conn, party_id):
    """
    PPA 中的 generate / propagate 区间合并。

    对两个相邻区间：
        low  表示低位区间
        high 表示高位区间

    合并公式为：
        G = G_high OR (P_high AND G_low)
        P = P_high AND P_low

    在加法器 generate/propagate 语义中，
    G_high 和 P_high AND G_low 不会同时为 1，
    因此 OR 可以用 XOR 表示。
    """

    # P_high AND G_low
    p_and_g = bit_and(
        xi=p_high,
        yi=g_low,
        conn=conn,
        party_id=party_id,
    )

    # G_high OR (P_high AND G_low)
    new_g = (g_high ^ p_and_g).astype(np.uint32)

    # P_high AND P_low
    new_p = bit_and(
        xi=p_high,
        yi=p_low,
        conn=conn,
        party_id=party_id,
    ).astype(np.uint32)

    return new_g, new_p


def secure_msb_ppa(xi, conn, party_id, bit_len=32):
    """
    PPA 版 secure MSB。

    目标：
        在不公开 x 的情况下，得到 x = x0 + x1 mod 2^32 的最高位 MSB。

    原版做法：
        逐位 full-adder 传播 carry：
        carry_0 -> carry_1 -> ... -> carry_31
        通信轮数约为 O(l)

    PPA 做法：
        使用 generate / propagate 并行前缀结构：
        step = 1, 2, 4, 8, 16
        通信轮数约为 O(log l)

    注意：
        这里的 bit_and 是批量调用的。
        例如 step=1 时，不是对 31 个 bit 分别通信，
        而是把 31 个 bit 打包成一个张量统一做 bit_and。
    """
    inc("ppa_msb_calls")

    with time_block("ppa_msb_time"):
        xi = xi.astype(np.uint32)

        # --------------------------------------------------------
        # 1. 构造两方加法的 bit share
        # --------------------------------------------------------
        # party0 贡献 x0 的 bit；
        # party1 贡献 x1 的 bit。
        #
        # 从整体看：
        #   x = x0 + x1 mod 2^32
        #
        # 我们把它看成两个 bit 数组相加：
        #   a = x0
        #   b = x1
        if party_id == 0:
            a_bits = _bit_stack(xi, bit_len)
            b_bits = _zero_bit_stack_like(xi, bit_len)
        else:
            a_bits = _zero_bit_stack_like(xi, bit_len)
            b_bits = _bit_stack(xi, bit_len)

        # --------------------------------------------------------
        # 2. 初始化 propagate 和 generate
        # --------------------------------------------------------
        # p_i = a_i XOR b_i
        # g_i = a_i AND b_i
        #
        # p_i 可以本地 XOR。
        # g_i 涉及 AND，需要安全 bit_and。
        p_init = (a_bits ^ b_bits).astype(np.uint32)

        g_init = bit_and(
            xi=a_bits,
            yi=b_bits,
            conn=conn,
            party_id=party_id,
        ).astype(np.uint32)

        p = p_init.copy()
        g = g_init.copy()

        # --------------------------------------------------------
        # 3. PPA 并行前缀传播
        # --------------------------------------------------------
        # 32 bit 下：
        #   step=1
        #   step=2
        #   step=4
        #   step=8
        #   step=16
        #
        # 共 5 层。
        step = 1
        levels = 0

        while step < bit_len:
            g_old = g.copy()
            p_old = p.copy()

            high = slice(step, bit_len)
            low = slice(0, bit_len - step)

            new_g, new_p = _prefix_combine(
                g_high=g_old[high],
                p_high=p_old[high],
                g_low=g_old[low],
                p_low=p_old[low],
                conn=conn,
                party_id=party_id,
            )

            g[high] = new_g
            p[high] = new_p

            step *= 2
            levels += 1

        inc("ppa_levels", levels)

        # --------------------------------------------------------
        # 4. 计算最高位 sum，也就是 MSB
        # --------------------------------------------------------
        # g[i] 表示从 bit0 到 bit i 的整体 generate，
        # 也就是 carry_{i+1}。
        #
        # 第 31 位的 sum:
        #   sum_31 = p_31 XOR carry_31
        #
        # carry_31 来自低 31 位，也就是 g[30]。
        if bit_len == 1:
            carry_into_msb = np.zeros_like(xi, dtype=np.uint32)
        else:
            carry_into_msb = g[bit_len - 2].astype(np.uint32)

        msb_share = (p_init[bit_len - 1] ^ carry_into_msb).astype(np.uint32)

        return msb_share


def secure_compare_zero_ppa(xi, conn, party_id):
    """
    PPA 版 secure compare zero。

    二补码中：
        MSB = 0 表示 x >= 0
        MSB = 1 表示 x < 0

    所以：
        x >= 0 等价于 NOT(MSB)

    Boolean share 下取反，只需要让其中一方 XOR 1。
    """
    msb_i = secure_msb_ppa(
        xi=xi,
        conn=conn,
        party_id=party_id,
        bit_len=32,
    )

    if party_id == 0:
        return (msb_i ^ 1).astype(np.uint32)

    return msb_i.astype(np.uint32)


def build_ppa_bit_plan(value_shape, bit_len=32):
    """
    为 PPA secure MSB 生成 bit triple plan。

    旧版逐位 carry 的通信会比较碎。
    PPA 这里按张量批量做 bit_and。

    需要的 bit_and 调用：

    1. 初始 generate:
        shape = (32, *value_shape)

    2. prefix combine:
        step=1  -> shape = (31, *value_shape)，需要 2 次 bit_and
        step=2  -> shape = (30, *value_shape)，需要 2 次 bit_and
        step=4  -> shape = (28, *value_shape)，需要 2 次 bit_and
        step=8  -> shape = (24, *value_shape)，需要 2 次 bit_and
        step=16 -> shape = (16, *value_shape)，需要 2 次 bit_and

    总 bit_and 调用次数：
        1 + 2 * 5 = 11
    """
    value_shape = tuple(value_shape)

    plan = [
        ((bit_len,) + value_shape, 1)
    ]

    step = 1
    while step < bit_len:
        plan.append(
            ((bit_len - step,) + value_shape, 2)
        )
        step *= 2

    return plan