import numpy as np

# 当前项目中使用的整数环大小。
# 2^32 对应 uint32 的自然溢出范围，也是 fixed-point 编码和 arithmetic sharing 的基础。
MOD = 2**32


# Arithmetic sharing
def share_arith(x):
    # 算术秘密分享：
    # 将一个秘密值 x 拆成 x0 和 x1，使得：
    #   x = x0 + x1 mod 2^32
    #
    # 其中 x0 随机生成，x1 由 x - x0 得到。
    # 单独看 x0 或 x1 都无法还原原始 x。
    x0 = np.random.randint(0, MOD, size=x.shape, dtype=np.uint32)

    # 在 Z_(2^32) 环上计算另一份 share。
    # 即使 x - x0 出现负数，也会通过 mod 映射回环中。
    x1 = (x - x0) % MOD

    return x0, x1


def reconstruct_arith(x0, x1):
    # 算术分享重构：
    # 两方 share 相加并对 MOD 取模，就能恢复原始环元素。
    return (x0 + x1) % MOD


# Boolean sharing
def share_bool(x):
    # 布尔秘密分享：
    # 将 bit 或 bit 数组 x 拆成 x0 和 x1，使得：
    #   x = x0 XOR x1
    #
    # 这里先随机生成一份 r，另一份为 x XOR r。
    r = np.random.randint(0, 2, size=x.shape, dtype=np.uint32)

    return r, x ^ r


def reconstruct_bool(x0, x1):
    # 布尔分享重构：
    # 两方 share 做 XOR，就能恢复原始 bit。
    return x0 ^ x1