import numpy as np

from mpc.share import MOD
from net.socket_utils import send_data, recv_data
from net.profiler import inc, time_block


def _ring_add(x, y):
    """
    Z_(2^32) 环上的加法。
    这里统一转成 uint64 做中间计算，再取 mod 2^32，最后转回 uint32。
    """
    return ((x.astype(np.uint64) + y.astype(np.uint64)) % MOD).astype(np.uint32)


def _ring_sub(x, y):
    """
    Z_(2^32) 环上的减法。
    注意这里不能直接用普通有符号减法，否则负数会影响环表示。
    """
    return ((x.astype(np.uint64) - y.astype(np.uint64)) % MOD).astype(np.uint32)


def _ring_matmul(x, y):
    """
    Z_(2^32) 环上的矩阵乘法。

    普通矩阵乘法是：
        Z = X @ Y

    在环上计算时，最终结果需要 mod 2^32。
    numpy 的 uint64 中间乘加即使发生溢出，最后再转 uint32，
    本质上也等价于在 2^32 环中取低 32 位。
    """
    return ((x.astype(np.uint64) @ y.astype(np.uint64)) % MOD).astype(np.uint32)


def _open_matrix_values(conn, party_id, e_i, f_i):
    """
    打开矩阵级 Beaver 协议中的 E 和 F。

    每一方本地有：
        E_i = X_i - A_i
        F_i = W_i - B_i

    双方交换后重构：
        E = E_0 + E_1
        F = F_0 + F_1

    E 和 F 可以公开，因为 A、B 是随机矩阵掩码。
    """
    if party_id == 0:
        send_data(conn, (e_i, f_i))
        e_j, f_j = recv_data(conn)
    else:
        e_j, f_j = recv_data(conn)
        send_data(conn, (e_i, f_i))

    e = _ring_add(e_i, e_j)
    f = _ring_add(f_i, f_j)

    return e, f


def secure_matmul(x_i, w_i, triple_i, conn, party_id):
    """
    矩阵级 Beaver triple 安全矩阵乘法。

    目标：
        Z = X @ W

    两方分别持有：
        X_i, W_i

    offline 阶段准备矩阵 triple：
        A, B, C
        C = A @ B

    online 阶段：
        E = X - A
        F = W - B

    展开推导：
        X @ W
      = (A + E) @ (B + F)
      = A@B + E@B + A@F + E@F
      = C + E@B + A@F + E@F

    其中 E 和 F 是打开后的公开矩阵。
    E@F 是公开项，只需要其中一方加一次，避免重复。
    """

    inc("secure_matmul_calls")

    with time_block("secure_matmul_time"):
        a_i, b_i, c_i = triple_i

        if x_i.shape != a_i.shape:
            raise ValueError(
                f"x_i shape {x_i.shape} does not match A_i shape {a_i.shape}"
            )

        if w_i.shape != b_i.shape:
            raise ValueError(
                f"w_i shape {w_i.shape} does not match B_i shape {b_i.shape}"
            )

        # E_i = X_i - A_i
        e_i = _ring_sub(x_i, a_i)

        # F_i = W_i - B_i
        f_i = _ring_sub(w_i, b_i)

        # 打开 E 和 F。
        e, f = _open_matrix_values(
            conn=conn,
            party_id=party_id,
            e_i=e_i,
            f_i=f_i
        )

        # 当前方先拿到 C_i。
        z_i = c_i.astype(np.uint32).copy()

        # 加上 E @ B_i。
        z_i = _ring_add(z_i, _ring_matmul(e, b_i))

        # 加上 A_i @ F。
        z_i = _ring_add(z_i, _ring_matmul(a_i, f))

        # E @ F 是公开项，只让 party0 加一次，避免两方重复加。
        if party_id == 0:
            z_i = _ring_add(z_i, _ring_matmul(e, f))

        return z_i.astype(np.uint32)