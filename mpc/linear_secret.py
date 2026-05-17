import numpy as np

from mpc.share import MOD
from mpc.triple import (
    get_arith_triple,
    get_matmul_triple,
    has_available_matmul_triple,
)
from mpc.mul import secure_mul
from mpc.matmul_secret import secure_matmul
from net.profiler import inc, time_block


def _ring_add(x, y):
    """
    Z_(2^32) 环上的加法。
    """
    return ((x.astype(np.uint64) + y.astype(np.uint64)) % MOD).astype(np.uint32)


def _linear_secret_weight_old_loop(x_i, w_i, b_i, conn, party_id):
    """
    旧版安全线性层实现。

    思路：
        按照矩阵乘法公式逐项展开：
            Y[:, j] = sum_k X[:, k] * W[k, j]

    每一项 X[:,k] * W[k,:] 都调用 secure_mul。
    这个版本功能正确，但通信会被切得比较碎。
    """
    batch, in_dim = x_i.shape
    _, out_dim = w_i.shape

    y_i = np.zeros((batch, out_dim), dtype=np.uint32)

    for k in range(in_dim):
        x_part = x_i[:, k:k + 1]
        w_part = w_i[k:k + 1, :]

        x_term = np.repeat(x_part, out_dim, axis=1)
        w_term = np.repeat(w_part, batch, axis=0)

        triple_i = get_arith_triple(
            conn=conn,
            party_id=party_id,
            shape=x_term.shape
        )

        prod_i = secure_mul(
            xi=x_term,
            yi=w_term,
            triple_i=triple_i,
            conn=conn,
            party_id=party_id
        )

        y_i = _ring_add(y_i, prod_i)

    if b_i is not None:
        y_i = _ring_add(y_i, b_i.reshape(1, -1))

    return y_i.astype(np.uint32)


def _linear_secret_weight_matmul(x_i, w_i, b_i, conn, party_id):
    """
    优化版安全线性层实现。

    思路：
        使用矩阵级 Beaver triple，直接完成：
            Y = X @ W

    offline 阶段准备矩阵 triple：
        A, B, C
        C = A @ B

    online 阶段只需要打开：
        E = X - A
        F = W - B

    然后本地计算：
        Y = C + E@B + A@F + E@F

    相比旧版逐项 secure_mul，这个版本可以把矩阵乘法的通信合并起来。
    """
    triple_i = get_matmul_triple(
        conn=conn,
        party_id=party_id,
        x_shape=x_i.shape,
        w_shape=w_i.shape
    )

    y_i = secure_matmul(
        x_i=x_i,
        w_i=w_i,
        triple_i=triple_i,
        conn=conn,
        party_id=party_id
    )

    if b_i is not None:
        y_i = _ring_add(y_i, b_i.reshape(1, -1))

    return y_i.astype(np.uint32)


def linear_secret_weight(x_i, w_i, b_i, conn, party_id):
    """
    秘密权重安全线性层：
        Y = XW + b

    X、W、b 都是 arithmetic share。

    当前实现包含两种路径：

    1. 优化路径：
        如果 triple_pool 中已经准备了矩阵 triple，
        就调用 secure_matmul()，使用矩阵级 Beaver triple 完成 X@W。

    2. 兼容路径：
        如果没有准备矩阵 triple，
        就退回原来的逐项 secure_mul 版本，
        确保旧测试脚本仍然可以运行。

    这样改的好处：
        先保证原有功能不被破坏；
        后续只要在 offline 阶段加入 matmul_plan，
        就可以自动走矩阵级安全乘法路径。
    """
    inc("linear_secret_calls")

    with time_block("linear_secret_time"):
        x_i = x_i.astype(np.uint32)
        w_i = w_i.astype(np.uint32)

        if b_i is not None:
            b_i = b_i.astype(np.uint32)

        if len(x_i.shape) != 2 or len(w_i.shape) != 2:
            raise ValueError(
                f"linear_secret_weight expects 2D matrices, "
                f"got x_i.shape={x_i.shape}, w_i.shape={w_i.shape}"
            )

        if x_i.shape[1] != w_i.shape[0]:
            raise ValueError(
                f"invalid matrix shapes for XW: "
                f"x_i.shape={x_i.shape}, w_i.shape={w_i.shape}"
            )

        if has_available_matmul_triple(x_i.shape, w_i.shape):
            return _linear_secret_weight_matmul(
                x_i=x_i,
                w_i=w_i,
                b_i=b_i,
                conn=conn,
                party_id=party_id
            )

        return _linear_secret_weight_old_loop(
            x_i=x_i,
            w_i=w_i,
            b_i=b_i,
            conn=conn,
            party_id=party_id
        )