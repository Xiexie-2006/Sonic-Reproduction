from mpc.triple_pool import (
    pop_arith_triple,
    pop_bit_triple,
    pop_matmul_triple,
    has_matmul_triple,
)


def get_arith_triple(conn, party_id, shape):
    """
    获取 arithmetic Beaver triple。

    这里保留 conn 和 party_id 参数，是为了和前面直接在线生成 triple 的接口保持一致。
    当前实现中，真正的 triple 来源是 triple_pool。

    arithmetic triple 满足：
        c = a * b mod 2^32

    主要用于：
        secure_mul
        SBN
        SReLU 中的 gate*x
        trunc / B2A 中的算术乘法
    """
    return pop_arith_triple(shape)


def get_bit_triple(conn, party_id, shape):
    """
    获取 Boolean Beaver triple。

    Boolean triple 满足：
        c = a AND b

    主要用于：
        bit_and
        A2B
        secure MSB
        secure compare
    """
    return pop_bit_triple(shape)


def get_matmul_triple(conn, party_id, x_shape, w_shape):
    """
    获取矩阵 Beaver triple。

    矩阵 triple 满足：
        C = A @ B

    其中：
        A.shape = x_shape
        B.shape = w_shape

    用于：
        secure_matmul
        优化版 SFC
        通过 im2col 复用后的 SCONV
    """
    return pop_matmul_triple(x_shape, w_shape)


def has_available_matmul_triple(x_shape, w_shape):
    """
    判断是否存在可用矩阵 triple。

    主要用于 linear_secret_weight 的兼容逻辑：
    - 如果已经准备了矩阵 triple，则使用 secure_matmul；
    - 如果没有准备矩阵 triple，则退回旧版逐项 secure_mul。
    """
    return has_matmul_triple(x_shape, w_shape)