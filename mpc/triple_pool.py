from net.socket_utils import send_data, recv_data
from net.profiler import inc

from mpc.bit_triple_ot import generate_bit_triples_ot
from mpc.arith_triple_ot import generate_arith_triples_ot


# arithmetic triple 缓存池。
# key 是 shape，value 是对应 shape 的 triple 列表。
_ARITH_POOL = {}

# Boolean triple 缓存池。
_BIT_POOL = {}

# 矩阵 Beaver triple 缓存池。
# key 是 (x_shape, w_shape)，value 是对应矩阵乘法 triple 列表。
#
# 对于矩阵乘法：
#   Z = X @ W
#
# 需要矩阵 triple：
#   A.shape = X.shape
#   B.shape = W.shape
#   C.shape = X @ W 的输出 shape
#   C = A @ B
_MATMUL_POOL = {}


def _shape_key(shape):
    """
    统一把 shape 转成 tuple，作为字典 key。
    """
    return tuple(shape)


def _matmul_key(x_shape, w_shape):
    """
    矩阵 triple 的 key 由输入矩阵形状和权重矩阵形状共同决定。
    """
    return tuple(x_shape), tuple(w_shape)


def clear_triple_pools():
    """
    清空本地 triple pool。

    每次新测试或新推理前清空，可以避免使用到上一轮剩余的 triple。
    """
    global _ARITH_POOL, _BIT_POOL, _MATMUL_POOL

    _ARITH_POOL = {}
    _BIT_POOL = {}
    _MATMUL_POOL = {}


def _push_arith(shape, triples):
    """
    将一批 arithmetic triples 放入 pool。
    """
    key = _shape_key(shape)

    if key not in _ARITH_POOL:
        _ARITH_POOL[key] = []

    _ARITH_POOL[key].extend(triples)


def _push_bit(shape, triples):
    """
    将一批 Boolean triples 放入 pool。
    """
    key = _shape_key(shape)

    if key not in _BIT_POOL:
        _BIT_POOL[key] = []

    _BIT_POOL[key].extend(triples)


def _push_matmul(x_shape, w_shape, triples):
    """
    将一批矩阵 Beaver triples 放入 pool。

    每个 triple 的格式为：
        (A_i, B_i, C_i)

    并且完整矩阵满足：
        C = A @ B
    """
    key = _matmul_key(x_shape, w_shape)

    if key not in _MATMUL_POOL:
        _MATMUL_POOL[key] = []

    _MATMUL_POOL[key].extend(triples)


def pop_arith_triple(shape):
    """
    从 arithmetic pool 中取出一个指定 shape 的 triple。

    secure_mul 会通过这个接口拿预处理好的 Beaver triple。
    """
    key = _shape_key(shape)

    if key not in _ARITH_POOL or len(_ARITH_POOL[key]) == 0:
        raise RuntimeError(f"Arithmetic triple pool exhausted for shape={key}")

    return _ARITH_POOL[key].pop()


def pop_bit_triple(shape):
    """
    从 Boolean pool 中取出一个指定 shape 的 triple。

    bit_and 会通过这个接口拿预处理好的 Boolean triple。
    """
    key = _shape_key(shape)

    if key not in _BIT_POOL or len(_BIT_POOL[key]) == 0:
        raise RuntimeError(f"Boolean triple pool exhausted for shape={key}")

    return _BIT_POOL[key].pop()


def pop_matmul_triple(x_shape, w_shape):
    """
    从矩阵 triple pool 中取出一个指定矩阵乘法形状的 triple。

    用于 secure_matmul：
        X.shape = x_shape
        W.shape = w_shape
        Z = X @ W
    """
    key = _matmul_key(x_shape, w_shape)

    if key not in _MATMUL_POOL or len(_MATMUL_POOL[key]) == 0:
        raise RuntimeError(
            f"Matmul triple pool exhausted for x_shape={tuple(x_shape)}, "
            f"w_shape={tuple(w_shape)}"
        )

    return _MATMUL_POOL[key].pop()


def has_matmul_triple(x_shape, w_shape):
    """
    判断当前 pool 中是否有可用的矩阵 triple。

    这个函数主要用于兼容旧测试：
    如果没有提前准备矩阵 triple，linear_secret_weight 可以退回旧版逐项 secure_mul。
    """
    key = _matmul_key(x_shape, w_shape)
    return key in _MATMUL_POOL and len(_MATMUL_POOL[key]) > 0


def _setup_arith_pool_by_ot_extension(conn, party_id, arith_plan):
    """
    使用 OT Extension 批量生成 Arithmetic Beaver triples。

    生成后直接放入本地 offline pool。
    arith_plan 中每一项表示：
        shape: triple 的张量形状
        count: 需要生成的数量
    """
    for shape, count in arith_plan:
        triples = generate_arith_triples_ot(
            conn=conn,
            party_id=party_id,
            shape=shape,
            count=count
        )

        _push_arith(shape, triples)
        inc("offline_arith_triples", count)


def _setup_bit_pool_by_ot_extension(conn, party_id, bit_plan):
    """
    使用 OT Extension 批量生成 Boolean Beaver triples。

    生成后直接放入本地 offline pool。
    bit_plan 的格式和 arith_plan 类似，
    只是生成的是用于 bit_and 的 Boolean triple。
    """
    for shape, count in bit_plan:
        triples = generate_bit_triples_ot(
            conn=conn,
            party_id=party_id,
            shape=shape,
            count=count
        )

        _push_bit(shape, triples)
        inc("offline_bit_triples", count)


def setup_triple_pool(conn, party_id, arith_plan, bit_plan):
    """
    Offline / Online 分离版 triple pool。

    当前版本：
        Arithmetic triple：OT Extension 批量生成
        Boolean triple   ：OT Extension 批量生成

    注意：
        这里暂时没有用 OT Extension 生成矩阵 triple。
        矩阵 triple 当前先通过 trusted dealer 版本生成，
        也就是 mpc/triple_dealer.py 中的 setup_triple_pool_by_dealer。

    Online 阶段：
        secure_mul / bit_and / secure_matmul 只从 pool 取 triple。
    """
    clear_triple_pools()

    _setup_arith_pool_by_ot_extension(
        conn=conn,
        party_id=party_id,
        arith_plan=arith_plan
    )

    _setup_bit_pool_by_ot_extension(
        conn=conn,
        party_id=party_id,
        bit_plan=bit_plan
    )

    # 双方同步 offline 阶段结束。
    if party_id == 0:
        send_data(conn, ("OFFLINE_DONE", None, None))
    else:
        tag, _, _ = recv_data(conn)
        if tag != "OFFLINE_DONE":
            raise RuntimeError(f"Unexpected offline finish tag: {tag}")