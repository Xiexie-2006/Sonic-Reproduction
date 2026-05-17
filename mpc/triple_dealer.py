import numpy as np

from mpc.share import MOD
import mpc.triple_pool as triple_pool
from net.profiler import inc


def _get_pool_dict(name_candidates):
    """
    从 triple_pool 模块中查找可能存在的 pool 字典。

    这里写多个候选名字，是为了兼容前后版本中不同的变量命名。
    """
    for name in name_candidates:
        if hasattr(triple_pool, name):
            obj = getattr(triple_pool, name)
            if isinstance(obj, dict):
                return obj

    raise RuntimeError(
        "Cannot find triple pool dictionary in mpc.triple_pool.py. "
        "Please send me your mpc/triple_pool.py if this error appears."
    )


def _get_arith_pool():
    """
    获取 arithmetic Beaver triple 的缓存池。
    """
    return _get_pool_dict([
        "_ARITH_POOL",
        "ARITH_POOL",
        "_arith_pool",
        "arith_pool",
        "_arith_triple_pool",
        "arith_triple_pool",
    ])


def _get_bit_pool():
    """
    获取 Boolean Beaver triple 的缓存池。
    """
    return _get_pool_dict([
        "_BIT_POOL",
        "BIT_POOL",
        "_bit_pool",
        "bit_pool",
        "_bit_triple_pool",
        "bit_triple_pool",
    ])


def _get_matmul_pool():
    """
    获取矩阵 Beaver triple 的缓存池。
    """
    return _get_pool_dict([
        "_MATMUL_POOL",
        "MATMUL_POOL",
        "_matmul_pool",
        "matmul_pool",
        "_matmul_triple_pool",
        "matmul_triple_pool",
    ])


def _share_arith_value(x, rng):
    """
    对一个环元素 x 做 arithmetic sharing：

        x = s0 + s1 mod 2^32

    这里由 trusted dealer 直接生成两方 share。
    """
    x = x.astype(np.uint32)

    s0 = rng.integers(
        low=0,
        high=2 ** 32,
        size=x.shape,
        dtype=np.uint32
    )

    s1 = (
        x.astype(np.uint64) - s0.astype(np.uint64)
    ) % MOD

    return s0.astype(np.uint32), s1.astype(np.uint32)


def _share_bit_value(x, rng):
    """
    对 bit 值做 Boolean sharing：

        x = s0 XOR s1

    这里 x 通常是 0/1 数组。
    """
    x = x.astype(np.uint8)

    s0 = rng.integers(
        low=0,
        high=2,
        size=x.shape,
        dtype=np.uint8
    )

    s1 = np.bitwise_xor(x, s0).astype(np.uint8)

    return s0, s1


def _append_to_pool(pool, shape, triples):
    """
    按 shape 把生成好的 triple 放进对应 pool。
    """
    key = tuple(shape)

    if key not in pool:
        pool[key] = []

    pool[key].extend(triples)


def _append_matmul_to_pool(pool, x_shape, w_shape, triples):
    """
    按矩阵乘法形状把生成好的矩阵 triple 放进 pool。

    key:
        (x_shape, w_shape)
    """
    key = (tuple(x_shape), tuple(w_shape))

    if key not in pool:
        pool[key] = []

    pool[key].extend(triples)


def _ring_matmul(a, b):
    """
    Z_(2^32) 环上的矩阵乘法。
    """
    return ((a.astype(np.uint64) @ b.astype(np.uint64)) % MOD).astype(np.uint32)


def _generate_arith_triples_for_party(shape, count, party_id, rng):
    """
    trusted dealer 方式生成 arithmetic Beaver triples。

    每个 triple 满足：
        c = a * b mod 2^32
    """
    triples = []

    for _ in range(count):
        a = rng.integers(
            low=0,
            high=2 ** 32,
            size=shape,
            dtype=np.uint32
        )

        b = rng.integers(
            low=0,
            high=2 ** 32,
            size=shape,
            dtype=np.uint32
        )

        c = (
            a.astype(np.uint64) * b.astype(np.uint64)
        ) % MOD
        c = c.astype(np.uint32)

        a0, a1 = _share_arith_value(a, rng)
        b0, b1 = _share_arith_value(b, rng)
        c0, c1 = _share_arith_value(c, rng)

        if party_id == 0:
            triples.append((a0, b0, c0))
        else:
            triples.append((a1, b1, c1))

    return triples


def _generate_bit_triples_for_party(shape, count, party_id, rng):
    """
    trusted dealer 方式生成 Boolean Beaver triples。

    每个 triple 满足：
        c = a AND b
    """
    triples = []

    for _ in range(count):
        a = rng.integers(
            low=0,
            high=2,
            size=shape,
            dtype=np.uint8
        )

        b = rng.integers(
            low=0,
            high=2,
            size=shape,
            dtype=np.uint8
        )

        c = np.bitwise_and(a, b).astype(np.uint8)

        a0, a1 = _share_bit_value(a, rng)
        b0, b1 = _share_bit_value(b, rng)
        c0, c1 = _share_bit_value(c, rng)

        if party_id == 0:
            triples.append((a0, b0, c0))
        else:
            triples.append((a1, b1, c1))

    return triples


def _generate_matmul_triples_for_party(x_shape, w_shape, count, party_id, rng):
    """
    trusted dealer 方式生成矩阵 Beaver triples。

    对于安全矩阵乘法：
        Z = X @ W

    需要生成：
        A.shape = X.shape
        B.shape = W.shape
        C = A @ B

    每个 party 最终拿到：
        A_i, B_i, C_i
    """
    x_shape = tuple(x_shape)
    w_shape = tuple(w_shape)

    if len(x_shape) != 2 or len(w_shape) != 2:
        raise ValueError(
            f"matmul triple only supports 2D matrices, "
            f"got x_shape={x_shape}, w_shape={w_shape}"
        )

    if x_shape[1] != w_shape[0]:
        raise ValueError(
            f"invalid matmul shapes: x_shape={x_shape}, w_shape={w_shape}"
        )

    triples = []

    for _ in range(count):
        a = rng.integers(
            low=0,
            high=2 ** 32,
            size=x_shape,
            dtype=np.uint32
        )

        b = rng.integers(
            low=0,
            high=2 ** 32,
            size=w_shape,
            dtype=np.uint32
        )

        c = _ring_matmul(a, b)

        a0, a1 = _share_arith_value(a, rng)
        b0, b1 = _share_arith_value(b, rng)
        c0, c1 = _share_arith_value(c, rng)

        if party_id == 0:
            triples.append((a0, b0, c0))
        else:
            triples.append((a1, b1, c1))

    return triples


def setup_triple_pool_by_dealer(
    party_id,
    arith_plan=None,
    bit_plan=None,
    matmul_plan=None,
    seed=202405
):
    """
    Offline trusted dealer triple generation.

    作用：
        为功能测试生成 Beaver triples，并填入现有 triple_pool。

    说明：
        Sonic 论文中 multiplication triple 是数据无关 offline material，
        在线推理时假设 triples 已经可用。

    arith_plan:
        [((shape...), count), ...]

    bit_plan:
        [((shape...), count), ...]

    matmul_plan:
        [((x_shape...), (w_shape...), count), ...]

        例如：
            [
                ((1, 2), (2, 2), 1)
            ]

        表示为一次矩阵乘法 X @ W 准备矩阵 triple：
            X.shape = (1, 2)
            W.shape = (2, 2)

    注意：
        这个文件实现的是“可信 dealer”版本的 offline triple 生成。
        适合功能测试和复现实验调试。
    """
    if arith_plan is None:
        arith_plan = []

    if bit_plan is None:
        bit_plan = []

    if matmul_plan is None:
        matmul_plan = []

    arith_pool = _get_arith_pool()
    bit_pool = _get_bit_pool()
    matmul_pool = _get_matmul_pool()

    arith_pool.clear()
    bit_pool.clear()
    matmul_pool.clear()

    rng = np.random.default_rng(seed)

    # 生成普通 arithmetic triples。
    for shape, count in arith_plan:
        shape = tuple(shape)

        triples = _generate_arith_triples_for_party(
            shape=shape,
            count=count,
            party_id=party_id,
            rng=rng
        )

        _append_to_pool(arith_pool, shape, triples)
        inc("offline_arith_triples", count)

    # 生成 Boolean triples。
    for shape, count in bit_plan:
        shape = tuple(shape)

        triples = _generate_bit_triples_for_party(
            shape=shape,
            count=count,
            party_id=party_id,
            rng=rng
        )

        _append_to_pool(bit_pool, shape, triples)
        inc("offline_bit_triples", count)

    # 生成矩阵 Beaver triples。
    for x_shape, w_shape, count in matmul_plan:
        x_shape = tuple(x_shape)
        w_shape = tuple(w_shape)

        triples = _generate_matmul_triples_for_party(
            x_shape=x_shape,
            w_shape=w_shape,
            count=count,
            party_id=party_id,
            rng=rng
        )

        _append_matmul_to_pool(matmul_pool, x_shape, w_shape, triples)
        inc("offline_matmul_triples", count)