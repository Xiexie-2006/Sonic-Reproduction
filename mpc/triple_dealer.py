import numpy as np

from mpc.share import MOD
import mpc.triple_pool as triple_pool
from net.profiler import inc


def _get_pool_dict(name_candidates):
    # 从 triple_pool 模块中查找可能存在的 pool 字典。
    # 这里写多个候选名字，是为了兼容前后版本中不同的变量命名。
    for name in name_candidates:
        if hasattr(triple_pool, name):
            obj = getattr(triple_pool, name)
            if isinstance(obj, dict):
                return obj

    # 如果没有找到对应的字典，说明 triple_pool.py 的结构和当前代码预期不一致。
    raise RuntimeError(
        "Cannot find triple pool dictionary in mpc.triple_pool.py. "
        "Please send me your mpc/triple_pool.py if this error appears."
    )


def _get_arith_pool():
    # 获取 arithmetic Beaver triple 的缓存池。
    return _get_pool_dict([
        "_ARITH_POOL",
        "ARITH_POOL",
        "_arith_pool",
        "arith_pool",
        "_arith_triple_pool",
        "arith_triple_pool"
    ])


def _get_bit_pool():
    # 获取 Boolean Beaver triple 的缓存池。
    return _get_pool_dict([
        "_BIT_POOL",
        "BIT_POOL",
        "_bit_pool",
        "bit_pool",
        "_bit_triple_pool",
        "bit_triple_pool"
    ])


def _share_arith_value(x, rng):
    # 对一个环元素 x 做 arithmetic sharing：
    # x = s0 + s1 mod 2^32

    # 这里由 dealer 直接生成两方的 share。
    x = x.astype(np.uint32)

    # s0 随机生成。
    s0 = rng.integers(
        low=0,
        high=2 ** 32,
        size=x.shape,
        dtype=np.uint32
    )

    # s1 = x - s0 mod MOD。
    s1 = (
        x.astype(np.uint64)
        - s0.astype(np.uint64)
    ) % MOD

    return s0.astype(np.uint32), s1.astype(np.uint32)


def _share_bit_value(x, rng):
    # 对 bit 值做 Boolean sharing：
    # x = s0 XOR s1

    # 这里 x 通常是 0/1 数组
    x = x.astype(np.uint8)

    # s0 随机生成 0/1。
    s0 = rng.integers(
        low=0,
        high=2,
        size=x.shape,
        dtype=np.uint8
    )

    # s1 = x XOR s0。
    s1 = np.bitwise_xor(x, s0).astype(np.uint8)

    return s0, s1


def _append_to_pool(pool, shape, triples):
    # 按 shape 把生成好的 triple 放进对应 pool。
    # 不同 shape 的乘法需要不同形状的 triple，所以这里用 shape 作为 key。
    key = tuple(shape)

    if key not in pool:
        pool[key] = []

    pool[key].extend(triples)


def setup_triple_pool_by_dealer(party_id, arith_plan=None, bit_plan=None, seed=202405):
    """
    Offline trusted dealer triple generation.

    作用：
        为大模型功能测试生成 Beaver triples，并填入现有 triple_pool。

    说明：
        Sonic 论文中 multiplication triple 是数据无关 offline material，
        在线推理时假设 triples 已经可用。

    arith_plan:
        [((shape...), count), ...]

    bit_plan:
        [((shape...), count), ...]

    输出：
        直接填充 mpc.triple_pool 中的全局 pool。

    这个文件实现的是“可信 dealer”版本的 offline triple 生成。
    它更适合功能测试和复现实验调试，因为不需要每次都通过 OT 生成 triple。
    """

    # 如果没有传入计划，就默认不生成对应类型的 triple。
    if arith_plan is None:
        arith_plan = []

    if bit_plan is None:
        bit_plan = []

    # 取出当前 triple_pool 中的全局缓存池。
    arith_pool = _get_arith_pool()
    bit_pool = _get_bit_pool()

    # 每次 setup 前先清空旧 pool，避免不同测试之间的 triple 混用。
    arith_pool.clear()
    bit_pool.clear()

    # 固定随机种子，保证测试可复现。
    rng = np.random.default_rng(seed)

    # 生成 arithmetic Beaver triples。
    for shape, count in arith_plan:
        shape = tuple(shape)
        triples = []

        for _ in range(count):
            # 随机生成完整的 a 和 b。
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

            # 计算 c = a * b mod 2^32。
            c = (
                a.astype(np.uint64)
                * b.astype(np.uint64)
            ) % MOD
            c = c.astype(np.uint32)

            # 将 a、b、c 分别拆成两方 arithmetic share。
            a0, a1 = _share_arith_value(a, rng)
            b0, b1 = _share_arith_value(b, rng)
            c0, c1 = _share_arith_value(c, rng)

            # party0 拿第 0 份，party1 拿第 1 份。
            if party_id == 0:
                triples.append((a0, b0, c0))
            else:
                triples.append((a1, b1, c1))

        # 放入 arithmetic pool。
        _append_to_pool(arith_pool, shape, triples)

        # 记录 offline arithmetic triple 数量。
        inc("offline_arith_triples", count)

    # 生成 Boolean Beaver triples。
    for shape, count in bit_plan:
        shape = tuple(shape)
        triples = []

        for _ in range(count):
            # 随机生成完整的 bit a 和 b。
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

            # Boolean triple 中 c = a AND b。
            c = np.bitwise_and(a, b).astype(np.uint8)

            # 对 a、b、c 分别做 Boolean sharing。
            a0, a1 = _share_bit_value(a, rng)
            b0, b1 = _share_bit_value(b, rng)
            c0, c1 = _share_bit_value(c, rng)

            if party_id == 0:
                triples.append((a0, b0, c0))
            else:
                triples.append((a1, b1, c1))

        # 放入 Boolean pool。
        _append_to_pool(bit_pool, shape, triples)

        # 记录 offline Boolean triple 数量。
        inc("offline_bit_triples", count)