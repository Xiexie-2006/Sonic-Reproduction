from net.socket_utils import send_data, recv_data
from net.profiler import inc

from mpc.bit_triple_ot import generate_bit_triples_ot
from mpc.arith_triple_ot import generate_arith_triples_ot


# arithmetic triple 缓存池。
# key 是 shape，value 是对应 shape 的 triple 列表。
_ARITH_POOL = {}

# Boolean triple 缓存池。
_BIT_POOL = {}


def _shape_key(shape):
    # 统一把 shape 转成 tuple，作为字典 key。
    return tuple(shape)


def clear_triple_pools():
    # 清空本地 triple pool。
    # 每次新测试或新推理前清空，可以避免使用到上一轮剩余的 triple。
    global _ARITH_POOL, _BIT_POOL
    _ARITH_POOL = {}
    _BIT_POOL = {}


def _push_arith(shape, triples):
    # 将一批 arithmetic triples 放入 pool。
    key = _shape_key(shape)

    if key not in _ARITH_POOL:
        _ARITH_POOL[key] = []

    _ARITH_POOL[key].extend(triples)


def _push_bit(shape, triples):
    # 将一批 Boolean triples 放入 pool。
    key = _shape_key(shape)

    if key not in _BIT_POOL:
        _BIT_POOL[key] = []

    _BIT_POOL[key].extend(triples)


def pop_arith_triple(shape):
    # 从 arithmetic pool 中取出一个指定 shape 的 triple。
    # secure_mul 会通过这个接口拿预处理好的 Beaver triple。
    key = _shape_key(shape)

    if key not in _ARITH_POOL or len(_ARITH_POOL[key]) == 0:
        raise RuntimeError(f"Arithmetic triple pool exhausted for shape={key}")

    return _ARITH_POOL[key].pop()


def pop_bit_triple(shape):
    # 从 Boolean pool 中取出一个指定 shape 的 triple。
    # bit_and 会通过这个接口拿预处理好的 Boolean triple。
    key = _shape_key(shape)

    if key not in _BIT_POOL or len(_BIT_POOL[key]) == 0:
        raise RuntimeError(f"Boolean triple pool exhausted for shape={key}")

    return _BIT_POOL[key].pop()


def _setup_arith_pool_by_ot_extension(conn, party_id, arith_plan):
    """
    使用 OT Extension 批量生成 Arithmetic Beaver triples。

    生成后直接放入本地 offline pool。

    arith_plan 中每一项表示：
        shape: triple 的张量形状
        count: 需要生成的数量
    """

    for shape, count in arith_plan:
        # 通过 OT Extension 生成当前 shape 的 arithmetic triples。
        triples = generate_arith_triples_ot(
            conn=conn,
            party_id=party_id,
            shape=shape,
            count=count
        )

        # 放入本地 pool，供 online 阶段直接取用。
        _push_arith(shape, triples)

        # 记录 offline arithmetic triple 数量。
        inc("offline_arith_triples", count)


def _setup_bit_pool_by_ot_extension(conn, party_id, bit_plan):
    """
    使用 OT Extension 批量生成 Boolean Beaver triples。

    生成后直接放入本地 offline pool。

    bit_plan 的格式和 arith_plan 类似，
    只是生成的是用于 bit_and 的 Boolean triple。
    """

    for shape, count in bit_plan:
        # 通过 OT Extension 生成 Boolean triples。
        triples = generate_bit_triples_ot(
            conn=conn,
            party_id=party_id,
            shape=shape,
            count=count
        )

        # 放入 Boolean pool。
        _push_bit(shape, triples)

        # 记录 offline Boolean triple 数量。
        inc("offline_bit_triples", count)


def setup_triple_pool(conn, party_id, arith_plan, bit_plan):
    """
    Offline / Online 分离版 triple pool。

    当前版本：
        Arithmetic triple：OT Extension 批量生成
        Boolean triple   ：OT Extension 批量生成

    Online 阶段：
        secure_mul / bit_and 只从 pool 取 triple。

    这对应 Sonic 这类协议里的典型设计：
    乘法相关的随机材料可以在 offline 阶段提前准备，
    online 推理阶段只负责消耗这些材料，从而把在线计算流程简化。
    """

    # 先清空旧的 pool，保证当前推理使用的是本轮生成的 triple。
    clear_triple_pools()

    # Phase 1：Arithmetic triples by OT Extension。
    # 主要服务于 secure_mul、线性层、卷积、SBN 等算术乘法。
    _setup_arith_pool_by_ot_extension(
        conn=conn,
        party_id=party_id,
        arith_plan=arith_plan
    )

    # Phase 2：Boolean triples by OT Extension。
    # 主要服务于 bit_and、A2B、MSB、比较等 bit 级协议。
    _setup_bit_pool_by_ot_extension(
        conn=conn,
        party_id=party_id,
        bit_plan=bit_plan
    )

    # Phase 3：同步结束标志。
    # party0 发送 OFFLINE_DONE，party1 接收确认，
    # 这样可以保证双方都完成 offline 阶段后再进入 online 推理。
    if party_id == 0:
        send_data(conn, ("OFFLINE_DONE", None, None))
    else:
        tag, _, _ = recv_data(conn)

        if tag != "OFFLINE_DONE":
            raise RuntimeError(f"Unexpected offline finish tag: {tag}")