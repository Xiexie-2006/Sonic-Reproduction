from mpc.triple_pool import pop_arith_triple, pop_bit_triple


def get_arith_triple(conn, party_id, shape):
    # 获取 arithmetic Beaver triple。

    # 这里保留 conn 和 party_id 参数，是为了和前面直接在线生成 triple 的接口保持一致。
    # 当前实现中，真正的 triple 来源是 triple_pool，
    # 也就是预先生成或缓存好的三元组池。

    # arithmetic triple 一般满足：
    # c = a * b mod 2^32
    # 主要用于 secure_mul、SBN、线性层、卷积等算术乘法场景。
    return pop_arith_triple(shape)


def get_bit_triple(conn, party_id, shape):
    # 获取 Boolean Beaver triple。

    # Boolean triple 一般满足：
    # c = a AND b
    # 其中 a、b、c 都是 Boolean share。

    # 它主要用于 bit_and，
    # 进一步支撑 A2B、MSB、secure_compare_zero 等 bit 级协议。
    return pop_bit_triple(shape)