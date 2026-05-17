from mpc.compare_ppa import secure_compare_zero_ppa
from mpc.b2a_secure import b2a_secure
from mpc.triple import get_arith_triple
from mpc.mul import secure_mul
from net.profiler import inc, time_block


def srelu_ppa(xi, conn, party_id):
    """
    PPA 优化版 Secure ReLU。

    Sonic 中：
        ReLU(x) = NOT(MSB(x)) * x

    原版 SReLU 的 MSB 是逐位 carry。
    这里改为 PPA 版 secure MSB。

    流程：
        1. secure_compare_zero_ppa 得到 gate 的 Boolean share；
        2. B2A 将 Boolean gate 转成 Arithmetic share；
        3. secure_mul 计算 gate * x；
        4. 输出 ReLU(x) 的 arithmetic share。
    """
    inc("relu_calls")

    with time_block("relu_time"):
        gate_bool_i = secure_compare_zero_ppa(
            xi=xi,
            conn=conn,
            party_id=party_id,
        )

        gate_arith_i = b2a_secure(
            xb_i=gate_bool_i,
            conn=conn,
            party_id=party_id,
        )

        triple_i = get_arith_triple(
            conn=conn,
            party_id=party_id,
            shape=xi.shape,
        )

        zi = secure_mul(
            xi=gate_arith_i,
            yi=xi,
            triple_i=triple_i,
            conn=conn,
            party_id=party_id,
        )

        return zi