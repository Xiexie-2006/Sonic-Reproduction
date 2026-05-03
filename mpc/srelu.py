from mpc.compare import secure_compare_zero
from mpc.b2a_secure import b2a_secure
from mpc.triple import get_arith_triple
from mpc.mul import secure_mul
from net.profiler import inc, time_block


def srelu(xi, conn, party_id):
    """
    Secure ReLU / SReLU:

        SReLU(x) = max(0, x)

    协议流程：

        1. secure_compare_zero(x)
           得到 gate:
                gate = 1, if x >= 0
                gate = 0, if x < 0

        2. B2A(gate)
           Boolean share -> Arithmetic share

        3. secure_mul(gate, x)
           输出 gate * x

    输入：
        xi:
            当前方持有的 x 的 arithmetic share

    输出：
        当前方持有的 SReLU(x) 的 arithmetic share

    SReLU 是 Sonic 复现中非线性激活的核心部分。
    普通 ReLU 在明文里只需要判断 x 是否小于 0，
    但在 MPC 中 x 不能公开，所以需要通过安全比较和安全乘法完成。
    """

    # 记录 ReLU/SReLU 调用次数。
    # 这一项在性能统计中比较重要，因为安全激活通常开销较大。
    inc("relu_calls")

    # 统计 SReLU 执行时间。
    with time_block("relu_time"):
        # 第一步：安全比较 x 是否非负。
        # 返回的是 Boolean share 形式的 gate。
        gate_bool_i = secure_compare_zero(
            xi=xi,
            conn=conn,
            party_id=party_id
        )

        # 第二步：把 Boolean share 的 gate 转成 arithmetic share。
        # 因为后面的乘法 secure_mul 是在 arithmetic share 上进行的。
        gate_arith_i = b2a_secure(
            xb_i=gate_bool_i,
            conn=conn,
            party_id=party_id
        )

        # 第三步：准备 Beaver triple，用于安全计算 gate * x。
        triple_i = get_arith_triple(
            conn=conn,
            party_id=party_id,
            shape=xi.shape
        )

        # gate=1 时输出 x，gate=0 时输出 0。
        # 这样就实现了 max(0, x)。
        zi = secure_mul(
            xi=gate_arith_i,
            yi=xi,
            triple_i=triple_i,
            conn=conn,
            party_id=party_id
        )

        return zi