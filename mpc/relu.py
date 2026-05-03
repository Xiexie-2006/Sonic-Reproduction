from mpc.compare import secure_compare_zero
from mpc.b2a_secure import b2a_secure
from mpc.triple import get_arith_triple
from mpc.mul import secure_mul
from net.profiler import inc, time_block


def relu(xi, conn, party_id):
    """
    ReLU(x) = gate * x

    gate = 1 if x >= 0 else 0

    在明文神经网络中，ReLU 只是 max(x, 0)。
    但在 MPC 中，x 是秘密分享状态，不能直接判断正负，
    所以需要先通过安全比较得到 gate，再用 gate 乘以 x。
    """
    # 记录 ReLU 调用次数。
    inc("relu_calls")

    # 统计安全 ReLU 的执行时间。
    with time_block("relu_time"):
        # 安全判断 xi 是否大于等于 0。
        # 返回结果是 Boolean share：
        #   gate = 1 表示 x >= 0
        #   gate = 0 表示 x < 0
        gate_bool_i = secure_compare_zero(xi, conn, party_id)

        # 将 Boolean share 转成 Arithmetic share。
        # 因为后面需要计算 gate * x，乘法协议使用的是 arithmetic share。
        gate_arith_i = b2a_secure(gate_bool_i, conn, party_id)

        # 为 gate * x 生成 Beaver triple。
        triple_i = get_arith_triple(conn, party_id, xi.shape)

        # 计算 ReLU(x) = gate * x。
        # 如果 x 非负，gate=1，输出 x；
        # 如果 x 为负，gate=0，输出 0。
        zi = secure_mul(gate_arith_i, xi, triple_i, conn, party_id)

        return zi