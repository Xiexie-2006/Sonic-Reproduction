import numpy as np
from mpc.share import MOD
from mpc.triple import get_arith_triple
from mpc.mul import secure_mul
from net.profiler import inc, time_block


def linear_secret_weight(x_i, w_i, b_i, conn, party_id):
    """
    秘密权重安全线性层：

        Y = XW + b

    X、W、b 都是 arithmetic share。

    这个函数对应 MPC 版本中的安全全连接层。
    和公开权重线性层不同，这里输入 X、权重 W、偏置 b 都处于秘密分享状态，
    因此矩阵乘法中的每一项乘法都需要通过 secure_mul 来完成。
    """
    # 记录安全线性层调用次数，方便统计整体推理中线性层的协议开销。
    inc("linear_secret_calls")

    # 统计安全线性层运行时间。
    with time_block("linear_secret_time"):
        # batch 表示样本数量，in_dim 表示输入维度。
        batch, in_dim = x_i.shape

        # w_i 的形状为 [in_dim, out_dim]。
        # out_dim 表示当前线性层输出维度。
        _, out_dim = w_i.shape

        # 初始化输出 share。
        # 最终 y_i 的形状为 [batch, out_dim]。
        y_i = np.zeros((batch, out_dim), dtype=np.uint32)

        # 按照矩阵乘法公式逐项累加：
        #   Y[:, j] = sum_k X[:, k] * W[k, j]
        #
        # 因为 X 和 W 都是秘密分享，所以每一项乘法都要调用安全乘法协议。
        for k in range(in_dim):
            # 取出输入矩阵第 k 列，形状为 [batch, 1]。
            x_part = x_i[:, k:k+1]

            # 取出权重矩阵第 k 行，形状为 [1, out_dim]。
            w_part = w_i[k:k+1, :]

            # 为了让 x_part 和 w_part 能按元素做安全乘法，
            # 这里把 x_part 扩展成 [batch, out_dim]。
            x_term = np.repeat(x_part, out_dim, axis=1)

            # 同理，把 w_part 扩展成 [batch, out_dim]。
            w_term = np.repeat(w_part, batch, axis=0)

            # 为当前这一批乘法生成 Beaver triple。
            triple_i = get_arith_triple(conn, party_id, x_term.shape)

            # 安全计算 X[:, k] * W[k, :]。
            prod_i = secure_mul(
                xi=x_term,
                yi=w_term,
                triple_i=triple_i,
                conn=conn,
                party_id=party_id
            )

            # 将当前 k 对应的乘积结果累加到输出中。
            # 所有加法都在 Z_(2^32) 环上完成。
            y_i = (y_i.astype(np.uint64) + prod_i.astype(np.uint64)) % MOD
            y_i = y_i.astype(np.uint32)

        # 加上 bias。
        # 这里 b_i 本身也是 arithmetic share，因此两方都加自己的那一份即可。
        if b_i is not None:
            y_i = (y_i.astype(np.uint64) + b_i.reshape(1, -1).astype(np.uint64)) % MOD
            y_i = y_i.astype(np.uint32)

        return y_i