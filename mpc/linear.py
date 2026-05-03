import numpy as np
from mpc.share import MOD


def linear_public_weight(xi, W, b=None, party_id=0):
    """
    安全线性层：Y = XW + b

    xi:
        当前方持有的 arithmetic share
        shape = [batch, in_dim]

    W:
        公开权重
        shape = [in_dim, out_dim]

    b:
        公开偏置
        shape = [out_dim]

    说明：
        当前版本采用“公开模型参数 + 秘密输入”的推理设定。
        权重 W 公开，所以每方可以本地计算 xi @ W。
        偏置 b 只由 party 0 加一次，避免重复加偏置。

    这个函数对应安全推理中的公开权重线性层。
    因为模型参数不是秘密，所以这里不需要安全乘法协议；
    只需要对输入 share 分别乘公开权重，最后两方结果相加即可恢复完整输出。
    """

    # 保持输入和权重都在 uint32 环表示下计算。
    xi = xi.astype(np.uint32)
    W = W.astype(np.uint32)

    batch, in_dim = xi.shape
    out_dim = W.shape[1]

    # 使用 uint64 做中间累加，避免 uint32 乘法和加法提前溢出。
    yi = np.zeros((batch, out_dim), dtype=np.uint64)

    # 逐项累加，避免多维矩阵乘法中 uint64 溢出。
    # 这里本质上是在计算 xi @ W。
    for k in range(in_dim):
        # left: [batch, 1]
        # right: [1, out_dim]
        # 广播相乘后得到 [batch, out_dim]。
        left = xi[:, k:k+1].astype(np.uint64)
        right = W[k:k+1, :].astype(np.uint64)
        yi = (yi + left * right) % MOD

    yi = yi.astype(np.uint32)

    # 偏置是公开值，如果两方都加一次就会重复。
    # 因此约定只由 party0 加 bias，party1 不加。
    if b is not None and party_id == 0:
        yi = (yi.astype(np.uint64) + b.astype(np.uint64)) % MOD
        yi = yi.astype(np.uint32)

    return yi