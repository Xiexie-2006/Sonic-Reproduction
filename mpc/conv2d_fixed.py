from mpc.conv2d_secret import conv2d_secret
from mpc.trunc import secure_trunc


def conv2d_secret_fixed(
    x_i,
    w_i,
    b_i,
    scale_bits,
    conn,
    party_id,
    stride=1,
    padding=0
):
    """
    Fixed-point Secret Conv2D:

        Y_raw = Conv2D(X, W) + b
        Y     = Trunc(Y_raw, scale_bits)

    说明：
        X 的 scale = 2^f
        W 的 scale = 2^f
        Conv 输出 scale = 2^(2f)
        所以需要 secure_trunc 恢复到 2^f。

    这个函数是在 secret Conv2D 外面再包一层 fixed-point 处理。
    因为卷积本质上是乘加运算，输入和权重都是定点数时，
    相乘后小数缩放倍数会扩大一倍，所以卷积结束后必须做一次安全截断。
    """

    # 先执行秘密分享状态下的卷积。
    # 这里得到的是未截断的结果，scale 仍然是 2^(2f)。
    y_raw_i = conv2d_secret(
        x_i=x_i,
        w_i=w_i,
        b_i=b_i,
        conn=conn,
        party_id=party_id,
        stride=stride,
        padding=padding
    )

    # 对卷积结果做 secure_trunc。
    # 这一步相当于除以 2^scale_bits，
    # 把结果从 2^(2f) 的尺度恢复到 2^f。
    y_i = secure_trunc(
        x_i=y_raw_i,
        shift_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    return y_i