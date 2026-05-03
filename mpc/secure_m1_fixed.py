from mpc.linear_secret import linear_secret_weight
from mpc.trunc import secure_trunc
from mpc.sbn import sbn_fixed
from mpc.srelu import srelu


def flatten_nchw(x_i):
    """
    [N, C, H, W] -> [N, C*H*W]

    M1 是全连接网络，因此输入图像需要先 flatten 成二维特征。
    reshape 本身只是本地重排，不会影响 secret share 的正确性。
    """
    n = x_i.shape[0]
    return x_i.reshape(n, -1)


def secure_m1_fixed_inference(
    x_i,
    fc1_w_i,
    fc1_b_i,
    bn1_eps1_i,
    bn1_eps2_i,
    fc2_w_i,
    fc2_b_i,
    bn2_eps1_i,
    bn2_eps2_i,
    fc3_w_i,
    fc3_b_i,
    bn3_eps1_i,
    bn3_eps2_i,
    scale_bits,
    conn,
    party_id
):
    """
    Sonic paper M1 fixed-point MPC inference.

    M1 structure:

        Flatten
        → SFC1
        → SBN1
        → SReLU1
        → SFC2
        → SBN2
        → SReLU2
        → SFC3
        → SBN3

    注意：
        FC 层乘法后 scale = 2^(2f)
        因此每个 FC 后需要 secure_trunc 恢复到 2^f

        SBN 内部也会：
            eps1 * x + eps2
            然后 secure_trunc 恢复到 2^f

    M1 是 Sonic 论文中的 MNIST 全连接模型。
    这个文件主要验证 SFC、SBN、SReLU 这些基础模块在多层网络中能否正确组合。
    """

    # 输入图像先展平。
    # 例如 MNIST 的 [N, 1, 28, 28] 会变成 [N, 784]。
    x_flat_i = flatten_nchw(x_i)

    # FC1
    # 第一层安全全连接，输出维度通常是 128。
    # 因为 X 和 W 都是 fixed-point，乘法结果的尺度会变成 2^(2f)。
    h1_raw_i = linear_secret_weight(
        x_i=x_flat_i,
        w_i=fc1_w_i,
        b_i=fc1_b_i,
        conn=conn,
        party_id=party_id
    )

    # FC1 后截断，把尺度恢复到 2^f。
    h1_i = secure_trunc(
        x_i=h1_raw_i,
        shift_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # SBN1
    # 对第一层全连接输出做安全 BN。
    h1_bn_i = sbn_fixed(
        x_i=h1_i,
        eps1_i=bn1_eps1_i,
        eps2_i=bn1_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # SReLU1
    # BN 后接安全激活函数。
    h1_act_i = srelu(
        xi=h1_bn_i,
        conn=conn,
        party_id=party_id
    )

    # FC2
    # 第二层安全全连接，继续在 arithmetic share 上计算。
    h2_raw_i = linear_secret_weight(
        x_i=h1_act_i,
        w_i=fc2_w_i,
        b_i=fc2_b_i,
        conn=conn,
        party_id=party_id
    )

    # FC2 后截断。
    h2_i = secure_trunc(
        x_i=h2_raw_i,
        shift_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # SBN2
    h2_bn_i = sbn_fixed(
        x_i=h2_i,
        eps1_i=bn2_eps1_i,
        eps2_i=bn2_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # SReLU2
    h2_act_i = srelu(
        xi=h2_bn_i,
        conn=conn,
        party_id=party_id
    )

    # FC3
    # 最后一层全连接输出分类 logits。
    out_raw_i = linear_secret_weight(
        x_i=h2_act_i,
        w_i=fc3_w_i,
        b_i=fc3_b_i,
        conn=conn,
        party_id=party_id
    )

    # 最后一层 FC 后同样需要 secure_trunc。
    out_i = secure_trunc(
        x_i=out_raw_i,
        shift_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    # SBN3
    # M1 结构中最后一层 FC 后还接 BN，最终输出 logits share。
    out_bn_i = sbn_fixed(
        x_i=out_i,
        eps1_i=bn3_eps1_i,
        eps2_i=bn3_eps2_i,
        scale_bits=scale_bits,
        conn=conn,
        party_id=party_id
    )

    return out_bn_i