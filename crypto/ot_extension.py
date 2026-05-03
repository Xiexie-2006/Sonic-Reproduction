import hashlib
import secrets
import numpy as np

from crypto.ot import ot_send, ot_recv
from net.socket_utils import send_data, recv_data


# OT Extension 的安全参数。
# 这里表示先做 128 次基础 OT，然后用 PRG 扩展出更多 OT。
KAPPA = 128


def _random_seed():
    """
    生成一个随机 seed。

    在 OT Extension 中，双方先通过基础 OT 交换少量 seed，
    后面再用这些 seed 扩展出长随机 bit 串。
    """
    return secrets.token_bytes(16)


def _prg_bits(seed, n_bits):
    """
    PRG: seed -> n_bits bits

    用 SHA-256(seed || counter) 模拟伪随机生成器。
    输入一个短 seed，输出 n_bits 个伪随机 bit。
    """
    n_bytes = (n_bits + 7) // 8
    out = b""
    counter = 0

    # 一次 SHA-256 只能输出 32 字节。
    # 如果需要更多随机字节，就不断增加 counter 继续哈希。
    while len(out) < n_bytes:
        h = hashlib.sha256()
        h.update(seed)
        h.update(counter.to_bytes(4, "big"))
        out += h.digest()
        counter += 1

    # 把字节拆成 bit，并截取需要的长度。
    arr = np.frombuffer(out[:n_bytes], dtype=np.uint8)
    bits = np.unpackbits(arr)[:n_bits].astype(np.uint8)

    return bits


def _hash_bit(bits):
    """
    将一个 bit 向量压缩成 1 个 bit mask。

    在 OT Extension 中，每个扩展 OT 对应矩阵的一列。
    这里把这一列哈希成一个 bit，用来加密 0/1 消息。
    """
    bits = np.asarray(bits, dtype=np.uint8)
    packed = np.packbits(bits).tobytes()

    h = hashlib.sha256()
    h.update(packed)

    # 只取最低 1 bit 作为掩码。
    return h.digest()[0] & 1


def ot_ext_send_bits(conn, m0_arr, m1_arr, kappa=KAPPA):
    """
    OT Extension 的发送方。

    Sender 持有：
        m0_arr, m1_arr

    Receiver 持有：
        choice_arr

    Receiver 最终只能得到：
        choice == 0 时得到 m0
        choice == 1 时得到 m1
    """

    m0_arr = np.asarray(m0_arr, dtype=np.uint32)
    m1_arr = np.asarray(m1_arr, dtype=np.uint32)

    # 两组消息必须形状一致，才能逐元素做 OT。
    assert m0_arr.shape == m1_arr.shape

    shape = m0_arr.shape
    m = m0_arr.size

    # 展平成一维，方便按列批量处理。
    m0_flat = m0_arr.reshape(-1)
    m1_flat = m1_arr.reshape(-1)

    # 在 IKNP 风格 OT Extension 中，
    # 扩展 OT 的 Sender 在 base OT 阶段反过来充当 Receiver。
    # s_bits 是 Sender 自己的随机选择向量。
    s_bits = np.random.randint(0, 2, size=kappa, dtype=np.uint8)

    selected_seeds = []

    # 做 kappa 次基础 OT。
    # Sender 根据 s_bits[i] 选择 seed0 或 seed1。
    for i in range(kappa):
        seed_i = ot_recv(conn, int(s_bits[i]))
        selected_seeds.append(seed_i)

    # 接收 Receiver 发来的 U 矩阵。
    # U 的形状是 kappa × m。
    U = recv_data(conn)
    U = np.asarray(U, dtype=np.uint8)

    # Q 矩阵由 Sender 根据 selected_seeds 和 U 构造。
    Q = np.zeros((kappa, m), dtype=np.uint8)

    for i in range(kappa):
        # 用 base OT 得到的 seed 扩展出一行随机 bit。
        q_i = _prg_bits(selected_seeds[i], m)

        # 如果 s_i = 1，需要和 U_i 异或。
        if s_bits[i] == 1:
            q_i = q_i ^ U[i]

        Q[i] = q_i

    # c0、c1 是两组加密后的 bit 消息。
    c0 = np.zeros(m, dtype=np.uint32)
    c1 = np.zeros(m, dtype=np.uint32)

    for j in range(m):
        # 第 j 列对应第 j 个扩展 OT。
        q_col = Q[:, j]

        # 对 m0 和 m1 分别生成不同的 mask。
        key0 = _hash_bit(q_col)
        key1 = _hash_bit(q_col ^ s_bits)

        # bit 消息加密：cipher = message xor key。
        c0[j] = int(m0_flat[j]) ^ key0
        c1[j] = int(m1_flat[j]) ^ key1

    # 发送两组密文。
    # Receiver 只能根据自己的 choice 解出其中一组。
    send_data(conn, (c0.reshape(shape), c1.reshape(shape)))


def ot_ext_recv_bits(conn, choice_arr, kappa=KAPPA):
    """
    OT Extension 的接收方。

    Receiver 输入：
        choice_arr: 选择 bit 数组，元素为 0/1

    返回：
        out_arr，其中 out[j] = m_choice[j]
    """

    choice_arr = np.asarray(choice_arr, dtype=np.uint32)

    shape = choice_arr.shape
    choices = choice_arr.reshape(-1).astype(np.uint8)
    m = choices.size

    seed0_list = []
    seed1_list = []

    # Receiver 在 base OT 阶段反过来充当 Sender。
    # 每轮发送一对 seed，让另一方根据自己的 s_i 选择其中一个。
    for _ in range(kappa):
        seed0 = _random_seed()
        seed1 = _random_seed()

        seed0_list.append(seed0)
        seed1_list.append(seed1)

        ot_send(conn, seed0, seed1)

    # T 是 Receiver 自己保存的矩阵。
    # U 会发送给 Sender，用于帮助 Sender 构造 Q。
    T = np.zeros((kappa, m), dtype=np.uint8)
    U = np.zeros((kappa, m), dtype=np.uint8)

    for i in range(kappa):
        # 分别用 seed0 和 seed1 扩展出两行随机 bit。
        g0 = _prg_bits(seed0_list[i], m)
        g1 = _prg_bits(seed1_list[i], m)

        T[i] = g0

        # U_i = G(seed0_i) xor G(seed1_i) xor choices
        # 这里把 Receiver 的选择向量隐藏进 U 中。
        U[i] = g0 ^ g1 ^ choices

    # 把 U 矩阵发给 Sender。
    send_data(conn, U)

    # 接收 Sender 发来的两组密文。
    c0, c1 = recv_data(conn)

    c0 = np.asarray(c0, dtype=np.uint32).reshape(-1)
    c1 = np.asarray(c1, dtype=np.uint32).reshape(-1)

    out = np.zeros(m, dtype=np.uint32)

    for j in range(m):
        # Receiver 用 T 的第 j 列生成解密 mask。
        t_col = T[:, j]
        key = _hash_bit(t_col)

        # 根据 choice 解密对应的那一条消息。
        if choices[j] == 0:
            out[j] = int(c0[j]) ^ key
        else:
            out[j] = int(c1[j]) ^ key

    return out.reshape(shape).astype(np.uint32)