import hashlib
import secrets
import numpy as np

from crypto.ot import ot_send, ot_recv
from net.socket_utils import send_data, recv_data


# OT Extension 的安全参数。
# 这里表示先执行 128 次基础 OT，再通过 PRG 扩展出大量 OT。
KAPPA = 128


def _random_seed():
    """
    生成一个随机 seed。

    在 OT Extension 中，base OT 阶段只交换少量 seed，
    后续通过 PRG 把 seed 扩展成长随机 bit 串。
    """
    return secrets.token_bytes(16)


def _prg_bits(seed, n_bits):
    """
    PRG: seed -> n_bits bits

    用 SHA-256(seed || counter) 实现一个简单的伪随机生成器。
    输入一个短 seed，输出 n_bits 个伪随机 bit。
    """
    n_bytes = (n_bits + 7) // 8
    out = b""
    counter = 0

    # SHA-256 每次输出 32 字节。
    # 如果需要更多随机字节，就不断增加 counter 继续哈希。
    while len(out) < n_bytes:
        h = hashlib.sha256()
        h.update(seed)
        h.update(counter.to_bytes(4, "big"))
        out += h.digest()
        counter += 1

    # 将字节流拆成 bit 数组，并截取需要的长度。
    arr = np.frombuffer(out[:n_bytes], dtype=np.uint8)
    bits = np.unpackbits(arr)[:n_bits].astype(np.uint8)

    return bits


def _hash_uint32(bits):
    """
    将一个 bit 向量哈希成 uint32 掩码。

    ot_extension.py 中的 _hash_bit 只生成 1 bit mask，
    因为它处理的是 bit 消息。

    这里处理的是 uint32 消息，所以需要生成 32 bit 掩码。
    """
    bits = np.asarray(bits, dtype=np.uint8)
    packed = np.packbits(bits).tobytes()

    h = hashlib.sha256()
    h.update(packed)

    # 取 SHA-256 输出的前 4 字节，解释成 little-endian uint32。
    return np.uint32(int.from_bytes(h.digest()[:4], "little"))


def ot_ext_send_uint32(conn, m0_arr, m1_arr, kappa=KAPPA):
    """
    uint32 版本 OT Extension 的发送方。

    Sender 持有：
        m0_arr, m1_arr

    Receiver 持有：
        choice_arr

    Receiver 最终只能得到：
        choice == 0 -> m0
        choice == 1 -> m1

    这里的消息是 uint32，而不是单个 bit。
    """
    m0_arr = np.asarray(m0_arr, dtype=np.uint32)
    m1_arr = np.asarray(m1_arr, dtype=np.uint32)

    # 两组消息必须形状一致，才能逐元素执行 OT。
    assert m0_arr.shape == m1_arr.shape

    shape = m0_arr.shape
    m = m0_arr.size

    # 展平成一维，方便按列批量处理。
    m0_flat = m0_arr.reshape(-1)
    m1_flat = m1_arr.reshape(-1)

    # 扩展 OT 的 Sender 在 base OT 阶段反过来作为 Receiver。
    # s_bits 是 Sender 自己的随机选择向量。
    s_bits = np.random.randint(0, 2, size=kappa, dtype=np.uint8)

    selected_seeds = []

    # 做 kappa 次基础 OT。
    # Sender 根据 s_bits[i] 选择 seed0 或 seed1。
    for i in range(kappa):
        seed_i = ot_recv(conn, int(s_bits[i]))
        selected_seeds.append(seed_i)

    # 接收 Receiver 发送的 U 矩阵。
    # U 的形状是 kappa × m。
    U = recv_data(conn)
    U = np.asarray(U, dtype=np.uint8)

    # Q 矩阵由 Sender 根据 selected_seeds 和 U 构造。
    Q = np.zeros((kappa, m), dtype=np.uint8)

    for i in range(kappa):
        # 用 seed 扩展出一行随机 bit。
        q_i = _prg_bits(selected_seeds[i], m)

        # 如果 s_i = 1，需要与 U_i 异或。
        if s_bits[i] == 1:
            q_i = q_i ^ U[i]

        Q[i] = q_i

    # c0、c1 是两组加密后的 uint32 消息。
    c0 = np.zeros(m, dtype=np.uint32)
    c1 = np.zeros(m, dtype=np.uint32)

    for j in range(m):
        # 第 j 列对应第 j 个扩展 OT。
        q_col = Q[:, j]

        # 分别生成 uint32 掩码，用来隐藏 m0[j] 和 m1[j]。
        key0 = _hash_uint32(q_col)
        key1 = _hash_uint32(q_col ^ s_bits)

        # uint32 消息加密：cipher = message xor mask。
        c0[j] = np.uint32(int(m0_flat[j]) ^ int(key0))
        c1[j] = np.uint32(int(m1_flat[j]) ^ int(key1))

    # 发送两组密文。
    send_data(conn, (c0.reshape(shape), c1.reshape(shape)))


def ot_ext_recv_uint32(conn, choice_arr, kappa=KAPPA):
    """
    uint32 版本 OT Extension 的接收方。

    Receiver 输入：
        choice_arr: 选择 bit 数组，元素为 0/1

    返回：
        与 choice_arr 同 shape 的 uint32 数组。
        每个位置根据 choice 得到 m0 或 m1。
    """
    choice_arr = np.asarray(choice_arr, dtype=np.uint32)

    shape = choice_arr.shape
    choices = choice_arr.reshape(-1).astype(np.uint8)
    m = choices.size

    seed0_list = []
    seed1_list = []

    # Receiver 在 base OT 阶段反过来作为 Sender。
    # 每轮提供两个 seed，另一方根据自己的 s_i 选择其中一个。
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
        key = _hash_uint32(t_col)

        # 根据 choice 解密对应消息。
        if choices[j] == 0:
            out[j] = np.uint32(int(c0[j]) ^ int(key))
        else:
            out[j] = np.uint32(int(c1[j]) ^ int(key))

    return out.reshape(shape).astype(np.uint32)