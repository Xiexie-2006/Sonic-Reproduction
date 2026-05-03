import pickle
import hashlib
import secrets
import numpy as np

from net.socket_utils import send_data, recv_data


# 一个可运行的 ElGamal-style 1-out-of-2 OT 群参数。
# 这里使用的是一个 127-bit 的 Mersenne prime：
#     P = 2^127 - 1

# 注意：
# 1. 这个实现的目标是复现基础 OT 的功能流程；
# 2. 它是真实的公钥 OT 思路，不是“把两条消息都发过去”的模拟版本；
# 3. 后续如果追求论文级性能，一般会用少量 Base OT 再扩展成大量 OT，
#    也就是 OT Extension。
P = (1 << 127) - 1
G = 3


def _rand_scalar():
    """
    生成群中的随机指数。

    这里避开 0、1 这类过小值，返回范围大致是 [2, P-2]。
    在 OT 中，这类随机指数会作为私钥或临时随机数使用。
    """

    return secrets.randbelow(P - 3) + 2


def _hash_to_bytes(value, length):
    """
    将 Diffie-Hellman 共享值扩展成指定长度的字节密钥。

    OT 中双方会通过指数运算得到 shared value。
    这个 shared value 本身是一个整数，不能直接和消息异或，
    所以这里用 SHA-256 把它扩展成和消息长度一致的伪随机字节流。

    参数：
        value  : 共享密钥对应的整数
        length : 需要生成的字节长度

    返回：
        长度为 length 的 bytes
    """

    # 先把整数转换成字节，作为哈希种子。
    seed = value.to_bytes((value.bit_length() + 7) // 8, "big")

    out = b""
    counter = 0

    # SHA-256 每次输出 32 字节。
    # 如果消息较长，就通过 counter 多次哈希拼接，直到长度足够。
    while len(out) < length:
        h = hashlib.sha256()
        h.update(seed)
        h.update(counter.to_bytes(4, "big"))
        out += h.digest()
        counter += 1

    return out[:length]


def _xor_bytes(a, b):
    """
    对两个 bytes 做逐字节异或。

    这里用于“流加密”形式的加密/解密：
        ciphertext = plaintext XOR key
        plaintext  = ciphertext XOR key
    """

    return bytes(x ^ y for x, y in zip(a, b))


def _encrypt(pk, msg):
    """
    使用接收方给出的 public key 加密一条消息。

    这是一个简化的 ElGamal-style 加密过程：
        e      : 发送方临时随机数
        c1     : G^e
        shared : pk^e
        key    : H(shared)
        c2     : msg XOR key

    参数：
        pk  : 接收方给出的 public key
        msg : 需要加密的消息，可以是 Python 对象

    返回：
        (c1, c2) 形式的密文
    """

    # pickle 用于把 Python 对象序列化成 bytes。
    # 这样 msg 可以是 int、tuple、numpy 相关对象等。
    msg_bytes = pickle.dumps(msg)

    # 发送方生成一次性随机数 e。
    e = _rand_scalar()

    # c1 = G^e mod P，接收方后续用自己的 sk 可以算出 shared。
    c1 = pow(G, e, P)

    # shared = pk^e mod P。
    # 如果接收方确实知道 pk 对应的私钥 sk，就能算出相同 shared。
    shared = pow(pk, e, P)

    # 从 shared 派生出和消息等长的密钥流。
    key = _hash_to_bytes(shared, len(msg_bytes))

    # 用异或完成加密。
    c2 = _xor_bytes(msg_bytes, key)

    return c1, c2


def _decrypt(sk, ct):
    """
    使用接收方私钥解密密文。

    对于密文：
        c1 = G^e
        c2 = msg XOR H(pk^e)

    接收方已知：
        sk
        pk = G^sk

    因此可以计算：
        shared = c1^sk = (G^e)^sk = G^(e*sk)

    这和发送方计算的：
        pk^e = (G^sk)^e = G^(sk*e)
    是同一个值。
    """

    c1, c2 = ct

    # 还原 Diffie-Hellman 共享值。
    shared = pow(c1, sk, P)

    # 生成和加密阶段相同的密钥流。
    key = _hash_to_bytes(shared, len(c2))

    # 再异或一次即可恢复明文 bytes。
    msg_bytes = _xor_bytes(c2, key)

    # 反序列化回原始 Python 对象。
    return pickle.loads(msg_bytes)


def ot_send(conn, m0, m1):
    """
    1-out-of-2 OT 的发送方逻辑。

    发送方持有两条消息：
        m0, m1

    接收方只能根据自己的 choice 得到其中一条：
        choice = 0 -> 得到 m0
        choice = 1 -> 得到 m1

    发送方不知道接收方选的是哪一条。
    """

    # 接收方发来两个 public key：
    # 一个是真正有私钥的 pk_choice；
    # 另一个是 dummy pk。

    # 发送方并不知道哪个 public key 对应接收方真正知道的私钥。
    pk0, pk1 = recv_data(conn)

    # 分别用 pk0、pk1 加密 m0、m1。
    # 接收方只能解开自己 choice 对应的那一条。
    ct0 = _encrypt(pk0, m0)
    ct1 = _encrypt(pk1, m1)

    # 把两条密文都发给接收方。
    send_data(conn, (ct0, ct1))


def ot_recv(conn, choice):
    """
    1-out-of-2 OT 的接收方逻辑。

    参数：
        choice : 只能是 0 或 1

    返回：
        choice 对应的消息：
            choice = 0 -> m0
            choice = 1 -> m1

    接收方不会得到另一条消息。
    """

    choice = int(choice)

    # 接收方只为自己想要的那条消息生成真实私钥 sk。
    sk = _rand_scalar()
    pk_choice = pow(G, sk, P)

    # 另一个 public key 是随机群元素。
    # 接收方不保存它对应的私钥，因此后续无法解密另一条消息。
    pk_dummy = pow(G, _rand_scalar(), P)

    # 根据 choice 决定哪个位置放真实 public key。
    if choice == 0:
        pk0 = pk_choice
        pk1 = pk_dummy
    else:
        pk0 = pk_dummy
        pk1 = pk_choice

    # 把两个 public key 发给发送方。
    # 发送方无法区分哪个是真实可解密的 public key。
    send_data(conn, (pk0, pk1))

    # 接收两条密文。
    ct0, ct1 = recv_data(conn)

    # 只解密 choice 对应的那一条。
    # 另一条由于没有对应私钥，无法解密。
    if choice == 0:
        return _decrypt(sk, ct0)
    else:
        return _decrypt(sk, ct1)


def ot_send_batch_bits(conn, m0_arr, m1_arr):
    """
    对 bit 数组逐元素执行 1-out-of-2 OT。

    m0_arr 和 m1_arr 是两个形状相同的数组。
    对每个位置 i：
        发送方持有 m0_arr[i] 和 m1_arr[i]
        接收方根据 choice_arr[i] 选择其中一个

    这个函数是发送方批量版本。
    """

    # 展平成一维，方便逐元素执行 OT。
    flat0 = m0_arr.reshape(-1)
    flat1 = m1_arr.reshape(-1)

    # 每个 bit 位置单独执行一次基础 OT。
    # 这是功能复现版本，重点是清楚验证 OT 流程；
    # 性能优化版本通常会使用 OT Extension 批量处理。
    for m0, m1 in zip(flat0, flat1):
        ot_send(conn, int(m0), int(m1))


def ot_recv_batch_bits(conn, choice_arr):
    """
    对 bit 数组逐元素执行 1-out-of-2 OT。

    choice_arr 是选择 bit 数组。
    对每个位置 i：
        choice_arr[i] = 0 -> 接收 m0_arr[i]
        choice_arr[i] = 1 -> 接收 m1_arr[i]

    返回：
        与 choice_arr 同 shape 的结果数组。
    """

    shape = choice_arr.shape
    flat_choice = choice_arr.reshape(-1)

    out = []

    # 对每个 choice 单独执行一次基础 OT。
    for c in flat_choice:
        out.append(ot_recv(conn, int(c)))

    # 恢复成原来的数组形状。
    return np.array(out, dtype=np.uint32).reshape(shape)