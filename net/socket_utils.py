import pickle
import struct

from net.profiler import inc


def send_data(conn, data):
    # 将 Python 对象序列化成字节流。
    # 当前项目中通信的数据可能是 numpy 数组、tuple、字符串标志等，
    # 用 pickle 可以比较方便地统一处理。
    data_bytes = pickle.dumps(data)

    # 发送前先打包 4 字节长度头。
    # 接收方先读长度，再按长度读取完整 payload，避免 TCP 粘包/拆包问题。
    header = struct.pack(">I", len(data_bytes))

    # 先发送长度头，再发送实际数据。
    conn.sendall(header)
    conn.sendall(data_bytes)

    # 统计发送字节数和消息数量。
    inc("bytes_sent", len(header) + len(data_bytes))
    inc("send_messages", 1)


def recv_data(conn):
    # 先接收 4 字节长度头。
    raw_len = recvall(conn, 4)

    if not raw_len:
        return None

    # 按大端格式解析 payload 长度。
    length = struct.unpack(">I", raw_len)[0]

    # 根据长度继续读取完整数据。
    data = recvall(conn, length)

    # 统计接收字节数和消息数量。
    inc("bytes_recv", 4 + length)
    inc("recv_messages", 1)

    # 反序列化恢复原始 Python 对象。
    return pickle.loads(data)


def recvall(conn, n):
    # TCP 的 recv 不保证一次就能读到 n 个字节。
    # 所以这里循环读取，直到长度满足要求。
    data = b""

    while len(data) < n:
        packet = conn.recv(n - len(data))

        # 如果对方关闭连接，packet 为空。
        if not packet:
            return None

        data += packet

    return data