import time
from contextlib import contextmanager


_STATS = {}


def reset_stats():
    """
    重置所有统计项。

    每次单独跑一个测试或一次推理前，可以调用这个函数清空旧数据。
    """
    global _STATS

    _STATS = {
        "bytes_sent": 0,
        "bytes_recv": 0,
        "send_messages": 0,
        "recv_messages": 0,

        "secure_mul_calls": 0,
        "secure_matmul_calls": 0,
        "bit_and_calls": 0,
        "b2a_calls": 0,
        "a2b_calls": 0,
        "trunc_calls": 0,
        "relu_calls": 0,
        "linear_secret_calls": 0,

        "offline_arith_triples": 0,
        "offline_bit_triples": 0,
        "offline_matmul_triples": 0,

        "secure_mul_time": 0.0,
        "secure_matmul_time": 0.0,
        "bit_and_time": 0.0,
        "b2a_time": 0.0,
        "a2b_time": 0.0,
        "trunc_time": 0.0,
        "relu_time": 0.0,
        "linear_secret_time": 0.0,

        "offline_time": 0.0,
        "online_time": 0.0,
        "total_time": 0.0,
    }


def inc(key, value=1):
    """
    对某个统计项做累加。

    如果 key 不存在，就自动创建，方便后续扩展新的统计项。
    """
    if key not in _STATS:
        _STATS[key] = 0

    _STATS[key] += value


def add_time(key, value):
    """
    对某个时间统计项做累加。
    """
    if key not in _STATS:
        _STATS[key] = 0.0

    _STATS[key] += value


@contextmanager
def time_block(key):
    """
    上下文管理器，用来统计某段代码的运行时间。
    """
    start = time.perf_counter()

    try:
        yield
    finally:
        end = time.perf_counter()
        add_time(key, end - start)


def get_stats():
    """
    返回当前统计信息的拷贝。
    """
    return dict(_STATS)


def print_report(title="Profiler Report"):
    """
    打印当前性能统计报告。
    """
    stats = get_stats()

    bytes_sent = stats.get("bytes_sent", 0)
    bytes_recv = stats.get("bytes_recv", 0)
    send_messages = stats.get("send_messages", 0)
    recv_messages = stats.get("recv_messages", 0)

    print("\n==========", title, "==========")
    print(f"bytes_sent        : {bytes_sent}")
    print(f"bytes_recv        : {bytes_recv}")
    print(f"total_comm_bytes  : {bytes_sent + bytes_recv}")

    print("\n----- Message Counts -----")
    print(f"send_messages     : {send_messages}")
    print(f"recv_messages     : {recv_messages}")
    print(f"total_messages    : {send_messages + recv_messages}")

    print("\n----- Offline Materials -----")
    print(f"offline_arith_triples  : {stats.get('offline_arith_triples', 0)}")
    print(f"offline_bit_triples    : {stats.get('offline_bit_triples', 0)}")
    print(f"offline_matmul_triples : {stats.get('offline_matmul_triples', 0)}")

    print("\n----- Call Counts -----")
    print(f"secure_mul_calls      : {stats.get('secure_mul_calls', 0)}")
    print(f"secure_matmul_calls   : {stats.get('secure_matmul_calls', 0)}")
    print(f"bit_and_calls         : {stats.get('bit_and_calls', 0)}")
    print(f"b2a_calls             : {stats.get('b2a_calls', 0)}")
    print(f"a2b_calls             : {stats.get('a2b_calls', 0)}")
    print(f"trunc_calls           : {stats.get('trunc_calls', 0)}")
    print(f"relu_calls            : {stats.get('relu_calls', 0)}")
    print(f"linear_calls          : {stats.get('linear_secret_calls', 0)}")

    print("\n----- Time Cost (seconds) -----")
    print(f"offline_time          : {stats.get('offline_time', 0.0):.6f}")
    print(f"online_time           : {stats.get('online_time', 0.0):.6f}")
    print(f"secure_mul_time       : {stats.get('secure_mul_time', 0.0):.6f}")
    print(f"secure_matmul_time    : {stats.get('secure_matmul_time', 0.0):.6f}")
    print(f"bit_and_time          : {stats.get('bit_and_time', 0.0):.6f}")
    print(f"b2a_time              : {stats.get('b2a_time', 0.0):.6f}")
    print(f"a2b_time              : {stats.get('a2b_time', 0.0):.6f}")
    print(f"trunc_time            : {stats.get('trunc_time', 0.0):.6f}")
    print(f"relu_time             : {stats.get('relu_time', 0.0):.6f}")
    print(f"linear_time           : {stats.get('linear_secret_time', 0.0):.6f}")
    print(f"total_time            : {stats.get('total_time', 0.0):.6f}")
    print("====================================\n")


reset_stats()