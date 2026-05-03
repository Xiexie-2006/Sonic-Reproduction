import numpy as np

from mpc.share import MOD
from mpc.compare import secure_compare_zero
from mpc.b2a_secure import b2a_secure
from mpc.triple import get_arith_triple
from mpc.mul import secure_mul
from net.profiler import inc, time_block


def _public_const_share(value, shape, party_id):
    """
    生成公开常数 value 的 arithmetic share。

    party0 持有 value
    party1 持有 0

    两方相加后得到 value。

    这里用于把公开类别下标 cls 转成 arithmetic share，
    这样后面就可以和秘密分享状态下的 idx 一起参与更新计算。
    """

    if party_id == 0:
        return np.full(shape, value, dtype=np.uint32)
    else:
        return np.zeros(shape, dtype=np.uint32)


def secure_argmax(logits_i, conn, party_id, strict=True):
    """
    Secure Argmax for arithmetic shared logits.

    输入:
        logits_i:
            当前方持有的 logits arithmetic share
            shape = [batch, num_classes]

    输出:
        idx_i:
            当前方持有的 argmax index arithmetic share
            shape = [batch]

        max_i:
            当前方持有的 max logit arithmetic share
            shape = [batch]

    说明:
        如果 strict=True:
            使用 candidate > current_max 更新
            遇到相等值时保留更早的 index，和 PyTorch argmax 行为一致。

        如果 strict=False:
            使用 candidate >= current_max 更新
            遇到相等值时会更新为更靠后的 index。

    整体思路：
        从第 0 类开始作为当前最大值，
        然后逐个类别比较 candidate 和 current_max。
        如果 candidate 更大，就用 gate=1 更新 max 和 idx；
        否则 gate=0，保持原值不变。
    """

    # 记录 argmax 协议调用次数。
    inc("argmax_calls")

    # 统计 argmax 协议运行时间。
    with time_block("argmax_time"):
        batch_size, num_classes = logits_i.shape

        # 初始 max = 第 0 类 logit。
        # 每一方只保存自己那一份 arithmetic share。
        max_i = logits_i[:, 0].copy().astype(np.uint32)

        # 初始 index = 0。
        # 这里 0 是公开值，所以两方都初始化为 0 也可以表示 index=0。
        idx_i = np.zeros((batch_size,), dtype=np.uint32)

        for cls in range(1, num_classes):
            # 当前待比较的类别 logit。
            candidate_i = logits_i[:, cls].copy().astype(np.uint32)

            # diff = candidate - current_max
            #
            # 因为是在 Z_(2^32) 环上计算，所以减法需要通过 uint64 过渡，
            # 再对 MOD 取模，最后转回 uint32。
            diff_i = (
                candidate_i.astype(np.uint64)
                - max_i.astype(np.uint64)
            ) % MOD
            diff_i = diff_i.astype(np.uint32)

            # strict=True 时判断 candidate > current_max。
            #
            # secure_compare_zero 判断的是 x >= 0。
            # 为了实现严格大于：
            #   candidate > current_max
            # 等价于：
            #   candidate - current_max - 1 >= 0
            #
            # 只让 party0 减 1，是因为公开常数 1 的 arithmetic share
            # 可以表示为 party0 持有 1，party1 持有 0。
            if strict:
                diff_cmp_i = diff_i.copy()

                if party_id == 0:
                    diff_cmp_i = (
                        diff_cmp_i.astype(np.uint64)
                        - 1
                    ) % MOD
                    diff_cmp_i = diff_cmp_i.astype(np.uint32)
            else:
                # 非严格模式下直接判断 candidate - current_max >= 0。
                diff_cmp_i = diff_i

            # gate = 1 if candidate should replace current_max else 0
            #
            # secure_compare_zero 返回的是 Boolean share 形式的比较结果。
            gate_bool_i = secure_compare_zero(
                xi=diff_cmp_i,
                conn=conn,
                party_id=party_id
            )

            # 后面要用 gate 参与乘法更新 max 和 idx，
            # 所以需要把 Boolean share 转成 arithmetic share。
            gate_arith_i = b2a_secure(
                xb_i=gate_bool_i,
                conn=conn,
                party_id=party_id
            )

            # 更新 max:
            # max_new = max_old + gate * (candidate - max_old)
            #
            # 如果 gate=1，则 max_new = candidate；
            # 如果 gate=0，则 max_new = max_old。
            triple_max_i = get_arith_triple(
                conn=conn,
                party_id=party_id,
                shape=diff_i.shape
            )

            selected_diff_i = secure_mul(
                xi=gate_arith_i,
                yi=diff_i,
                triple_i=triple_max_i,
                conn=conn,
                party_id=party_id
            )

            max_i = (
                max_i.astype(np.uint64)
                + selected_diff_i.astype(np.uint64)
            ) % MOD
            max_i = max_i.astype(np.uint32)

            # 更新 index:
            # idx_new = idx_old + gate * (cls - idx_old)
            #
            # 这个写法和上面更新 max 是一样的。
            # gate=1 时更新到当前类别 cls，gate=0 时保持原 index。
            cls_share_i = _public_const_share(
                value=cls,
                shape=idx_i.shape,
                party_id=party_id
            )

            idx_diff_i = (
                cls_share_i.astype(np.uint64)
                - idx_i.astype(np.uint64)
            ) % MOD
            idx_diff_i = idx_diff_i.astype(np.uint32)

            triple_idx_i = get_arith_triple(
                conn=conn,
                party_id=party_id,
                shape=idx_i.shape
            )

            selected_idx_diff_i = secure_mul(
                xi=gate_arith_i,
                yi=idx_diff_i,
                triple_i=triple_idx_i,
                conn=conn,
                party_id=party_id
            )

            idx_i = (
                idx_i.astype(np.uint64)
                + selected_idx_diff_i.astype(np.uint64)
            ) % MOD
            idx_i = idx_i.astype(np.uint32)

        return idx_i, max_i