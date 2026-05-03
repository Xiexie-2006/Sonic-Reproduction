import os
import sys
import socket
import numpy as np
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from models.sonic_pytorch_models import (
    build_sonic_torch_model,
    export_sonic_model_params,
)

from mpc.share import share_arith, reconstruct_arith
from mpc.fixed_point import encode_fixed, decode_fixed
from mpc.triple_dealer import setup_triple_pool_by_dealer
from mpc.secure_m2_fixed import secure_m2_fixed_inference


HOST = "127.0.0.1"
PORT = 9000


def build_dummy_mnist_input(batch_size=1):
    total = batch_size * 1 * 28 * 28

    x = torch.linspace(
        start=-1.0,
        end=1.0,
        steps=total,
        dtype=torch.float32
    )

    x = x.reshape(batch_size, 1, 28, 28)

    return x


def get_exported(exported, name):
    for item in exported:
        if item["name"] == name:
            return item

    raise KeyError(f"Cannot find exported layer: {name}")


def zero_bias(out_dim, scale_bits):
    return encode_fixed(
        np.zeros((out_dim,), dtype=np.float64),
        scale_bits * 2
    )


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    # 这里从 8 改成 12。
    # M2 的 PyTorch logits 很小，scale_bits=8 时最小精度是 1/256=0.00390625，
    # 容易把多个 logits 量化成 0，导致 argmax 被影响。
    # scale_bits=12 时最小精度是 1/4096=0.000244，更适合 M2。
    scale_bits = 12
    batch_size = 1

    model = build_sonic_torch_model(
        name="M2",
        seed=1234,
        use_bias=False
    )

    model.eval()

    x_torch = build_dummy_mnist_input(batch_size=batch_size)

    with torch.no_grad():
        y_torch = model(x_torch).detach().cpu().numpy().astype(np.float64)

    exported = export_sonic_model_params(model)

    conv1 = get_exported(exported, "conv1")
    bn1 = get_exported(exported, "bn1")

    conv2 = get_exported(exported, "conv2")
    bn2 = get_exported(exported, "bn2")

    fc1 = get_exported(exported, "fc1")
    bn3 = get_exported(exported, "bn3")

    fc2 = get_exported(exported, "fc2")
    bn4 = get_exported(exported, "bn4")

    X_plain = x_torch.detach().cpu().numpy().astype(np.float64)

    conv1_w_plain = conv1["weight"]
    conv2_w_plain = conv2["weight"]

    fc1_w_plain = fc1["weight"]
    fc2_w_plain = fc2["weight"]

    X = encode_fixed(X_plain, scale_bits)

    conv1_w = encode_fixed(conv1_w_plain, scale_bits)
    conv2_w = encode_fixed(conv2_w_plain, scale_bits)

    fc1_w = encode_fixed(fc1_w_plain, scale_bits)
    fc2_w = encode_fixed(fc2_w_plain, scale_bits)

    conv1_b = zero_bias(16, scale_bits)
    conv2_b = zero_bias(16, scale_bits)

    fc1_b = zero_bias(100, scale_bits)
    fc2_b = zero_bias(10, scale_bits)

    bn1_eps1 = encode_fixed(bn1["eps1"], scale_bits)
    bn2_eps1 = encode_fixed(bn2["eps1"], scale_bits)
    bn3_eps1 = encode_fixed(bn3["eps1"], scale_bits)
    bn4_eps1 = encode_fixed(bn4["eps1"], scale_bits)

    bn1_eps2 = encode_fixed(bn1["eps2"], scale_bits * 2)
    bn2_eps2 = encode_fixed(bn2["eps2"], scale_bits * 2)
    bn3_eps2 = encode_fixed(bn3["eps2"], scale_bits * 2)
    bn4_eps2 = encode_fixed(bn4["eps2"], scale_bits * 2)

    X0, X1 = share_arith(X)

    conv1_w0, conv1_w1 = share_arith(conv1_w)
    conv1_b0, conv1_b1 = share_arith(conv1_b)
    bn1_eps10, bn1_eps11 = share_arith(bn1_eps1)
    bn1_eps20, bn1_eps21 = share_arith(bn1_eps2)

    conv2_w0, conv2_w1 = share_arith(conv2_w)
    conv2_b0, conv2_b1 = share_arith(conv2_b)
    bn2_eps10, bn2_eps11 = share_arith(bn2_eps1)
    bn2_eps20, bn2_eps21 = share_arith(bn2_eps2)

    fc1_w0, fc1_w1 = share_arith(fc1_w)
    fc1_b0, fc1_b1 = share_arith(fc1_b)
    bn3_eps10, bn3_eps11 = share_arith(bn3_eps1)
    bn3_eps20, bn3_eps21 = share_arith(bn3_eps2)

    fc2_w0, fc2_w1 = share_arith(fc2_w)
    fc2_b0, fc2_b1 = share_arith(fc2_b)
    bn4_eps10, bn4_eps11 = share_arith(bn4_eps1)
    bn4_eps20, bn4_eps21 = share_arith(bn4_eps2)

    # M2 main tensor shapes:
    conv1_mul_shape = (batch_size * 24 * 24, 16)
    conv1_feature_shape = (batch_size, 16, 24, 24)
    pool1_shape = (batch_size, 16, 12, 12)

    conv2_mul_shape = (batch_size * 8 * 8, 16)
    conv2_feature_shape = (batch_size, 16, 8, 8)
    pool2_shape = (batch_size, 16, 4, 4)

    fc1_shape = (batch_size, 100)
    fc2_shape = (batch_size, 10)

    # 使用 trusted offline dealer。
    # 当前 M2 功能测试重点是验证论文模型结构的 MPC 推理链路。
    arith_plan = [
        (conv1_mul_shape, 80),
        (conv1_feature_shape, 140),
        (pool1_shape, 120),

        (conv2_mul_shape, 460),
        (conv2_feature_shape, 140),
        (pool2_shape, 120),

        (fc1_shape, 420),
        (fc2_shape, 240),
    ]

    bit_plan = [
        (conv1_feature_shape, 220),
        (pool1_shape, 420),

        (conv2_feature_shape, 220),
        (pool2_shape, 420),

        (fc1_shape, 260),
        (fc2_shape, 180),
    ]

    with time_block("total_time"):

        send_data(s, (
            "M2_MPC_CONFIG",
            arith_plan,
            bit_plan
        ))

        with time_block("offline_time"):
            setup_triple_pool_by_dealer(
                party_id=0,
                arith_plan=arith_plan,
                bit_plan=bit_plan,
                seed=202406
            )

        send_data(s, (
            X1,

            conv1_w1,
            conv1_b1,
            bn1_eps11,
            bn1_eps21,

            conv2_w1,
            conv2_b1,
            bn2_eps11,
            bn2_eps21,

            fc1_w1,
            fc1_b1,
            bn3_eps11,
            bn3_eps21,

            fc2_w1,
            fc2_b1,
            bn4_eps11,
            bn4_eps21,

            scale_bits
        ))

        with time_block("online_time"):
            y0 = secure_m2_fixed_inference(
                x_i=X0,

                conv1_w_i=conv1_w0,
                conv1_b_i=conv1_b0,
                bn1_eps1_i=bn1_eps10,
                bn1_eps2_i=bn1_eps20,

                conv2_w_i=conv2_w0,
                conv2_b_i=conv2_b0,
                bn2_eps1_i=bn2_eps10,
                bn2_eps2_i=bn2_eps20,

                fc1_w_i=fc1_w0,
                fc1_b_i=fc1_b0,
                bn3_eps1_i=bn3_eps10,
                bn3_eps2_i=bn3_eps20,

                fc2_w_i=fc2_w0,
                fc2_b_i=fc2_b0,
                bn4_eps1_i=bn4_eps10,
                bn4_eps2_i=bn4_eps20,

                scale_bits=scale_bits,
                conn=s,
                party_id=0
            )

            y1 = recv_data(s)

    y_ring = reconstruct_arith(y0, y1)
    y_mpc = decode_fixed(y_ring, scale_bits)

    diff = np.abs(y_mpc - y_torch)
    max_abs_error = float(np.max(diff))

    pred_torch = np.argmax(y_torch, axis=1)
    pred_mpc = np.argmax(y_mpc, axis=1)

    print("===== Sonic M2 MPC Functional Test =====")
    print("scale_bits =", scale_bits)
    print("Input shape =", X_plain.shape)
    print("PyTorch output shape =", y_torch.shape)
    print("MPC output shape =", y_mpc.shape)

    print("\nPyTorch logits =")
    print(y_torch)

    print("\nMPC logits =")
    print(y_mpc)

    print("\nabs_error =")
    print(diff)

    print("max_abs_error =", max_abs_error)
    print("PyTorch pred =", pred_torch)
    print("MPC pred     =", pred_mpc)

    # scale_bits=12 后精度更细，这里阈值收紧到 0.05。
    pred_same = np.array_equal(pred_torch, pred_mpc)
    logits_close = max_abs_error <= 0.05

    if pred_same and logits_close:
        print("Sonic M2 MPC functional test PASSED ✅")
    else:
        print("Sonic M2 MPC functional test FAILED ❌")

    print_report("Party0 Sonic M2 MPC Profiler")

    s.close()


if __name__ == "__main__":
    main()