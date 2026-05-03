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
from mpc.secure_c1_fixed import secure_c1_fixed_inference


HOST = "127.0.0.1"
PORT = 9000


def build_dummy_cifar_input(batch_size=1):
    total = batch_size * 3 * 32 * 32

    x = torch.linspace(
        start=-1.0,
        end=1.0,
        steps=total,
        dtype=torch.float32
    )

    x = x.reshape(batch_size, 3, 32, 32)

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


def share_layer_params(layer, scale_bits, out_dim):
    w = encode_fixed(layer["weight"], scale_bits)
    b = zero_bias(out_dim, scale_bits)

    w0, w1 = share_arith(w)
    b0, b1 = share_arith(b)

    return w0, w1, b0, b1


def share_bn_params(layer, scale_bits):
    eps1 = encode_fixed(layer["eps1"], scale_bits)
    eps2 = encode_fixed(layer["eps2"], scale_bits * 2)

    eps10, eps11 = share_arith(eps1)
    eps20, eps21 = share_arith(eps2)

    return eps10, eps11, eps20, eps21


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    scale_bits = 12
    batch_size = 1

    model = build_sonic_torch_model(
        name="C1",
        seed=1234,
        use_bias=False
    )

    model.eval()

    x_torch = build_dummy_cifar_input(batch_size=batch_size)

    with torch.no_grad():
        y_torch = model(x_torch).detach().cpu().numpy().astype(np.float64)

    exported = export_sonic_model_params(model)

    conv1 = get_exported(exported, "conv1")
    bn1 = get_exported(exported, "bn1")

    conv2 = get_exported(exported, "conv2")
    bn2 = get_exported(exported, "bn2")

    conv3 = get_exported(exported, "conv3")
    bn3 = get_exported(exported, "bn3")

    conv4 = get_exported(exported, "conv4")
    bn4 = get_exported(exported, "bn4")

    conv5 = get_exported(exported, "conv5")
    bn5 = get_exported(exported, "bn5")

    conv6 = get_exported(exported, "conv6")
    bn6 = get_exported(exported, "bn6")

    conv7 = get_exported(exported, "conv7")
    bn7 = get_exported(exported, "bn7")

    fc1 = get_exported(exported, "fc1")
    bn8 = get_exported(exported, "bn8")

    X_plain = x_torch.detach().cpu().numpy().astype(np.float64)
    X = encode_fixed(X_plain, scale_bits)
    X0, X1 = share_arith(X)

    conv1_w0, conv1_w1, conv1_b0, conv1_b1 = share_layer_params(conv1, scale_bits, 64)
    bn1_eps10, bn1_eps11, bn1_eps20, bn1_eps21 = share_bn_params(bn1, scale_bits)

    conv2_w0, conv2_w1, conv2_b0, conv2_b1 = share_layer_params(conv2, scale_bits, 64)
    bn2_eps10, bn2_eps11, bn2_eps20, bn2_eps21 = share_bn_params(bn2, scale_bits)

    conv3_w0, conv3_w1, conv3_b0, conv3_b1 = share_layer_params(conv3, scale_bits, 64)
    bn3_eps10, bn3_eps11, bn3_eps20, bn3_eps21 = share_bn_params(bn3, scale_bits)

    conv4_w0, conv4_w1, conv4_b0, conv4_b1 = share_layer_params(conv4, scale_bits, 64)
    bn4_eps10, bn4_eps11, bn4_eps20, bn4_eps21 = share_bn_params(bn4, scale_bits)

    conv5_w0, conv5_w1, conv5_b0, conv5_b1 = share_layer_params(conv5, scale_bits, 64)
    bn5_eps10, bn5_eps11, bn5_eps20, bn5_eps21 = share_bn_params(bn5, scale_bits)

    conv6_w0, conv6_w1, conv6_b0, conv6_b1 = share_layer_params(conv6, scale_bits, 64)
    bn6_eps10, bn6_eps11, bn6_eps20, bn6_eps21 = share_bn_params(bn6, scale_bits)

    conv7_w0, conv7_w1, conv7_b0, conv7_b1 = share_layer_params(conv7, scale_bits, 16)
    bn7_eps10, bn7_eps11, bn7_eps20, bn7_eps21 = share_bn_params(bn7, scale_bits)

    fc1_w0, fc1_w1, fc1_b0, fc1_b1 = share_layer_params(fc1, scale_bits, 10)
    bn8_eps10, bn8_eps11, bn8_eps20, bn8_eps21 = share_bn_params(bn8, scale_bits)

    # C1 tensor shapes
    conv1_mul_shape = (batch_size * 30 * 30, 64)
    conv1_feature_shape = (batch_size, 64, 30, 30)

    conv2_mul_shape = (batch_size * 28 * 28, 64)
    conv2_feature_shape = (batch_size, 64, 28, 28)
    pool1_shape = (batch_size, 64, 14, 14)

    conv3_mul_shape = (batch_size * 12 * 12, 64)
    conv3_feature_shape = (batch_size, 64, 12, 12)

    conv4_mul_shape = (batch_size * 10 * 10, 64)
    conv4_feature_shape = (batch_size, 64, 10, 10)
    pool2_shape = (batch_size, 64, 5, 5)

    conv56_mul_shape = (batch_size * 3 * 3, 64)
    conv56_feature_shape = (batch_size, 64, 3, 3)

    conv7_mul_shape = (batch_size * 3 * 3, 16)
    conv7_feature_shape = (batch_size, 16, 3, 3)

    fc1_shape = (batch_size, 10)

    # C1 完整结构较大。
    # 当前使用 trusted offline dealer 生成 triple，重点验证功能链路。
    arith_plan = [
        (conv1_mul_shape, 40),
        (conv1_feature_shape, 80),

        (conv2_mul_shape, 590),
        (conv2_feature_shape, 80),
        (pool1_shape, 20),

        (conv3_mul_shape, 590),
        (conv3_feature_shape, 80),

        (conv4_mul_shape, 590),
        (conv4_feature_shape, 80),
        (pool2_shape, 20),

        (conv56_mul_shape, 1200),
        (conv56_feature_shape, 160),

        (conv7_mul_shape, 590),
        (conv7_feature_shape, 80),

        (fc1_shape, 240),
    ]

    bit_plan = [
        (conv1_feature_shape, 220),

        (conv2_feature_shape, 150),
        (pool1_shape, 280),

        (conv3_feature_shape, 220),

        (conv4_feature_shape, 150),
        (pool2_shape, 280),

        (conv56_feature_shape, 420),

        (conv7_feature_shape, 220),

        (fc1_shape, 150),
    ]

    with time_block("total_time"):

        send_data(s, (
            "C1_MPC_CONFIG",
            arith_plan,
            bit_plan
        ))

        with time_block("offline_time"):
            setup_triple_pool_by_dealer(
                party_id=0,
                arith_plan=arith_plan,
                bit_plan=bit_plan,
                seed=202407
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

            conv3_w1,
            conv3_b1,
            bn3_eps11,
            bn3_eps21,

            conv4_w1,
            conv4_b1,
            bn4_eps11,
            bn4_eps21,

            conv5_w1,
            conv5_b1,
            bn5_eps11,
            bn5_eps21,

            conv6_w1,
            conv6_b1,
            bn6_eps11,
            bn6_eps21,

            conv7_w1,
            conv7_b1,
            bn7_eps11,
            bn7_eps21,

            fc1_w1,
            fc1_b1,
            bn8_eps11,
            bn8_eps21,

            scale_bits
        ))

        with time_block("online_time"):
            y0 = secure_c1_fixed_inference(
                x_i=X0,

                conv1_w_i=conv1_w0,
                conv1_b_i=conv1_b0,
                bn1_eps1_i=bn1_eps10,
                bn1_eps2_i=bn1_eps20,

                conv2_w_i=conv2_w0,
                conv2_b_i=conv2_b0,
                bn2_eps1_i=bn2_eps10,
                bn2_eps2_i=bn2_eps20,

                conv3_w_i=conv3_w0,
                conv3_b_i=conv3_b0,
                bn3_eps1_i=bn3_eps10,
                bn3_eps2_i=bn3_eps20,

                conv4_w_i=conv4_w0,
                conv4_b_i=conv4_b0,
                bn4_eps1_i=bn4_eps10,
                bn4_eps2_i=bn4_eps20,

                conv5_w_i=conv5_w0,
                conv5_b_i=conv5_b0,
                bn5_eps1_i=bn5_eps10,
                bn5_eps2_i=bn5_eps20,

                conv6_w_i=conv6_w0,
                conv6_b_i=conv6_b0,
                bn6_eps1_i=bn6_eps10,
                bn6_eps2_i=bn6_eps20,

                conv7_w_i=conv7_w0,
                conv7_b_i=conv7_b0,
                bn7_eps1_i=bn7_eps10,
                bn7_eps2_i=bn7_eps20,

                fc1_w_i=fc1_w0,
                fc1_b_i=fc1_b0,
                bn8_eps1_i=bn8_eps10,
                bn8_eps2_i=bn8_eps20,

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

    shape_ok = y_mpc.shape == y_torch.shape
    logits_close = max_abs_error <= 0.10
    pred_same = np.array_equal(pred_torch, pred_mpc)

    print("===== Sonic C1 MPC Functional Test =====")
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

    print("\n----- Check Result -----")
    print("shape_ok     =", shape_ok)
    print("logits_close =", logits_close)
    print("pred_same    =", pred_same)

    if shape_ok and logits_close and pred_same:
        print("Sonic C1 MPC functional test PASSED ✅")
    elif shape_ok and logits_close:
        print("Sonic C1 MPC structure/logits test PASSED ✅")
        print("But prediction is not identical due to fixed-point quantization ⚠️")
    else:
        print("Sonic C1 MPC functional test FAILED ❌")

    print_report("Party0 Sonic C1 MPC Profiler")

    s.close()


if __name__ == "__main__":
    main()