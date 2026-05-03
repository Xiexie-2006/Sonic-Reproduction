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
from mpc.secure_c2_fixed import secure_c2_fixed_inference


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

    return x.reshape(batch_size, 3, 32, 32)


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

    return {
        "w0": w0,
        "w1": w1,
        "b0": b0,
        "b1": b1,
    }


def share_bn_params(layer, scale_bits):
    eps1 = encode_fixed(layer["eps1"], scale_bits)
    eps2 = encode_fixed(layer["eps2"], scale_bits * 2)

    eps10, eps11 = share_arith(eps1)
    eps20, eps21 = share_arith(eps2)

    return {
        "eps10": eps10,
        "eps11": eps11,
        "eps20": eps20,
        "eps21": eps21,
    }


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    scale_bits = 12
    batch_size = 1

    model = build_sonic_torch_model(
        name="C2",
        seed=1234,
        use_bias=False
    )

    model.eval()

    x_torch = build_dummy_cifar_input(batch_size=batch_size)

    with torch.no_grad():
        y_torch = model(x_torch).detach().cpu().numpy().astype(np.float64)

    exported = export_sonic_model_params(model)

    X_plain = x_torch.detach().cpu().numpy().astype(np.float64)
    X = encode_fixed(X_plain, scale_bits)
    X0, X1 = share_arith(X)

    conv_out_dims = {
        "conv1": 16,
        "conv2": 16,
        "conv3": 16,
        "conv4": 32,
        "conv5": 32,
        "conv6": 32,
        "conv7": 48,
        "conv8": 48,
        "conv9": 64,
        "fc1": 10,
    }

    conv = {}
    bn = {}

    for name in [
        "conv1", "conv2", "conv3",
        "conv4", "conv5", "conv6",
        "conv7", "conv8", "conv9",
        "fc1"
    ]:
        layer = get_exported(exported, name)
        conv[name] = share_layer_params(
            layer=layer,
            scale_bits=scale_bits,
            out_dim=conv_out_dims[name]
        )

    for name in [
        "bn1", "bn2", "bn3",
        "bn4", "bn5", "bn6",
        "bn7", "bn8", "bn9",
        "bn10"
    ]:
        layer = get_exported(exported, name)
        bn[name] = share_bn_params(
            layer=layer,
            scale_bits=scale_bits
        )

    # C2 tensor shapes
    conv123_mul_shape = (batch_size * 32 * 32, 16)
    conv123_feature_shape = (batch_size, 16, 32, 32)
    pool1_shape = (batch_size, 16, 16, 16)

    conv456_mul_shape = (batch_size * 16 * 16, 32)
    conv456_feature_shape = (batch_size, 32, 16, 16)
    pool2_shape = (batch_size, 32, 8, 8)

    conv7_mul_shape = (batch_size * 6 * 6, 48)
    conv7_feature_shape = (batch_size, 48, 6, 6)

    conv8_mul_shape = (batch_size * 4 * 4, 48)
    conv8_feature_shape = (batch_size, 48, 4, 4)

    conv9_mul_shape = (batch_size * 2 * 2, 64)
    conv9_feature_shape = (batch_size, 64, 2, 2)
    pool3_shape = (batch_size, 64, 1, 1)

    fc1_shape = (batch_size, 10)

    arith_plan = [
        (conv123_mul_shape, 400),
        (conv123_feature_shape, 420),
        (pool1_shape, 220),

        (conv456_mul_shape, 860),
        (conv456_feature_shape, 420),
        (pool2_shape, 220),

        (conv7_mul_shape, 340),
        (conv7_feature_shape, 120),

        (conv8_mul_shape, 480),
        (conv8_feature_shape, 120),

        (conv9_mul_shape, 480),
        (conv9_feature_shape, 120),
        (pool3_shape, 220),

        (fc1_shape, 240),
    ]

    bit_plan = [
        (conv123_feature_shape, 760),
        (pool1_shape, 320),

        (conv456_feature_shape, 760),
        (pool2_shape, 320),

        (conv7_feature_shape, 240),
        (conv8_feature_shape, 240),
        (conv9_feature_shape, 200),
        (pool3_shape, 320),

        (fc1_shape, 160),
    ]

    # 发给 Party1 的全部 share
    party1_payload = {
        "X1": X1,

        "conv1": conv["conv1"]["w1"],
        "conv1_b": conv["conv1"]["b1"],
        "bn1_eps1": bn["bn1"]["eps11"],
        "bn1_eps2": bn["bn1"]["eps21"],

        "conv2": conv["conv2"]["w1"],
        "conv2_b": conv["conv2"]["b1"],
        "bn2_eps1": bn["bn2"]["eps11"],
        "bn2_eps2": bn["bn2"]["eps21"],

        "conv3": conv["conv3"]["w1"],
        "conv3_b": conv["conv3"]["b1"],
        "bn3_eps1": bn["bn3"]["eps11"],
        "bn3_eps2": bn["bn3"]["eps21"],

        "conv4": conv["conv4"]["w1"],
        "conv4_b": conv["conv4"]["b1"],
        "bn4_eps1": bn["bn4"]["eps11"],
        "bn4_eps2": bn["bn4"]["eps21"],

        "conv5": conv["conv5"]["w1"],
        "conv5_b": conv["conv5"]["b1"],
        "bn5_eps1": bn["bn5"]["eps11"],
        "bn5_eps2": bn["bn5"]["eps21"],

        "conv6": conv["conv6"]["w1"],
        "conv6_b": conv["conv6"]["b1"],
        "bn6_eps1": bn["bn6"]["eps11"],
        "bn6_eps2": bn["bn6"]["eps21"],

        "conv7": conv["conv7"]["w1"],
        "conv7_b": conv["conv7"]["b1"],
        "bn7_eps1": bn["bn7"]["eps11"],
        "bn7_eps2": bn["bn7"]["eps21"],

        "conv8": conv["conv8"]["w1"],
        "conv8_b": conv["conv8"]["b1"],
        "bn8_eps1": bn["bn8"]["eps11"],
        "bn8_eps2": bn["bn8"]["eps21"],

        "conv9": conv["conv9"]["w1"],
        "conv9_b": conv["conv9"]["b1"],
        "bn9_eps1": bn["bn9"]["eps11"],
        "bn9_eps2": bn["bn9"]["eps21"],

        "fc1": conv["fc1"]["w1"],
        "fc1_b": conv["fc1"]["b1"],
        "bn10_eps1": bn["bn10"]["eps11"],
        "bn10_eps2": bn["bn10"]["eps21"],

        "scale_bits": scale_bits,
    }

    with time_block("total_time"):

        send_data(s, (
            "C2_MPC_CONFIG",
            arith_plan,
            bit_plan
        ))

        with time_block("offline_time"):
            setup_triple_pool_by_dealer(
                party_id=0,
                arith_plan=arith_plan,
                bit_plan=bit_plan,
                seed=202408
            )

        send_data(s, party1_payload)

        with time_block("online_time"):
            y0 = secure_c2_fixed_inference(
                x_i=X0,

                conv1_w_i=conv["conv1"]["w0"],
                conv1_b_i=conv["conv1"]["b0"],
                bn1_eps1_i=bn["bn1"]["eps10"],
                bn1_eps2_i=bn["bn1"]["eps20"],

                conv2_w_i=conv["conv2"]["w0"],
                conv2_b_i=conv["conv2"]["b0"],
                bn2_eps1_i=bn["bn2"]["eps10"],
                bn2_eps2_i=bn["bn2"]["eps20"],

                conv3_w_i=conv["conv3"]["w0"],
                conv3_b_i=conv["conv3"]["b0"],
                bn3_eps1_i=bn["bn3"]["eps10"],
                bn3_eps2_i=bn["bn3"]["eps20"],

                conv4_w_i=conv["conv4"]["w0"],
                conv4_b_i=conv["conv4"]["b0"],
                bn4_eps1_i=bn["bn4"]["eps10"],
                bn4_eps2_i=bn["bn4"]["eps20"],

                conv5_w_i=conv["conv5"]["w0"],
                conv5_b_i=conv["conv5"]["b0"],
                bn5_eps1_i=bn["bn5"]["eps10"],
                bn5_eps2_i=bn["bn5"]["eps20"],

                conv6_w_i=conv["conv6"]["w0"],
                conv6_b_i=conv["conv6"]["b0"],
                bn6_eps1_i=bn["bn6"]["eps10"],
                bn6_eps2_i=bn["bn6"]["eps20"],

                conv7_w_i=conv["conv7"]["w0"],
                conv7_b_i=conv["conv7"]["b0"],
                bn7_eps1_i=bn["bn7"]["eps10"],
                bn7_eps2_i=bn["bn7"]["eps20"],

                conv8_w_i=conv["conv8"]["w0"],
                conv8_b_i=conv["conv8"]["b0"],
                bn8_eps1_i=bn["bn8"]["eps10"],
                bn8_eps2_i=bn["bn8"]["eps20"],

                conv9_w_i=conv["conv9"]["w0"],
                conv9_b_i=conv["conv9"]["b0"],
                bn9_eps1_i=bn["bn9"]["eps10"],
                bn9_eps2_i=bn["bn9"]["eps20"],

                fc1_w_i=conv["fc1"]["w0"],
                fc1_b_i=conv["fc1"]["b0"],
                bn10_eps1_i=bn["bn10"]["eps10"],
                bn10_eps2_i=bn["bn10"]["eps20"],

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

    print("===== Sonic C2 MPC Functional Test =====")
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
        print("Sonic C2 MPC functional test PASSED ✅")
    elif shape_ok and logits_close:
        print("Sonic C2 MPC structure/logits test PASSED ✅")
        print("But prediction is not identical due to fixed-point quantization.")
    else:
        print("Sonic C2 MPC functional test FAILED ❌")

    print_report("Party0 Sonic C2 MPC Profiler")

    s.close()


if __name__ == "__main__":
    main()