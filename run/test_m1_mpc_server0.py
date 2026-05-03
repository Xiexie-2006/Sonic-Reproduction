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
from mpc.secure_m1_fixed import secure_m1_fixed_inference


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

    scale_bits = 8
    batch_size = 1

    model = build_sonic_torch_model(
        name="M1",
        seed=1234,
        use_bias=False
    )

    model.eval()

    x_torch = build_dummy_mnist_input(batch_size=batch_size)

    with torch.no_grad():
        y_torch = model(x_torch).detach().cpu().numpy().astype(np.float64)

    exported = export_sonic_model_params(model)

    fc1 = get_exported(exported, "fc1")
    bn1 = get_exported(exported, "bn1")
    fc2 = get_exported(exported, "fc2")
    bn2 = get_exported(exported, "bn2")
    fc3 = get_exported(exported, "fc3")
    bn3 = get_exported(exported, "bn3")

    X_plain = x_torch.detach().cpu().numpy().astype(np.float64)

    fc1_w_plain = fc1["weight"]
    fc2_w_plain = fc2["weight"]
    fc3_w_plain = fc3["weight"]

    bn1_eps1_plain = bn1["eps1"]
    bn1_eps2_plain = bn1["eps2"]
    bn2_eps1_plain = bn2["eps1"]
    bn2_eps2_plain = bn2["eps2"]
    bn3_eps1_plain = bn3["eps1"]
    bn3_eps2_plain = bn3["eps2"]

    X = encode_fixed(X_plain, scale_bits)

    fc1_w = encode_fixed(fc1_w_plain, scale_bits)
    fc2_w = encode_fixed(fc2_w_plain, scale_bits)
    fc3_w = encode_fixed(fc3_w_plain, scale_bits)

    fc1_b = zero_bias(128, scale_bits)
    fc2_b = zero_bias(128, scale_bits)
    fc3_b = zero_bias(10, scale_bits)

    bn1_eps1 = encode_fixed(bn1_eps1_plain, scale_bits)
    bn2_eps1 = encode_fixed(bn2_eps1_plain, scale_bits)
    bn3_eps1 = encode_fixed(bn3_eps1_plain, scale_bits)

    bn1_eps2 = encode_fixed(bn1_eps2_plain, scale_bits * 2)
    bn2_eps2 = encode_fixed(bn2_eps2_plain, scale_bits * 2)
    bn3_eps2 = encode_fixed(bn3_eps2_plain, scale_bits * 2)

    X0, X1 = share_arith(X)

    fc1_w0, fc1_w1 = share_arith(fc1_w)
    fc1_b0, fc1_b1 = share_arith(fc1_b)

    bn1_eps10, bn1_eps11 = share_arith(bn1_eps1)
    bn1_eps20, bn1_eps21 = share_arith(bn1_eps2)

    fc2_w0, fc2_w1 = share_arith(fc2_w)
    fc2_b0, fc2_b1 = share_arith(fc2_b)

    bn2_eps10, bn2_eps11 = share_arith(bn2_eps1)
    bn2_eps20, bn2_eps21 = share_arith(bn2_eps2)

    fc3_w0, fc3_w1 = share_arith(fc3_w)
    fc3_b0, fc3_b1 = share_arith(fc3_b)

    bn3_eps10, bn3_eps11 = share_arith(bn3_eps1)
    bn3_eps20, bn3_eps21 = share_arith(bn3_eps2)

    shape_128 = (batch_size, 128)
    shape_10 = (batch_size, 10)

    # M1 维度较大：
    # FC1: 784 -> 128
    # FC2: 128 -> 128
    # FC3: 128 -> 10
    #
    # 这里使用 trusted offline dealer 填充 triple pool。
    # 注意：server0 和 server1 的 seed 必须一致。
    arith_plan = [
        (shape_128, 2200),
        (shape_10, 600),
    ]

    bit_plan = [
        (shape_128, 1200),
        (shape_10, 360),
    ]

    with time_block("total_time"):

        send_data(s, (
            "M1_MPC_CONFIG",
            arith_plan,
            bit_plan
        ))

        with time_block("offline_time"):
            setup_triple_pool_by_dealer(
                party_id=0,
                arith_plan=arith_plan,
                bit_plan=bit_plan,
                seed=202405
            )

        send_data(s, (
            X1,

            fc1_w1,
            fc1_b1,
            bn1_eps11,
            bn1_eps21,

            fc2_w1,
            fc2_b1,
            bn2_eps11,
            bn2_eps21,

            fc3_w1,
            fc3_b1,
            bn3_eps11,
            bn3_eps21,

            scale_bits
        ))

        with time_block("online_time"):
            y0 = secure_m1_fixed_inference(
                x_i=X0,

                fc1_w_i=fc1_w0,
                fc1_b_i=fc1_b0,
                bn1_eps1_i=bn1_eps10,
                bn1_eps2_i=bn1_eps20,

                fc2_w_i=fc2_w0,
                fc2_b_i=fc2_b0,
                bn2_eps1_i=bn2_eps10,
                bn2_eps2_i=bn2_eps20,

                fc3_w_i=fc3_w0,
                fc3_b_i=fc3_b0,
                bn3_eps1_i=bn3_eps10,
                bn3_eps2_i=bn3_eps20,

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

    print("===== Sonic M1 MPC Functional Test =====")
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

    pred_same = np.array_equal(pred_torch, pred_mpc)
    logits_close = max_abs_error <= 0.25

    if pred_same and logits_close:
        print("Sonic M1 MPC functional test PASSED ✅")
    else:
        print("Sonic M1 MPC functional test FAILED ❌")

    print_report("Party0 Sonic M1 MPC Profiler")

    s.close()


if __name__ == "__main__":
    main()