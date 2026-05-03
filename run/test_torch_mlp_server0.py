import socket
import numpy as np
import torch

from net.socket_utils import send_data, recv_data
from net.profiler import reset_stats, print_report, time_block

from mpc.share import share_arith, reconstruct_arith, MOD
from mpc.triple_pool import setup_triple_pool
from mpc.secure_nn_secret import secure_two_layer_mlp_secret

from models.simple_mlp import SimpleMLP, export_numpy_params


HOST = "127.0.0.1"
PORT = 9000


def to_ring(x):
    return (np.array(x, dtype=np.int64) % MOD).astype(np.uint32)


def decode_signed(x):
    return x.astype(np.uint32).view(np.int32)


def main():
    reset_stats()

    s = socket.socket()
    s.connect((HOST, PORT))

    model = SimpleMLP()
    model.eval()

    X_plain = np.array([[5, -3]], dtype=np.int64)

    with torch.no_grad():
        x_torch = torch.tensor(X_plain, dtype=torch.float32)
        y_torch = model(x_torch).detach().cpu().numpy().astype(np.int64)

    W1_plain, b1_plain, W2_plain, b2_plain = export_numpy_params(model)

    X = to_ring(X_plain)
    W1 = to_ring(W1_plain)
    b1 = to_ring(b1_plain)
    W2 = to_ring(W2_plain)
    b2 = to_ring(b2_plain)

    X0, X1 = share_arith(X)
    W10, W11 = share_arith(W1)
    b10, b11 = share_arith(b1)
    W20, W21 = share_arith(W2)
    b20, b21 = share_arith(b2)

    arith_plan = [
        ((1, 2), 40),
        ((1, 1), 40),
    ]

    bit_plan = [
        ((1, 2), 120),
    ]

    with time_block("total_time"):

        send_data(s, (
            "TORCH_MLP_CONFIG",
            arith_plan,
            bit_plan
        ))

        with time_block("offline_time"):
            setup_triple_pool(
                conn=s,
                party_id=0,
                arith_plan=arith_plan,
                bit_plan=bit_plan
            )

        send_data(s, (X1, W11, b11, W21, b21))

        with time_block("online_time"):
            Y0 = secure_two_layer_mlp_secret(
                x_i=X0,
                W1_i=W10,
                b1_i=b10,
                W2_i=W20,
                b2_i=b20,
                conn=s,
                party_id=0
            )

            Y1 = recv_data(s)

    Y = reconstruct_arith(Y0, Y1)
    Y_signed = decode_signed(Y)

    print("===== PyTorch MLP vs MPC MLP Test =====")
    print("X_plain =", X_plain)
    print("PyTorch Y =", y_torch)

    print("\n===== Exported Parameters =====")
    print("W1_plain =")
    print(W1_plain)
    print("b1_plain =", b1_plain)
    print("W2_plain =")
    print(W2_plain)
    print("b2_plain =", b2_plain)

    print("\n===== MPC Output =====")
    print("MPC Y =", Y_signed)

    if np.array_equal(Y_signed, y_torch):
        print("PyTorch MLP vs MPC MLP test PASSED ✅")
    else:
        print("PyTorch MLP vs MPC MLP test FAILED ❌")

    print_report("Party0 PyTorch MLP Profiler")

    s.close()


if __name__ == "__main__":
    main()