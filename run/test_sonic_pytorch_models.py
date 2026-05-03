import os
import sys
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from models.sonic_model_specs import SONIC_MODEL_SPECS
from models.sonic_pytorch_models import (
    build_sonic_torch_model,
    export_sonic_model_params,
)


def build_dummy_input(input_shape, batch_size=2):
    """
    构造模拟输入，不依赖真实 MNIST / CIFAR-10。
    """

    c, h, w = input_shape

    total = batch_size * c * h * w

    x = torch.linspace(
        start=-1.0,
        end=1.0,
        steps=total,
        dtype=torch.float32
    )

    x = x.reshape(batch_size, c, h, w)

    return x


def run_one_model(name):
    spec = SONIC_MODEL_SPECS[name]

    print("\n" + "=" * 90)
    print(f"[TEST] Sonic PyTorch Model: {name}")
    print("=" * 90)

    model = build_sonic_torch_model(
        name=name,
        seed=1234,
        use_bias=False
    )

    model.eval()

    x = build_dummy_input(
        input_shape=spec.input_shape,
        batch_size=2
    )

    with torch.no_grad():
        y, trace = model.forward_with_trace(x)

    expected_shape = (2, spec.num_classes)
    passed = tuple(y.shape) == expected_shape

    print("Dataset       :", spec.dataset)
    print("Input shape   :", tuple(x.shape))
    print("Output shape  :", tuple(y.shape))
    print("Expected shape:", expected_shape)
    print("Param count   :", model.count_parameters())

    print("\n----- Layer Shape Trace -----")
    for item in trace:
        print(
            f"{item['name']:>10s} | "
            f"{item['type']:<8s} | "
            f"{str(item['in_shape']):>22s} -> {item['out_shape']}"
        )

    print("\n----- Exported Params Summary -----")
    exported = export_sonic_model_params(model)

    for item in exported:
        layer_type = item["type"]
        layer_name = item["name"]

        if layer_type == "conv":
            print(
                f"{layer_name:>10s} | CONV    | "
                f"weight shape = {item['weight'].shape}, "
                f"stride = {item['stride']}, padding = {item['padding']}"
            )

        elif layer_type == "fc":
            print(
                f"{layer_name:>10s} | FC      | "
                f"weight shape = {item['weight'].shape}"
            )

        elif layer_type == "bn":
            print(
                f"{layer_name:>10s} | BN/SBN  | "
                f"eps1 shape = {item['eps1'].shape}, "
                f"eps2 shape = {item['eps2'].shape}"
            )

        elif layer_type == "relu":
            print(f"{layer_name:>10s} | ReLU    |")

        elif layer_type == "maxpool":
            print(
                f"{layer_name:>10s} | MaxPool | "
                f"kernel = {item['kernel_size']}, stride = {item['stride']}"
            )

        elif layer_type == "flatten":
            print(f"{layer_name:>10s} | Flatten |")

    if passed:
        print(f"\n{name} PyTorch forward shape test PASSED ✅")
    else:
        print(f"\n{name} PyTorch forward shape test FAILED ❌")

    return passed


def main():
    total_passed = 0
    total_failed = 0

    for name in ["M1", "M2", "C1", "C2"]:
        ok = run_one_model(name)

        if ok:
            total_passed += 1
        else:
            total_failed += 1

    print("\n" + "=" * 90)
    print("FINAL SONIC PYTORCH MODEL SHAPE SUMMARY")
    print("=" * 90)
    print("PASSED:", total_passed)
    print("FAILED:", total_failed)

    if total_failed == 0:
        print("All Sonic PyTorch model shape tests PASSED ✅")
    else:
        print("Some Sonic PyTorch model shape tests FAILED ❌")


if __name__ == "__main__":
    main()