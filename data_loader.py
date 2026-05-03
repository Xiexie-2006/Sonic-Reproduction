import os
import numpy as np


def load_npz_dataset(path, max_samples=None):
    # 将传入路径转成绝对路径，避免不同运行目录下找不到文件。
    path = os.path.abspath(path)

    # 如果数据集文件不存在，给出明确提示。
    # 这里提示用户先运行 make_toy_dataset.py 生成测试数据。
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Please run: python run/make_toy_dataset.py"
        )

    # 读取 npz 数据文件。
    data = np.load(path)

    # 当前测试数据至少需要包含 X_test 和 y_test。
    # X_test 是输入样本，y_test 是对应标签。
    if "X_test" not in data:
        raise KeyError("Dataset must contain X_test")

    if "y_test" not in data:
        raise KeyError("Dataset must contain y_test")

    # 输入统一转成 float64，便于后续 fixed-point 编码。
    X_test = data["X_test"].astype(np.float64)

    # 标签转成 int64，方便后续和预测类别做比较。
    y_test = data["y_test"].astype(np.int64)

    # 如果只想跑少量样本，可以通过 max_samples 截取前几个样本。
    # 这在 MPC 测试中比较有用，因为安全推理开销较大。
    if max_samples is not None:
        X_test = X_test[:max_samples]
        y_test = y_test[:max_samples]

    return X_test, y_test