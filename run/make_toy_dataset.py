import os
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_FILE = os.path.join(DATA_DIR, "toy_cnn_dataset.npz")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    X_test = np.array(
        [
            [
                [
                    [0.5, 1.0, 1.5],
                    [2.0, 2.5, 3.0],
                    [3.5, 4.0, 4.5]
                ]
            ],
            [
                [
                    [-1.0, -0.5, 0.0],
                    [0.5, 1.0, 1.5],
                    [2.0, 2.5, 3.0]
                ]
            ],
            [
                [
                    [1.0, 0.0, -1.0],
                    [2.0, 0.5, -2.0],
                    [3.0, 1.0, -3.0]
                ]
            ],
            [
                [
                    [0.25, 0.5, 0.75],
                    [1.0, 1.25, 1.5],
                    [1.75, 2.0, 2.25]
                ]
            ],
        ],
        dtype=np.float64
    )

    y_test = np.array([0, 1, 2, 0], dtype=np.int64)

    np.savez(DATA_FILE, X_test=X_test, y_test=y_test)

    print("Saved dataset to:", DATA_FILE)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)


if __name__ == "__main__":
    main()