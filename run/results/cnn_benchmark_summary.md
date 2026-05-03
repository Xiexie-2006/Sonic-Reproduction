# PyTorch CNN vs MPC CNN Benchmark Summary

## 1. Correctness and Performance

| Batch | Passed | Max Error | Offline Time (s) | Online Time (s) | Total Time (s) | Communication (MB) | Messages |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | True | 0.000000 | 2.5271 | 0.3801 | 2.9073 | 10.0361 | 3632 |
| 2 | True | 0.000000 | 5.6319 | 0.5584 | 6.1905 | 19.7055 | 3632 |
| 4 | True | 0.000000 | 10.1372 | 0.1702 | 10.3078 | 39.0445 | 3632 |

## 2. Protocol Call Counts

| Batch | secure_mul | bit_and | B2A | A2B | trunc | SReLU/ReLU | Linear |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 74 | 192 | 65 | 2 | 2 | 1 | 2 |
| 2 | 74 | 192 | 65 | 2 | 2 | 1 | 2 |
| 4 | 74 | 192 | 65 | 2 | 2 | 1 | 2 |

## 3. Offline Materials

| Batch | Arithmetic Triples | Boolean Triples |
|---:|---:|---:|
| 1 | 460 | 480 |
| 2 | 460 | 480 |
| 4 | 460 | 480 |

## 4. Time Breakdown

| Batch | secure_mul (s) | bit_and (s) | B2A (s) | A2B (s) | trunc (s) | SReLU/ReLU (s) | Linear (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0801 | 0.2864 | 0.0668 | 0.2016 | 0.2699 | 0.0925 | 0.0173 |
| 2 | 0.0974 | 0.4450 | 0.0776 | 0.2892 | 0.3689 | 0.1647 | 0.0241 |
| 4 | 0.0530 | 0.1077 | 0.0523 | 0.0608 | 0.1141 | 0.0527 | 0.0029 |

## 5. Experiment Interpretation

- All tested CNN batch sizes pass correctness checks. The MPC output matches the PyTorch plaintext CNN output within the fixed-point tolerance.
- Communication grows with batch size because the OT-extension-based offline preprocessing materials scale with tensor size.
- Message count remains stable across batch sizes because the protocol structure is unchanged; larger batches mainly increase payload size.
- The main cost comes from offline OT-extension triple generation and online Boolean operations used in truncation and SReLU.
- This benchmark verifies the end-to-end chain: PyTorch CNN parameter export, fixed-point encoding, secret sharing, Secret Conv2D, secure truncation, SReLU, Secret Linear, and result decoding.
