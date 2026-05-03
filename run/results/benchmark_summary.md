# Sonic-style MPC Benchmark Summary

## 1. Correctness and Performance

| Batch | Passed | Max Error | Offline Time (s) | Online Time (s) | Total Time (s) | Communication (MB) | Messages |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | True | 0.000000 | 0.5411 | 0.4557 | 0.9973 | 3.4351 | 2592 |
| 2 | True | 0.000000 | 0.7851 | 0.4316 | 1.2171 | 6.5975 | 2592 |
| 4 | True | 0.000000 | 1.1576 | 0.0871 | 1.2451 | 12.9224 | 2592 |
| 8 | True | 0.000000 | 1.9802 | 0.1997 | 2.1802 | 25.5722 | 2592 |

## 2. Protocol Call Counts

| Batch | secure_mul | bit_and | B2A | A2B | trunc | ReLU | Linear |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 70 | 192 | 65 | 2 | 2 | 1 | 2 |
| 2 | 70 | 192 | 65 | 2 | 2 | 1 | 2 |
| 4 | 70 | 192 | 65 | 2 | 2 | 1 | 2 |
| 8 | 70 | 192 | 65 | 2 | 2 | 1 | 2 |

## 3. Offline Materials

| Batch | Arithmetic Triples | Boolean Triples |
|---:|---:|---:|
| 1 | 240 | 400 |
| 2 | 240 | 400 |
| 4 | 240 | 400 |
| 8 | 240 | 400 |

## 4. Time Breakdown

| Batch | secure_mul (s) | bit_and (s) | B2A (s) | A2B (s) | trunc (s) | ReLU (s) | Linear (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0854 | 0.3586 | 0.0871 | 0.2347 | 0.3233 | 0.1310 | 0.0012 |
| 2 | 0.0910 | 0.3298 | 0.0798 | 0.2115 | 0.2926 | 0.1378 | 0.0011 |
| 4 | 0.0078 | 0.0741 | 0.0077 | 0.0268 | 0.0349 | 0.0513 | 0.0009 |
| 8 | 0.0636 | 0.1290 | 0.0646 | 0.0292 | 0.0948 | 0.1037 | 0.0009 |

## 5. Interpretation

- All tested batch sizes pass the correctness check, which means the MPC output matches the plaintext fixed-point inference result within the configured error bound.
- Communication grows with batch size because OT-extension-generated preprocessing materials scale with tensor shape.
- The number of messages remains stable across batch sizes in this experiment, because the protocol structure is unchanged and larger batches mainly increase payload size.
- The dominant online costs come from Boolean operations, A2B conversion, B2A conversion, and secure truncation, which is consistent with secure neural network inference protocols.
