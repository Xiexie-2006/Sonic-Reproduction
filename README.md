# Sonic 论文复现项目

本项目是对论文 **Securely Outsourcing Neural Network Inference to the Cloud with Lightweight Techniques** 中 Sonic 安全神经网络推理框架的阶段性复现。

项目主要围绕论文中的核心安全推理流程展开，复现内容包括 Additive Secret Sharing、Boolean Sharing、fixed-point 表示、安全乘法、A2B/B2A 转换、安全比较、SReLU、SBN、SFC、SCONV、SMP，以及 M1、M2、C1、C2 等模型的 fixed-point MPC 推理流程。

当前版本以 **功能级复现和协议流程验证** 为主，重点验证论文中安全算子和模型推理链路能否正确串联运行。

---

## 1. 项目说明

Sonic 论文的目标是在云端神经网络推理过程中，同时保护用户输入数据和模型参数隐私。论文采用两服务器半诚实、非串通模型，将用户输入和模型权重拆分为秘密分享形式，由两个服务器协同完成神经网络推理。

本项目基于 Python、NumPy、PyTorch 和 socket 通信实现了一个简化版 Sonic 复现框架。

其中：

- `mpc/` 目录实现核心 MPC 协议与安全神经网络算子；
- `models/` 目录实现 PyTorch 明文参考模型和 Sonic 论文模型结构；
- `crypto/` 目录实现 OT 和 OT Extension 相关功能；
- `net/` 目录实现 socket 通信和性能统计；
- `run/` 目录包含各类测试脚本、benchmark 脚本和一键测试入口；
- `data/` 目录包含用于测试的小型数据集。

---

## 2. 当前复现内容

本项目目前主要完成了以下内容。

### 2.1 秘密分享基础

已实现：

- Arithmetic Secret Sharing
- Boolean Sharing
- Arithmetic share 重构
- Boolean share 重构
- `Z_(2^32)` 环上的整数计算
- 负数二补码表示

对应文件主要包括：
- `mpc/share.py` --还原论文中的算术秘密分享机制，将明文 x 拆成 x0 和 x1，使 x = x0 + x1 mod 2^32
- `mpc/fixed_point.py`  --还原论文中的定点数编码思想，将浮点神经网络参数和输入映射到整数环中进行安全计算

## 2.2 Beaver Triple 与安全乘法

已实现：

- Arithmetic Beaver triple
- Boolean Beaver triple
- triple pool 缓存机制
- trusted dealer 版本 triple 生成
- OT Extension 版本 triple 生成
- 基于 Beaver triple 的安全乘法
- 安全 bit AND

对应文件主要包括：

- `mpc/mul.py` --还原论文中的 Beaver Triple 安全乘法协议，是 SFC、SCONV、SBN、SReLU 等安全算子的基础
- `mpc/triple.py` --还原论文中 Beaver Triple 的统一获取接口，为安全乘法和安全 bit-and 提供离线随机材料
- `mpc/triple_pool.py` --还原论文离线阶段的 triple pool 思想，按 tensor shape 管理 arithmetic triple 和 boolean triple
- `mpc/triple_dealer.py` --还原论文中离线预处理阶段提前生成数据无关 triple 的设定，用于 M1/M2/C1/C2 大模型功能测试
- `mpc/arith_triple_ot.py` --还原论文中通过 OT Extension 生成 arithmetic Beaver triple 的过程，生成满足 c = a * b 的算术三元组
- `mpc/bit_triple_ot.py` --还原论文中通过 OT Extension 生成 boolean Beaver triple 的过程，生成满足 c = a AND b 的布尔三元组
- `mpc/bit_and.py` --还原论文中的布尔分享安全 AND 运算，用于 MSB 提取、安全比较、SReLU、MaxPool 和 Argmax
- `crypto/ot.py` --还原论文离线阶段所需的基础 OT（Oblivious Transfer，不经意传输）思想，用于后续 OT Extension 和随机材料生成
- `crypto/ot_extension.py` --还原论文中 OT Extension 的基本流程，通过少量基础 OT 扩展出大量 OT，降低离线阶段随机材料生成成本
- `crypto/ot_extension_ring.py` --还原论文中面向整数环 Z2^32 的 OT Extension，用于生成 arithmetic Beaver triple 所需的环上随机材料

---

## 2.3 Fixed-point 编码与安全截断

由于 MPC 协议主要在整数环上进行计算，神经网络中的浮点数需要先编码为 fixed-point 整数。

已实现：

- 浮点数到 fixed-point ring element 的编码
- ring element 到浮点数的解码
- fixed-point scale 跟踪
- 安全截断 `secure_trunc`
- 乘法后从 `2^(2f)` 恢复到 `2^f`

对应文件主要包括：

- `mpc/fixed_point.py` --还原论文中的定点数编码思想，将浮点神经网络参数和输入映射到整数环中进行安全计算
- `mpc/trunc.py` --还原论文中的 Secure Truncation 安全截断算法，用于定点乘法后将 scale 从 2^(2f) 恢复到 2^f

---

## 2.4 A2B / B2A 与安全比较

已实现：

- Arithmetic share 到 Boolean bit share 的转换
- Boolean share 到 Arithmetic share 的转换
- secure MSB 功能级实现
- `x >= 0` 安全比较
- 安全 ReLU / SReLU 的比较基础

对应文件主要包括：

- `mpc/a2b.py` --还原论文中的 A2B 算法，即 Arithmetic Share 到 Boolean Share 的转换，用于将算术秘密分享转入比较和 MSB 逻辑
- `mpc/b2a_secure.py` --还原论文中的 B2A 算法，即 Boolean Share 到 Arithmetic Share 的转换，用于将比较结果转换为可参与算术计算的 gate
- `mpc/compare.py` --还原论文中的安全比较和 MSB 提取算法，用于判断数值正负和大小关系，是 SReLU、SMP、Secure Argmax 的核心依赖
- `mpc/relu.py` --实现普通 ReLU 的参考逻辑，用于和安全 ReLU 的结果进行对照
- `mpc/srelu.py` --还原论文中的 SReLU 安全激活函数，通过安全比较得到符号 gate，再计算 gate * x，实现 max(0,x)

说明：

当前版本实现了 secure MSB 的功能级版本，能够支持 SReLU、SMP、Argmax 等操作。但论文中使用的 PPA 低通信轮数 MSB 优化尚未完整复现。

---

## 2.5 Sonic 核心安全层函数

本项目复现了论文中的主要安全层函数：

| 论文安全层 | 含义 | 当前实现 |
| SFC | Secure Fully Connected Layer | 已实现 |
| SCONV | Secure Convolution Layer | 已实现 |
| SBN | Secure Batch Normalization | 已实现 |
| SReLU | Secure ReLU | 已实现 |
| SMP | Secure Max Pooling | 已实现 |
| Secure Argmax | 安全分类输出选择 | 已实现 |

对应文件主要包括：

- `mpc/linear.py` --实现明文线性层参考计算，用于和安全全连接层结果进行对照
- `mpc/linear_secret.py` --还原论文中的 SFC 安全全连接层，实现秘密分享状态下的 Y = XW + b
- `mpc/conv2d_secret.py` --实现基础安全卷积，用于前期验证秘密分享状态下卷积计算的正确性
- `mpc/conv2d_fixed.py` --还原论文中的 SCONV 安全卷积层，支持 fixed-point 定点数计算，是 M2/C1/C2 安全 CNN 的核心模块
- `mpc/sbn.py` --还原论文中的 SBN 安全批归一化，将 BatchNorm 转换为 eps1 * x + eps2 的安全线性计算
- `mpc/srelu.py` --还原论文中的 SReLU 安全激活函数，通过安全比较得到符号 gate，再计算 gate * x，实现 max(0,x)
- `mpc/maxpool2d_secret.py` --还原论文中的 SMP 安全最大池化，通过安全比较在池化窗口内选择最大值
- `mpc/avgpool2d_secret.py` --实现安全平均池化，作为安全池化算子的补充，用于扩展 CNN 安全推理能力
- `mpc/argmax_secure.py` --还原分类输出阶段的 Secure Argmax，通过安全比较选出最大 logit 及其类别编号

---

## 2.6 Sonic 模型结构复现

项目按照论文中的模型结构，复现了 M1、M2、C1、C2 四类模型。

| 模型 | 数据集 | 类型 | 当前复现情况 |
| M1 | MNIST | 全连接网络 | 已实现 fixed-point MPC 推理流程 |
| M2 | MNIST | CNN | 已实现 fixed-point MPC 推理流程 |
| C1 | CIFAR-10 | CNN | 已实现 fixed-point MPC 推理流程框架 |
| C2 | CIFAR-10 | CNN | 已实现 fixed-point MPC 推理流程框架 |

对应文件主要包括：

- `models/sonic_model_specs.py` --sonic_model_specs.py：还原论文 6.6 Model Architectures 中 M1、M2、C1、C2 四类模型结构，定义每层 Conv、FC、BN、ReLU、MaxPool 的维度与顺序
- `models/sonic_pytorch_models.py` --根据 sonic_model_specs.py 自动构建 PyTorch 明文模型，并导出 Conv、FC、BatchNorm 参数供 MPC 安全推理使用
- `mpc/secure_m1_fixed.py` --还原论文 M1 模型的 MPC 推理结构，实现 Flatten → FC → SBN → SReLU → FC → SBN → SReLU → FC → SBN
- `mpc/secure_m2_fixed.py` --还原论文 M2 模型的 MPC 推理结构，实现 Conv → SBN → SMP → SReLU → Conv → SBN → SMP → SReLU → FC → SBN → SReLU → FC → SBN
- `mpc/secure_c1_fixed.py` --还原论文 C1 模型的 MPC 推理结构，实现 7 个安全卷积块、2 个 MaxPool 和最终 FC/SBN
- `mpc/secure_c2_fixed.py` --还原论文 C2 模型的 MPC 推理结构，实现 9 个安全卷积块、3 个 MaxPool 和最终 FC/SBN

---

## 2.7 PyTorch 明文参考模型

PyTorch 在本项目中主要用于构建明文参考模型，不负责安全计算。

它的作用包括：

- 构建 M1/M2/C1/C2 明文模型；
- 检查每层输入输出 shape；
- 导出 Conv、FC、BN 参数；
- 与 MPC fixed-point 推理结果进行对照。

对应文件主要包括：

- `models/simple_mlp.py` --构建简单明文 MLP 模型，用于前期验证 PyTorch 参数导出与 MPC 安全推理结果是否一致
- `models/simple_cnn.py` --构建简单明文 CNN 模型，用于验证卷积层、激活层、Flatten、全连接层与 MPC CNN 推理的对应关系
- `models/simple_cnn_classifier.py` --构建简单 CNN 分类模型，用于 toy dataset 上的分类 accuracy、logits 对齐和 Secure Argmax 测试
- `models/sonic_model_specs.py` --sonic_model_specs.py：还原论文 6.6 Model Architectures 中 M1、M2、C1、C2 四类模型结构，定义每层 Conv、FC、BN、ReLU、MaxPool 的维度与顺序
- `models/sonic_pytorch_models.py` --根据 sonic_model_specs.py 自动构建 PyTorch 明文模型，并导出 Conv、FC、BatchNorm 参数供 MPC 安全推理使用

---

## 2.8 通信与性能统计

项目使用 socket 模拟两方服务器通信，并实现了基础性能统计工具。

已统计内容包括：

- 发送字节数
- 接收字节数
- 消息数量
- `secure_mul` 调用次数
- `bit_and` 调用次数
- A2B / B2A 调用次数
- truncation 调用次数
- ReLU 调用次数
- offline triple 数量
- online / offline 时间

对应文件主要包括：

- `net/socket_utils.py` --还原论文两方安全计算中的通信机制，封装 Party0 和 Party1 之间的数据发送与接收
- `net/profiler.py` --还原论文实验统计部分的通信量、消息数、运行时间统计功能，用于记录 bytes_sent、bytes_recv、total_time、secure_mul_calls 等指标

---

## 3. 项目目录结构

项目主要目录如下：

- `crypto/`  
  OT、OT Extension 以及 ring 上 OT Extension 的实现。

- `data/`  
  存放测试用 toy 数据集，例如 `toy_cnn_dataset.npz`。

- `models/`  
  存放 PyTorch 明文参考模型和 Sonic 论文模型结构定义。

- `mpc/`  
  存放核心 MPC 协议、安全算子和 fixed-point MPC 推理流程。

- `net/`  
  存放 socket 通信工具和 profiler 性能统计工具。

- `run/`  
  存放测试脚本、benchmark 脚本、数据生成脚本和一键测试脚本。

- `data_loader.py`  
  用于加载 `.npz` 测试数据集。

---

## 4. 环境要求

建议使用 Python 3.8 或以上版本。

主要依赖：

- `numpy`
- `torch`

可以使用以下命令安装依赖：

```bash
pip install numpy torch

## 5. 运行方式

### 5.1 生成测试数据集

如果 `data/toy_cnn_dataset.npz` 不存在，可以运行：

```bash
python run/make_toy_dataset.py
```

---

### 5.2 打印 Sonic 模型结构

可以查看 M1、M2、C1、C2 的模型结构：

```bash
python run/print_sonic_model_specs.py
```

---

### 5.3 运行全部测试

本项目提供了一键测试脚本：

```bash
python run/run_all_tests.py
```

该脚本会依次运行多个 `server0/server1` 测试文件，用于验证基础协议、安全算子、fixed-point 推理以及 M1/M2/C1/C2 模型流程。

---

### 5.4 单独运行某个双服务器测试

部分测试需要分别启动 `server0` 和 `server1`。

例如测试 M1 MPC 推理：

终端 1：

```bash
python run/test_m1_mpc_server0.py
```

终端 2：

```bash
python run/test_m1_mpc_server1.py
```

其他测试脚本也采用类似方式，例如测试 SReLU：

终端 1：

```bash
python run/test_srelu_server0.py
```

终端 2：

```bash
python run/test_srelu_server1.py
```

---

## 6. 测试内容说明

`run/` 目录中包含大量测试脚本，主要分为以下几类。

---

### 6.1 基础协议测试

用于验证 OT、OT Extension、Beaver triple、offline/online triple pool 等底层协议。

示例：

- `test_ot_extension_server0.py`
- `test_ot_extension_server1.py`
- `test_ot_bit_triple_server0.py`
- `test_ot_bit_triple_server1.py`
- `test_ot_arith_triple_server0.py`
- `test_ot_arith_triple_server1.py`
- `test_offline_online_mlp_server0.py`
- `test_offline_online_mlp_server1.py`

---

### 6.2 安全算子测试

用于验证 SReLU、SBN、secure Argmax、MaxPool、AvgPool 等安全算子。

示例：

- `test_relu_server0.py`
- `test_relu_server1.py`
- `test_srelu_server0.py`
- `test_srelu_server1.py`
- `test_sbn_server0.py`
- `test_sbn_server1.py`
- `test_secure_argmax_server0.py`
- `test_secure_argmax_server1.py`
- `test_maxpool2d_server0.py`
- `test_maxpool2d_server1.py`
- `test_avgpool2d_server0.py`
- `test_avgpool2d_server1.py`

---

### 6.3 线性层和卷积层测试

用于验证 SFC 和 SCONV。

示例：

- `test_linear_server0.py`
- `test_linear_server1.py`
- `test_secret_linear_server0.py`
- `test_secret_linear_server1.py`
- `test_secret_conv2d_server0.py`
- `test_secret_conv2d_server1.py`
- `test_fixed_conv2d_server0.py`
- `test_fixed_conv2d_server1.py`

---

### 6.4 MLP / CNN 推理测试

用于验证从简单 MLP、小型 CNN 到 fixed-point CNN 的安全推理流程。

示例：

- `test_mlp_server0.py`
- `test_mlp_server1.py`
- `test_secret_mlp_server0.py`
- `test_secret_mlp_server1.py`
- `test_fixed_secret_mlp_server0.py`
- `test_fixed_secret_mlp_server1.py`
- `test_fixed_trunc_mlp_server0.py`
- `test_fixed_trunc_mlp_server1.py`
- `test_fixed_cnn_server0.py`
- `test_fixed_cnn_server1.py`
- `test_fixed_cnn_pool_server0.py`
- `test_fixed_cnn_pool_server1.py`
- `test_fixed_cnn_sbn_server0.py`
- `test_fixed_cnn_sbn_server1.py`
- `test_fixed_cnn_sbn_pool_opt_server0.py`
- `test_fixed_cnn_sbn_pool_opt_server1.py`

---

### 6.5 Sonic 论文模型测试

用于验证论文中的 M1、M2、C1、C2 模型推理流程。

示例：

- `test_m1_mpc_server0.py`
- `test_m1_mpc_server1.py`
- `test_m2_mpc_server0.py`
- `test_m2_mpc_server1.py`
- `test_c1_mpc_server0.py`
- `test_c1_mpc_server1.py`
- `test_c2_mpc_server0.py`
- `test_c2_mpc_server1.py`

---

## 7. 当前完成程度

目前本项目已经完成 Sonic 论文核心安全推理流程的功能级复现，主要包括：

- 底层秘密分享协议；
- Beaver triple 安全乘法；
- fixed-point 编码和安全截断；
- A2B / B2A 转换；
- 安全比较与 SReLU；
- SBN、SFC、SCONV、SMP 等安全层；
- M1/M2/C1/C2 的 fixed-point MPC 推理框架；
- PyTorch 明文参考模型与参数导出；
- socket 双方通信与性能统计；
- 多组模块测试和一键测试脚本。

---

## 8. 尚未完全复现的部分

当前项目仍然存在一些不足，主要包括：

1. 论文中基于 PPA 的低通信轮数 secure MSB 优化尚未完整实现；
2. 论文中的完整 MNIST / CIFAR-10 正式准确率表尚未完全复现；
3. 与 MiniONN、Gazelle、EzPC、XONN 等工作的完整性能对比实验尚未复现；
4. 论文 Java 原型中的精确时间和带宽数据尚未完全对齐；
5. 论文第 5 节的形式化安全证明没有进行代码化复现；
6. 当前版本主要用于功能验证和流程复现，性能仍有进一步优化空间。

---

## 9. 复现总结

本项目目前已经实现了 Sonic 论文中大部分核心安全算子和模型推理流程，能够从底层秘密分享协议逐步支撑到 SFC、SCONV、SBN、SReLU、SMP，并进一步组合成 M1、M2、C1、C2 等模型的 fixed-point MPC 推理框架。

整体而言，当前版本属于 **核心协议与模型推理流程的功能级复现**。后续工作将继续围绕 PPA 版 secure MSB、完整准确率实验、性能对比实验和代码性能优化展开。

---

## 10. 说明

本项目仅用于论文学习与复现实验，重点在于理解 Sonic 中轻量级秘密分享安全推理的核心思想和实现流程。

如代码、报告或论文理解中仍存在不足，欢迎批评指正。