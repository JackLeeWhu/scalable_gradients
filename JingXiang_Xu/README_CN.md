# JingXiang Xu 的部分：TorchSDE adjoint 与梯度验证

根据 Team 7 的 proposal，我负责“Implement the TorchSDE adjoint comparison and help verify the gradients”。本文件夹是一份独立交付，可直接交给组员合并；没有修改父目录中的程序、报告或数据。

## 文件

- `adjoint_compare.py`：实际调用 `torchsde.sdeint_adjoint`，与 `torchsde.sdeint` 的普通自动微分比较。
- `outputs/gradient_comparison.csv`：各步数的损失、梯度及误差，可供绘图。
- `outputs/results.json`：完整设置、原始数值和数值差分检查。
- `REPORT_SECTION.md`：可供小组报告整合的英文方法、结果和讨论段落。
- `requirements.txt`：运行依赖。

## 运行

在此文件夹执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python adjoint_compare.py
```

默认用 CPU、256 条路径、float64、随机种子 5600，分别测试 16、32、64、128、256 个 Euler 步。也可指定 `--paths 128 --steps 16 32 64` 或在支持 CUDA 的机器上加 `--device cuda`。输出只写入本文件夹的 `outputs/`。

## 验证逻辑

模型为 Itô GBM，`dX = μX dt + σX dW`，损失为 `mean((X_T - 1.30)^2)`。两个 TorchSDE 求解器使用同一个 `BrownianInterval`、同一组参数、同样的前向 Euler 步长。反向 adjoint 求解器也明确使用 Euler。另有两个独立参照：

1. 对离散 Euler 更新 `X_{k+1}=X_k(1+μΔt+σΔW_k)` 手推并逐步计算参数导数。这是检查普通反传的正确参照。
2. 用连续 GBM 的解析解 `X_T=X_0 exp((μ-σ²/2)T+σW_T)` 及其解析导数，检查步长趋小时是否接近真实连续模型。

在最细网格上，程序还对 TorchSDE 前向损失做固定 Brownian 路径的中心差分。脚本若发现前向路径、离散导数或差分检查不符，会直接报错。

## 结果摘要

本机运行环境：PyTorch 2.11.0、TorchSDE 0.2.6、CPU。前向路径的最大差异小于 `4e-14`，普通反传与离散 Euler 导数的最大差异小于 `3e-14`，256 步中心差分与普通反传的最大差异小于 `1e-9`。

| 步数 | 普通反传 dL/dμ | Adjoint dL/dμ | 普通反传 dL/dσ | Adjoint dL/dσ | 两参数最大差异 |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.225800 | 0.250241 | 1.572459 | 1.579866 | 0.024441 |
| 32 | 0.254611 | 0.262918 | 1.601356 | 1.614135 | 0.012778 |
| 64 | 0.274294 | 0.278929 | 1.634255 | 1.648450 | 0.014195 |
| 128 | 0.285867 | 0.287838 | 1.661045 | 1.663945 | 0.002900 |
| 256 | 0.293540 | 0.295517 | 1.696544 | 1.694305 | 0.002239 |

同一路径的连续 GBM 解析梯度为 `dL/dμ=0.297170`、`dL/dσ=1.701558`。Adjoint 梯度并不等于离散 Euler 损失的精确梯度，因为其反向计算是另一次数值 SDE 求解。此样本中最大差异总体随步数增加而缩小，但 32 到 64 步有局部增加，不能声称每次细化都会单调改善。

父目录现有 `main.py` 使用 **GBM 精确指数更新** 和另一套随机数生成方式；此脚本使用 **TorchSDE Euler 更新**。因此两份脚本的梯度数值不能直接按行比较。这里的公平比较是在本脚本内部使用同一条 Brownian 路径进行的。

## 参考

- Li, X. et al. (2020), “Scalable Gradients for Stochastic Differential Equations,” Section 3. [PMLR 论文](https://proceedings.mlr.press/v108/li20i.html)。
- [TorchSDE 官方文档](https://github.com/google-research/torchsde/blob/master/DOCUMENTATION.md)，特别是 Adjoints 和 Brownian motion 两节。
