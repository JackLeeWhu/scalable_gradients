# Li et al. (2020) 第 5、7 节独立复现

本目录只放徐敬翔负责的独立工作，不改动组员原有文件。目标是检查论文后半部分的实验能复现到什么程度，而不是把“外形相似”的图当成原论文数值复现。论文正文：[PMLR](https://proceedings.mlr.press/v108/li20i.html)；完整附录：[PMLR supplement](https://proceedings.mlr.press/v108/li20i/li20i-supp.pdf)。

第 5 节提出 latent SDE、共享 diffusion 的先验和后验，以及由观测似然和路径 KL 构成的 ELBO；它本身不是独立的数据实验。第 7 节才用数值 SDE、合成 GBM/Lorenz 和真实 mocap 数据分别检验梯度方法及 latent SDE。正文 Figure 3 是模型结构示意图；正文 Figure 4–5、补充 Figure 8–9 和 Table 2 是需要检查的结果。

## 图表和数据来源

| 原文目标 | 原文数据 | 本目录做法 | 能否视为严格复现 |
|---|---|---|---|
| §7.1 / Figure 4(a) | 附录 9.11 的解析解测试 SDE；原始随机抽样、标量损失未发布 | `numerical_study.py` 使用 Example 2、10 维参数和明确写出的终点和损失，比较 Milstein 普通反传、Milstein adjoint 与解析梯度 | 否；可复现误差随步长下降的机制，不能复原原图数值 |
| §7.1 / Figure 4(b) | 同一 Example 2，改变 adaptive solver 的绝对误差容忍度 | `adaptive_study.py` 用 SRK 正向、Milstein adjoint 反向，统计函数求值数和解析梯度误差 | 否；原文没有给出所有求解器选项与随机数 |
| §7.1 / Figure 4(c) | 比较计算时间与梯度误差 | `efficiency_study.py` 在同一台 CPU 上比较 Euler backprop、Milstein backprop、Milstein adjoint；先预热后计时 | 否；硬件、实现、步长与论文不同，只能比较本机取舍 |
| 补充 Figure 7(a–f) | 附录 9.11 Example 1 和 3 的解析解 SDE | `supplement_examples_1_3.py` 重做固定步长、adaptive SRK/NFE 和 CPU 时间六个面板 | 否；原文随机种子、损失、完整容差表未发布；Example 1 印刷解析解与方程不一致 |
| §7.2 / Figure 5、补充 Figure 8 | 按附录 9.13.2 生成的 1024 条随机 Lorenz 轨迹 | `generate_data.py` 重新抽样；`train_synthetic_latent_sde.py lorenz` 训练并画数据、重建、先验采样 | 独立复现实验流程；原始训练种子、权重未发布 |
| §7.2 / 补充 Figure 9 | 按附录 9.13.1 生成的 1024 条 GBM 轨迹 | `generate_data.py` 重新抽样；`train_synthetic_latent_sde.py gbm` 训练并画数据、重建、先验采样 | 独立复现实验流程；原始训练种子、权重未发布 |
| §7.3 / Table 2 | 预处理 CMU subject 35 walking，16/3/4 条，300 帧 × 50 维 | 从 ODE2VAE 项目公开的 [Google Drive 数据目录](https://drive.google.com/drive/folders/1c0UMSqlvZRORmCNN_qVdiqu2n8sKcwnh) 获取 `mocap35.mat`；按补充 Figure 10 训练 400 次并以 50 次采样评估 297 帧 | 数据和评价长度复现；测试 MSE 为 32.69，未达到论文的 4.03 |

`data/mocap35.mat` 是第三方预处理数据，保留在本地，不提交进 GitHub。可从 [ODE2VAE 作者公开的文件](https://drive.google.com/file/d/1adqzHEXaGDVObSKbJgT5HBdoZkeMZ8S3/view) 下载到 `data/mocap35.mat`。数据 SHA-256、形状及实验参数在 `outputs/data_manifest.json`。

## 与论文一致及不一致的地方

- 两个合成数据集的路径数、观测时间、GBM 参数、Lorenz 参数和观测噪声按附录 9.13 设置；由于作者没有发布具体随机抽样，GBM 固定新种子 5600，Lorenz 固定新种子 5601。Lorenz 原文没有给出生成数据的数值积分步长，这里用 Euler–Maruyama 的 0.005 内部步长。
- 合成模型使用一个隐藏层 100 单元的 prior/posterior drift 和 diffusion、4 维 latent state、倒序 GRU encoder、线性 decoder；训练用 TorchSDE stochastic adjoint、Euler 步长 0.01、Adam 初始学习率 0.01、每步衰减 0.999、前 50 次线性 KL anneal。当前用 64 条路径的 minibatch；论文未给出确切 batch size 或训练总轮数。
- 按附录 9.13 的“同 GBM 架构”设置，Lorenz 主结果采用 1 维 encoder context，放在 `outputs/lorenz_context1/`；另保留 3 维 context 的探索性结果在根 `outputs/`，其重建较好但架构偏离附录。GBM 图的后验区间由 512 次采样得到，与补充 Figure 9 的采样数一致。
- 动作数据模型按附录 Figure 10 使用 6 维 latent state、前三帧 encoder、prior/posterior drift 和逐维 diffusion。我们的实现有 11,845 个参数，论文报告 11,605 个，因此属于接近架构而非逐参数相同的重建。**差额 240 恰好等于图中 8 个以 `t` 为额外输入的 FC-30 网络（先验漂移 1 个、后验漂移 1 个、逐维扩散 6 个）各多 30 个权重。**删去全部这些显式时间输入就会得到 11,605，但会与 Figure 10 的箭头不一致；作者没有发布该模型原始代码，无法判断是参数统计还是图示有误。观测时间单位使用 ODE2VAE 原始预处理的 0.1/帧，solver 步长 0.02。
- Mocap 训练采用最后一层 drift 为零的初始化，以避免 300 帧上起始路径 KL 溢出；这是额外的稳定性选择，原文未说明。选择 KL 系数 0.1 并根据验证集保存最优 checkpoint；原文搜索了多个 KL 系数和是否退火，但没有发布最终超参数。论文 Table 2 的 95% 区间没有交代统计单位，因此本目录的区间明确只量化固定模型下 50 次预测采样的 Monte Carlo 误差。
- 论文第 7 节明确说，小规模实验用的是**存储 Brownian 查询值**的实现；虚拟布朗树并非这些实验的必要条件。
- 附录 9.11 Example 2 的漂移印为 `-(p²)² sin(X) cos³(X)`，但其解析解 `atan(pW+tan(x0))` 经 Itô 公式推出的是 `-p² sin(X) cos³(X)`。`numerical_study.py` 使用与解析解自洽的表达式；该差异必须在报告中注明。
- 同页 Example 1 也有类似不一致：SDE 写 `αX dt + βX dW`，解析解却把 `β` 放在时间增长项、把 `α` 放在布朗项。这个目录的解析梯度实验因此选 Example 2 并明确校正式子。
- 补充 Figure 7 的 Example 1 则按印出的 SDE 使用正确解 `X_T=X_0 exp[(α−β²/2)T+βW_T]`；Example 3 的方程与解析解自洽。`supplement_examples_1_3.py` 也写明了我们自行选择的标量损失、随机种子和步长。

## 实际结果与边界

正文 Figure 4 和补充 Figure 7 的固定步长实验都显示数值梯度误差随步长缩小而下降；adaptive SRK 正向加 adaptive Milstein adjoint 反向的总函数求值数与误差呈预期的计算量交换。具体数值见 `RESULTS_FOR_TEAM_CN.md` 和对应 CSV。正文 Figure 5、补充 Figure 8、9 的模型图已由重新生成的数据和独立训练得到，能讨论后验重建与先验样本的定性行为。

**Table 2 未复现作者性能**：同一 CMU 文件、300 帧、400 次训练、50 次测试采样下，我们的最佳验证 MSE 为 `31.27`。固定新种子下，编码前三帧后使用后验漂移预测的测试 MSE 为 `32.691 ± 0.011`，改用生成模型的先验漂移为 `32.881 ± 0.019`；论文为 `4.03 ± 0.20`。本机零预测基线约 `33.67`，说明当前模型的预测只略好于零预测。两个 95% t 区间只描述固定模型下 50 次预测样本的 Monte Carlo 波动，不能作为与论文区间同口径的复现。论文没有公开测试时选择哪一个漂移场的实现细节，故同时列出两种。另一个 KL 系数 `0.001` 的 120 次迭代试跑验证 MSE 仍约 `31.86`。模型参数数目、初始化和原论文未发布的训练细节均限制了严格复现。

检索到 [另一个公开复现尝试](https://github.com/nkiyohara/latent-sde-mocap) 报告约 `9.0`，但其 `train_mocap.py` 将整条序列做成每个时刻的三帧滑动窗口，测试时编码器仍读取后续帧。这个信息量与“仅给前三帧，预测其余 297 帧”的 Table 2 任务不同，因此没有采用它的分数或代码作为严格复现依据。

## 运行

在本目录下：

```bash
python3 -m pip install -r requirements.txt
python3 generate_data.py
python3 numerical_study.py
python3 adaptive_study.py
python3 efficiency_study.py
python3 supplement_examples_1_3.py
python3 train_synthetic_latent_sde.py gbm --iterations 300 --batch-size 64 --adjoint --device cpu
python3 render_gbm_figure.py
# Lorenz 的主结果要先将生成的 lorenz_1024.npz 放入 outputs/lorenz_context1/
python3 train_synthetic_latent_sde.py lorenz --iterations 300 --batch-size 64 --context-dim 1 --adjoint --device cpu --output-dir outputs/lorenz_context1
python3 render_extended_lorenz.py --output-dir outputs/lorenz_context1
python3 evaluate_synthetic.py gbm
python3 evaluate_synthetic.py lorenz --output-dir outputs/lorenz_context1
# 完整 mocap 设置：300 帧、最多 400 轮、测试时 50 次采样
python3 train_mocap_latent_sde.py --frames 300 --iterations 400 --dt 0.02 --eval-samples 50 --adjoint --device cpu --output-dir outputs/mocap_full_400
python3 evaluate_mocap_checkpoint.py --output-dir outputs/mocap_full_400 --samples 50 --drift posterior
python3 evaluate_mocap_checkpoint.py --output-dir outputs/mocap_full_400 --samples 50 --drift prior
# 短试跑仅用于检查程序是否可执行，不能与 Table 2 比较
python3 train_mocap_latent_sde.py --frames 50 --iterations 50 --eval-samples 10 --adjoint --device cpu
```

所有生成的数据、图、训练历史和模型检查点放在 `outputs/`。检查点与第三方数据默认不提交 GitHub。

本次运行环境：macOS arm64 CPU、Python 3.14.2、PyTorch 2.15.0、TorchSDE 0.2.6、NumPy 2.5.5、SciPy 1.18.1。CPU 时间只适用于这台机器和本脚本设置。

## 本组整合时的判断标准

1. 第 2–4 节和本目录的第 5、7 节应分别写明自己验证了什么。布朗树重放、参数梯度、latent SDE 训练是不同实验，不能用某一个图替代另外一个结论。
2. 对每张图同时交代数据来源、随机种子、模型、积分器、训练轮数和实际是否用 adjoint。
3. 只有完整 300 帧实验、50 次预测采样及合理的验证集模型选择，才报告与论文 Table 2 的 4.03 的对比；短试跑只验证流程。
