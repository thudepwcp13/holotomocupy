# Task NFP-NN-01：基于 SelfPhish 思路的多位置共享 Probe / 全局 Sample 神经网络 NFP 相位恢复

**任务状态：待用户确认，未获明确批准前禁止启动实现或实验。**
**任务创建日期：2026-07-23**

---

## 1. 任务目标

在 `holotomocupy` 现有 near-field ptychography（NFP）流程基础上，实现一套基于 PyTorch 自动微分和 SelfPhish / deep-image-prior 思路的神经网络优化方法，用神经网络参数更新替代当前 `RecNFP.BH()` 中的 BH/CG 迭代。

必须保留现有 NFP 的物理前向模型、数据预处理、几何参数、扫描位置定义和输出约定，但神经网络优化不能像当前 SelfPhish 那样对每张 hologram 独立恢复并立即结束。新方法必须联合使用同一次 NFP 扫描的全部图像：

```text
一个共享 probe
+
一个全局 sample/object canvas
+
N 个扫描位置
+
N 张 detector NFP 图像
```

真实数据验收必须覆盖三组数据：

1. DanMAX Siemens Star（无 coded aperture），100 张扫描图像；
2. DanMAX Si spheres（含 coded aperture），100 张扫描图像；
3. ID16B NFP，按现有 launcher 默认选择 `sample_scan_ids=1:256:4`，共 64 个扫描位置。

数据接口必须可配置，不能把文件路径、帧数、几何参数或 HDF5 key 写死。DanMAX 和 ID16B 的文件组织与 HDF5 key 布局不同，必须复用各自现有、已验证的 reader/预处理逻辑。

本任务属于 dataset-specific physics-informed optimization，而不是有监督训练：

- 不要求 ground-truth probe 或 sample；
- 不要求预训练数据集；
- 每次实验针对当前一组 NFP 数据优化网络权重；
- 每个 epoch 必须完整覆盖所选的全部 NFP 扫描帧；
- 网络先验和物理一致性共同约束恢复结果。

---

## 2. 核心定义：不能退化成逐帧 SelfPhish

必须满足以下两个硬约束。

### 2.1 Probe 一致性

所有扫描位置必须使用同一个复 probe：

```text
P(r) = A_probe(r) * exp(i * phi_probe(r))
```

不允许：

```text
P_0, P_1, ..., P_(N-1)
```

分别由每张 detector 图独立恢复。

默认模式必须是严格共享 probe。将来如需研究 probe drift，可以另加显式、默认关闭的低维 drift 模型，但不能影响本任务的基础实现与验收。

### 2.2 Sample 重叠区域一致性

必须恢复一个全局 sample canvas：

```text
O_global(r) = delta(r) + i * beta(r)
```

每张扫描图像对应的局部 sample 只能由当前扫描位置从同一个全局 canvas 中取出：

```text
O_j = S_pos_j(O_global)
```

因此不同扫描帧覆盖到同一个全局 sample pixel 时，必须天然引用同一个变量。

首选实现是“单一全局 canvas + 可微 shift/crop”，使 overlap consistency 由参数化方式严格保证。若因显存限制必须采用 patch network，则必须维护显式 global consensus canvas，并加入 patch-to-global overlap consistency loss；严禁为每个扫描位置建立独立 sample 网络或互不关联的 sample 输出。

---

## 3. 目标物理模型

现有 `RecNFP` 的模型为：

```text
T_j = exp(i * S_pos_j(O_global))
Psi_j = P_shared * T_j
U_j = D(Psi_j)
A_pred_j = abs(U_j)
```

其中：

```text
O_global = proj_delta + i * proj_beta
exp(i * O_global) = exp(-proj_beta) * exp(i * proj_delta)
```

样品数据项：

```text
L_sample =
    mean over selected frames and valid detector pixels
    (A_pred_j - A_measured_j)^2
```

flat-field 分支：

```text
U_flat = D(P_shared)
A_flat_pred = abs(U_flat)
L_flat = mean((A_flat_pred - A_flat_target)^2)
```

总损失至少包括：

```text
L_total =
    w_sample   * L_sample
  + w_flat     * L_flat
  + w_probe    * R_probe
  + w_object   * R_object
  + w_position * R_position
```

可以配置 amplitude L1、amplitude MSE、intensity loss 或 Poisson likelihood，但默认和当前 NFP 一致，使用 amplitude-domain MSE，以便与迭代基线直接比较。

---

## 4. 需要复用和审计的代码

### 4.1 holotomocupy

重点文件：

```text
experimental/DanMAX_nano/step0.py
experimental/DanMAX_nano/step0_v2.py
experimental/DanMAX_nano/run_step0.sh
experimental/ID16B_NFP/step0.py
experimental/ID16B_NFP/step0_v2.py
src/holotomocupy/rec_nfp_mpi.py
src/holotomocupy/propagation.py
src/holotomocupy/shift.py
src/holotomocupy/writer.py
```

必须复用或严格保持一致的内容：

```text
dark / flat / sample 读取
frame ID 选择
dark subtraction
global intensity normalization
flat amplitude target 定义
energy / wavelength
z1 / focus-to-detector distance
Fresnel effective distance
sample-plane voxel size
motor position到pixel position转换
object canvas size nobj
valid detector mask
最终 HDF5 dataset 命名
```

### 4.2 PhaseNet

PhaseNet 仓库：

```text
/dtu/3d-imaging-center/projects/2022_DANFIX_08_BioComFert/raw_data_extern/Chengpeng/PycharmProjects/PhaseNet
```

重点参考：

```text
scripts/run_selfphish_hereon_h5.py
src/physics/phase_recon/selfphish_hereon_adapter.py
src/physics/phase_recon/selfphish_hereon/
src/models/selfphish.py
src/models/sample_unet.py
src/models/probe_parametrization.py
src/trainer/sample_unet_sim_shared_probe.py
src/physics/propagation.py
src/physics/propagator.py
```

必须先明确哪些组件可以直接复用，哪些仅可参考：

- SelfPhish 的 dataset-specific network optimization；
- phase / absorbance 输出约束；
- shared learnable probe；
- network checkpoint 和 optimizer resume；
- physics loss；
- probe smoothness / gauge regularization；
- safe/resumable HDF5 输出思路。

不能直接调用“单张图运行一次 SelfPhish”的外层接口来循环各扫描位置，因为那样不会得到共享 probe 和重叠一致的全局 sample。

PhaseNet 当前可能存在用户未提交修改。实施时先运行：

```bash
git -C /path/to/PhaseNet status --short
```

在没有用户额外授权时：

- PhaseNet 作为只读参考或只读 import 来源；
- 不得覆盖、清理、stash 或提交 PhaseNet 的现有修改；
- 如必须修改 PhaseNet，先停止并向用户说明修改范围；
- 默认以只读 import/组合方式复用 PhaseNet，主要新增内容只应是 `holotomocupy` 中的薄适配层、配置和必要测试。

---

## 5. 最小整合原则

本任务不是重写 holotomocupy 或 PhaseNet。实施前必须先做组件映射，优先直接复用：

```text
holotomocupy:
  DanMAX / ID16B reader、dark/flat 预处理、几何、位置转换、mask、输出约定

PhaseNet:
  已验证的 sample generator、SharedLowResProbe / probe parameterization、
  optimizer/checkpoint 训练组件、physics loss
```

允许新增的内容应收敛为：

```text
1 个薄的 neural-NFP integration/adapter
1 个可同时选择 DanMAX/ID16B preset 的 launcher/config
1 个紧凑的 parity/shared-state/synthetic smoke test（也可扩展现有测试）
必要时 1 个评估入口；已有指标脚本能完成时不得重复实现
```

禁止预先创建一整套与 PhaseNet 平行的 model/loss/checkpoint/solver 包。只有当现有组件无法满足 holotomocupy 的传播约定或 global-overlap 参数化时，才可增加最小兼容 wrapper，并在 execution log 记录“为什么不能直接复用”。

特别注意：holotomocupy 的 CuPy 传播本身不能直接穿过 PyTorch autograd。必须先检查 PhaseNet 的 PyTorch propagator 能否通过参数配置严格复现 holotomocupy 的 padding、FFT、频率和 crop 约定；只有 parity test 证明不能直接配置时，才新增一个很薄的 Torch-compatible propagation wrapper。

---

## 6. 神经网络参数化要求

### 6.1 Sample network

使用一个共享 sample generator：

```text
G_sample(z_sample, xy) -> [proj_delta_global, proj_beta_global]
```

可选择：

- U-Net / convolutional decoder；
- deep decoder；
- coordinate network；
- multi-resolution latent grid + decoder。

选择时必须考虑 `nobj` 可能大于 5000，不能假设完整全分辨率 U-Net 一定能放入 GPU。实现必须支持：

```text
低分辨率 smoke test
多分辨率 coarse-to-fine
全局 canvas
按 detector frame 分批计算前向模型
gradient accumulation
activation checkpointing（如需要）
```

输出约束：

```text
proj_beta >= 0                         默认启用
proj_delta scale / sign 可配置
proj_delta gauge 固定                 例如均值为零
```

不得把每张 detector hologram 直接映射成一张独立 phase 图作为最终 sample。

### 6.2 Probe network

使用一个共享 probe generator：

```text
G_probe(z_probe, optional flat channels)
    -> [probe_amp, probe_phase]
```

最低要求：

```text
probe_amp > 0
probe_phase 为实数弧度
所有帧引用同一个输出 tensor
支持从 P=1 初始化
支持从 flat-derived/back-propagated probe 初始化
```

建议实现受限残差参数化：

```text
P = P_init * exp(delta_amp + i * delta_phase)
```

并支持：

```text
probe amplitude smoothness
probe phase smoothness
probe mean-amplitude gauge
probe mean-phase gauge
低分辨率 probe correction
```

### 6.3 Position refinement

初始位置必须来自当前 NFP 的 motor metadata 和几何转换。

第一阶段默认固定位置。随后可选地优化：

```text
pos_final = pos_init + pos_err
```

位置修正需要：

```text
有界参数化或显式 max correction
mean-zero gauge
L2 penalty
可配置 warmup epoch
独立 learning rate
```

不能让位置优化在训练最开始吸收传播距离或 probe 的模型误差。

---

## 7. Epoch 与 batch 的严格定义

对于当前数据集中选中的 `M` 张 NFP 图像（D1/D2 为 100，D3 当前为 64）：

```text
1 epoch = M 张选中图像各自恰好参与一次 sample physics loss
```

必须记录并验证：

```text
epoch frame count
unique frame count
duplicate frame count
missing frame IDs
```

允许为显存使用 mini-batch，但：

- 所有 batch 共享同一个 sample network；
- 所有 batch 共享同一个 probe network；
- optimizer state 跨 batch 保留；
- epoch metric 必须在全部 `M` 帧上按有效 pixel 数正确加权；
- 不得把每个 batch 当成独立 SelfPhish 任务。

配置中应同时支持：

```ini
frames_per_batch=1/2/4/...
gradient_accumulation_frames=...
shuffle_frames=true/false
```

必须提供 full-epoch gradient accumulation 模式，使全部 `M` 帧的梯度可以在一次 optimizer step 前累计；同时允许标准 mini-batch step 作为更快的对照。

---

## 8. Flat-field 约束和诊断

flat target 必须与 sample 使用同一个 intensity normalization scale：

```text
flat_normalized = (flat_mean - dark_mean) / sample_global_mean
flat_target_amplitude = sqrt(max(flat_normalized, 0))
```

最终必须计算并保存：

```text
flat_target_amplitude
flat_pred_amplitude
flat_amplitude_residual =
    flat_pred_amplitude - flat_target_amplitude
```

必须报告：

```text
flat amplitude MSE
flat amplitude RMSE
flat amplitude NRMSE
flat amplitude MAE
flat target/prediction correlation
residual p1/p50/p99
```

必须生成相同显示范围下的：

```text
flat target amplitude
flat predicted amplitude
signed residual
absolute residual
horizontal/vertical residual profiles
```

注意：flat amplitude residual 只能验证 detector-plane amplitude，不能宣称它唯一证明 sample-plane complex probe 正确。报告必须明确这一相位不可辨识性。

---

## 9. 前向模型一致性是第一道验收

不能直接使用 PhaseNet 中现有的简单 circular-FFT Fresnel propagator，然后假设它与 holotomocupy 一致。

当前 holotomocupy `Propagation` 使用：

```text
2*n × 2*n 零填充
Fresnel transfer function
传播后中心裁剪
```

新的 PyTorch 前向算子必须复现同样的：

```text
padding
FFT normalization
frequency convention
distance
pixel size
中心裁剪
complex64 convention
```

必须完成以下 parity test：

### P1：Propagation parity

对以下 complex field：

```text
delta
Gaussian
随机 complex field
真实 probe crop
```

比较 CuPy `Propagation.D` 与 PyTorch 版本：

```text
complex relative L2
amplitude relative L2
intensity relative L2
maximum absolute error
```

目标：

```text
relative L2 <= 1e-4
```

若受 float32/FFT 实现影响不能达到，必须说明误差来源，不能无证据放宽。

### P2：Shift parity

对整数和亚像素位置比较：

```text
Shift.curlySc
PyTorch differentiable shift/crop
```

检查：

```text
row/column 顺序
正负号
中心定义
边界条件
relative error
```

### P3：Autograd finite difference

至少验证：

```text
sample phase gradient
sample beta gradient
probe amplitude gradient
probe phase gradient
position row gradient
position column gradient
```

---

## 10. 强制执行阶段

## Phase 0：仓库状态与设计审计

开始前记录：

```bash
hostname
git status --short
git log --oneline -10
git -C /path/to/PhaseNet status --short
git -C /path/to/PhaseNet log --oneline -10
python --version
python -c "import torch; print(torch.__version__)"
nvidia-smi
```

阅读：

```text
ai_skills/Task_general_instructions.md
ai_skills/HPC_Running_Guides.md
本任务列出的 holotomocupy / PhaseNet 文件
```

先提交一份简短 design note 到 execution log，明确：

```text
sample network 参数化
probe network 参数化
global overlap consistency 实现
torch forward parity 方案
显存估计
checkpoint/resume 方案
PhaseNet 代码复用边界
```

如果当前 worktree 有与本任务重叠的用户修改，不得覆盖；先记录并停止请求用户确认。

## Phase 1：可微物理算子

实现并测试：

```text
Torch Fresnel propagation
Torch global-object shift/crop
valid detector mask
amplitude loss
flat forward branch
position parameterization
```

只有 parity test 通过后才能进入网络训练。

## Phase 2：共享 sample / probe 网络

实现：

```text
一个 global sample generator
一个 shared probe generator
严格共享 tensor 检查
overlap coverage map
gauge constraints
regularization
独立 optimizer parameter groups
```

必须增加自动测试，证明：

```text
不同 frame 获取的 probe storage/value 完全一致
重叠 sample pixel 引用相同 global value
更新一个重叠 pixel 会影响所有覆盖它的 frame
```

## Phase 3：训练循环、checkpoint 和 resume

实现：

```text
all selected frames/scans per epoch
mini-batch / gradient accumulation
sample/probe/position staged schedule
per-epoch full-dataset evaluation
best/last checkpoint
optimizer/scheduler state resume
random seed
NaN/Inf detection
gradient norm logging
GPU memory logging
```

checkpoint 至少保存：

```text
sample network state
probe network state
position parameters
optimizer state
scheduler state
epoch
best metric
config
selected frame IDs
RNG state
```

## Phase 4：安全输出

最终 HDF5 必须先写临时文件：

```text
<name>.h5.part
```

关闭并验证后再原子重命名：

```text
<name>.h5
```

不能让 Silx 看到仍在写入的最终文件名。训练过程中不要长时间保持最终 HDF5 打开；中间状态使用 `.pt` checkpoint。

## Phase 5：最小合成验证

使用当前 holotomocupy 前向模型生成一个小型、至少 16 个二维重叠位置的 synthetic NFP。只保留能够阻止集成错误的两项必要验证：

```text
S1: known probe + fixed position，验证 global neural sample 可恢复并降低 physics loss
S2: unknown shared probe + fixed position，验证 sample/probe 联合更新、probe 严格共享、
    overlap pixel 严格引用同一全局变量
```

位置梯度由 finite-difference 和真实数据 staged-refinement smoke test 验证，不另建大型 synthetic benchmark。禁止为了本任务复制一套独立 synthetic 数据框架。

## Phase 6：真实数据 smoke test

在全分辨率前先运行：

```text
5 frames：I/O 和梯度检查
16/25 frames：二维覆盖 smoke test
低分辨率或中心 crop：显存和收敛检查
```

帧必须覆盖二维扫描区域，不能只取扫描序列的第一行。

检查：

```text
loss 是否下降
probe 是否对所有 frame 一致
global sample 是否正确拼接
overlap 区是否连续
flat loss 是否下降
是否出现 NaN/OOM
```

## Phase 7：三组真实数据完整实验

### D1：DanMAX Siemens Star（无 coded aperture）

```text
data folder:
  /dtu/3d-imaging-center/projects/2026_DANFIX_XHIST/raw_data_3DIM/DanMAX April 2026/NTT_multi_dist/
dark / flat / sample:
  scan-0076.h5 / scan-0097.h5 / scan-0096.h5
detector:
  /entry/measurement/orca
positions:
  /entry/measurement/tom_sam_x
  /entry/measurement/tom_y
frame count:
  100
energy / detector pixel size:
  19.55 keV / 5.5e-7 m
Z1 / focus-to-detector distance:
  0.12669 m / 1.55669 m
```

### D2：DanMAX Si spheres（含 coded aperture）

```text
data folder:
  /dtu/3d-imaging-center/projects/2026_DANFIX_XHIST/raw_data_3DIM/DanMAX April 2026/Si_spheres_tomo_code_550nm/
dark / flat / sample:
  scan-0156.h5 / scan-0187.h5 / scan-0185.h5
detector/positions:
  与 D1 相同
frame count:
  100
energy / detector pixel size:
  19.55 keV / 5.5e-7 m
Z1 / focus-to-detector distance:
  0.14669 m / 1.57669 m
```

D2 的 `Z1` 和 focus-to-detector distance 均比 D1 大 `0.02000 m`。两组 DanMAX 配置必须使用独立 preset；运行前打印并断言数据文件、`Z1` 和 focus-to-detector distance 的组合，禁止把 D1 几何用于 D2，或反之。

### D3：ID16B NFP

从 `experimental/ID16B_NFP/run_step0.sh` / `run_step0_v2.sh` 复用当前默认：

```text
dark:
  /zhome/64/c/214423/bioconfert/raw_data_extern/2026_05_07_ESRF_ID16B/RAW_DATA/
  S2_C1/S2_C1_ht_55nm_06/S2_C1_ht_55nm_06.h5
dark key / frames:
  /3.1/measurement/pco1 / 51

flat:
  /zhome/64/c/214423/bioconfert/raw_data_extern/2026_05_07_ESRF_ID16B/RAW_DATA/
  ptycho_ref_ter/ptycho_ref_ter_0001/ptycho_ref_ter_0001.h5
flat key / scan IDs:
  /{n}.1/measurement/pco1 / 1:10

sample:
  /zhome/64/c/214423/bioconfert/raw_data_extern/2026_05_07_ESRF_ID16B/RAW_DATA/
  ptycho_ht_55nm_06_ter/ptycho_ht_55nm_06_ter_0001/ptycho_ht_55nm_06_ter_0001.h5
sample key / scan IDs:
  /{n}.1/measurement/pco1 / 1:256:4  (64 positions)
motor keys:
  /{n}.1/instrument/positioners/sy
  /{n}.1/instrument/positioners/sz
energy / detector pixel size:
  29.63 keV / 6.5e-7 m
Z1 / focus-to-detector distance:
  0.059599 m / 0.704433 m
```

ID16B 的 scan-dependent HDF5 key、position signs 和单位必须由现有 ID16B reader/config 取得，禁止转换成 DanMAX 的固定-key 假设。

三组实验均必须记录完整绝对路径、实际 frame/scan IDs 和最终几何。不得静默更换 flat、geometry 或 frame subset。

至少运行：

### R1：当前迭代 NFP baseline

每组数据应比较现有两类已验证迭代结果（若对应结果存在）：

```text
v1: sample-data loss 的当前 NFP
v2: 加入 flat-field constraint 的当前 NFP
```

coded-aperture 数据已有的首选 reference candidates 为：

```text
/zhome/64/c/214423/BioToBank/raw_data_extern/XHIST/output/
  CA_output_step0_v1_3712_pos0p1_iter10_100frames/

/zhome/64/c/214423/BioToBank/raw_data_extern/XHIST/output/
  CA_output_step0_v2_3712_pos0p1_iter10_100frames/
```

复用已有结果前必须核验其生成 config/metadata。若与 neural 方法的以下条件不一致，不能直接作为公平 baseline；只在确有必要时用现有 launcher 重跑，避免无意义重复计算：

```text
相同 frames/scan IDs
preprocessing
normalization
geometry
position
valid detector mask
flat target
```

### R2：Neural NFP，位置固定

```text
shared probe
global sample
flat constraint
fixed motor positions
```

### R3：Neural NFP，位置后期开启

先完成 warmup，再允许有界 `pos_err`。

## Phase 8：受控参数搜索与 ablation

参数优化是验收的一部分，但必须避免大规模无约束 Cartesian sweep，也不得为搜索另写一套训练框架。

建议两阶段执行：

```text
Stage A:
  在低分辨率/中心 crop/二维覆盖 subset 上做短程筛选；
  每次只改变一个因素，或使用很小的设计矩阵。

Stage B:
  将前 2 个配置在完整扫描集上运行；
  最优候选至少使用 2–3 个 random seeds 检查稳定性。
```

最低 ablation 因素：

```text
w_flat:
  0 / 0.01 / 0.1 / 1
sample learning rate:
  1e-4 / 3e-4 / 1e-3
probe learning rate:
  1e-5 / 1e-4 / 5e-4
probe initialization:
  unity / flat-derived / current iterative-NFP result
position:
  fixed / warmup 后 staged refinement
update:
  full-epoch gradient accumulation / mini-batch optimizer step
```

网络结构只从 PhaseNet 已验证的现有选项中选择少量配置；禁止为 ablation 新造网络族。可选 adversarial term 只能在 plain physics-loss baseline 稳定后单独做 off/on 对照，不能成为默认方法。

优先在 Siemens Star 上筛选通用参数，再验证其向 coded-aperture 与 ID16B 数据迁移的表现；coded-aperture 数据允许额外做一次小范围 `w_flat`/probe-init ablation。若数据特性迫使三组数据使用不同最优参数，必须如实报告，不能只保留最好看的一组。

不要求神经方法必须优于 baseline 才能如实完成任务，但必须用相同数据和指标比较，不能只展示视觉上最好的一张图。

---

## 11. 最终 HDF5 输出

最终结果必须与当前 NFP 的核心命名兼容，至少包含：

```text
prb_amp
prb_phase
proj_beta
proj_delta
pos
pos_err
frame_ids
dark_mean
flat_mean
flat_minus_dark
flat_target_amplitude
flat_pred_amplitude
flat_amplitude_residual
```

由于用户描述中使用 `probe_amp` / `probe_phase`，输出必须明确记录以下映射：

```text
current NFP canonical names:
  prb_amp / prb_phase

semantic aliases:
  probe_amp / probe_phase
```

可以用 HDF5 hard link 或等价兼容方式提供 aliases，避免复制大型数组；若不创建 aliases，则报告和 metadata 中必须清楚声明二者等价，且下游比较脚本同时接受两组名称。

约定：

```text
prb_amp.shape       = [n, n] 或真实 detector grid
prb_phase.shape     = prb_amp.shape
proj_beta.shape     = [nobj, nobj]
proj_delta.shape    = [nobj, nobj]
pos.shape           = [N, 2]，初始位置
pos_err.shape       = [N, 2]，恢复的位置修正
pos_final           = pos + pos_err
```

允许额外保存：

```text
pos_final
overlap_coverage
sample_residual_mean
sample_residual_selected_frames
loss_history
flat metrics
network metadata
```

所有数组必须：

```text
shape 正确
dtype 明确
无 NaN/Inf
轴和单位写入 attributes
```

HDF5 root attributes 至少包括：

```text
algorithm
creation time
holotomocupy git SHA
PhaseNet git SHA
config text/path
selected frame count
selected frame IDs
energy
wavelength
z1
focus-to-detector distance
effective propagation distance
detector pixel size
sample-plane voxel size
magnification
n / nobj
normalization scale
epoch count
batch size
gradient accumulation
network architecture
loss weights
position refinement schedule
random seed
```

---

## 12. 必须记录的指标

每个 epoch 至少记录：

```text
epoch
learning rates
sample amplitude MSE/RMSE
flat amplitude MSE/RMSE
total loss
probe regularization
sample regularization
position regularization
probe amplitude min/mean/max
probe phase mean/std
proj_beta min/mean/max
proj_delta mean/std
pos_err mean/RMS/max
sample/probe gradient norms
GPU memory allocated/reserved
epoch wall time
unique/missing/duplicate frame count
```

最终 baseline 对比必须在相同 preprocessing、geometry、frame/scan IDs、mask 和 flat target 下完成，至少包括：

```text
sample forward amplitude MSE / RMSE / NRMSE / MAE
逐帧 residual 分布及最差帧
flat amplitude RMSE / NRMSE
probe amplitude correlation
gauge-aligned probe phase comparison
proj_beta / proj_delta visual comparison
probe-object leakage indicator
overlap seam metric
position correction
收敛曲线和 seed 稳定性
runtime
peak GPU memory
```

probe 比较前必须处理不可辨识的 global amplitude / phase gauge，不能直接对未经对齐的 probe phase 做 RMSE。

Siemens Star 还必须优先复用 `experimental/DanMAX_nano/compare_z1_roi_metrics.py` 已有能力，报告可用的：

```text
radial/spoke contrast
annular uniformity / coefficient of variation
x/y directional PSD or resolution cutoff
anisotropy / aspect-ratio / radial-symmetry indicators
```

对于没有 ground truth 的真实数据，禁止仅凭 training loss、肉眼锐度或单个复合分数宣布胜负。方法排序应综合：

```text
held-out forward residual（若扫描覆盖允许，使用空间分散且保持 overlap 的验证帧）
flat residual
Siemens Star 结构/分辨率指标
overlap seam 和 probe-object leakage
不同 seed 的稳定性
运行时间与峰值显存
```

若这些指标给出相互冲突的结论，报告 trade-off，不强行指定单一赢家。

---

## 13. 测试与最低验收标准

- [ ] PyTorch propagation 与 holotomocupy CuPy propagation parity 通过；
- [ ] PyTorch shift/crop 与当前 `Shift` 的 row/col、符号和亚像素定义一致；
- [ ] sample/probe/position autograd finite-difference test 通过；
- [ ] 一个 epoch 恰好覆盖全部选中帧；
- [ ] 所有帧严格共享一个 probe；
- [ ] 所有 sample patch 来自一个 global canvas；
- [ ] overlap consistency 单元测试通过；
- [ ] 至少一个 synthetic known-probe recovery 成功；
- [ ] 至少一个 synthetic joint sample/probe recovery 的 physics loss 明显下降；
- [ ] 真实数据 5-frame 与二维覆盖 smoke test 完成；
- [ ] 三组真实数据的 full-dataset epoch 均能运行、checkpoint、resume；
- [ ] 最终 HDF5 包含规定 datasets 和 metadata；
- [ ] flat target/prediction/residual 已保存并定量分析；
- [ ] D1 Siemens Star 使用 `Z1=0.12669 m`、`L=1.55669 m` 完成完整对比；
- [ ] D2 coded aperture 使用 `Z1=0.14669 m`、`L=1.57669 m` 完成完整对比；
- [ ] D3 ID16B 按现有 scan-dependent key 和 `1:256:4` scan IDs 完成完整对比；
- [ ] 与当前迭代 NFP 使用相同 preprocessing、geometry、frame IDs 和 mask 做公平比较；
- [ ] 完成受控超参数搜索/ablation，并保存全部配置而非只保存最佳结果；
- [ ] 通过多指标和 seed 稳定性说明 neural NFP 与 iterative NFP 各自优劣；
- [ ] Silx 可打开最终原子写入的 HDF5；
- [ ] 任务报告明确局限、不可辨识性和未解决风险。

“代码可以启动”不等于任务完成。若完整分辨率受显存限制，必须给出有测量依据的显存分析、可运行的 multi-resolution/tiling 方案和下一步命令，不能静默改成逐帧独立恢复。

---

## 14. 禁止事项

禁止：

1. 对各扫描图分别运行独立 SelfPhish，并把结果简单拼起来；
2. 为每个扫描位置创建独立 probe；
3. 为每个扫描位置创建互不一致的 sample；
4. 使用与当前 NFP 不同的 propagation，却不做 parity test；
5. 忽略 motor position 或把 frame 顺序当作位置；
6. 只报告 training loss，不计算 detector forward residual；
7. 只比较最终图片，不比较 flat、probe、overlap 和 position；
8. 在错误 geometry 下通过 position/probe 过拟合来宣称恢复成功；
9. 覆盖现有 NFP 输出、PhaseNet checkpoints 或用户数据；
10. 未经许可修改 PhaseNet 当前未提交内容；
11. 在 login node 启动长时间 GPU 训练；
12. 不检查已有 tmux/process 就重复启动任务；
13. 在正在写入的 HDF5 上运行 `h5clear`；
14. 长时间直接写最终 `.h5` 文件名；
15. 将大型 checkpoint 或 HDF5 提交到 git。
16. 为本任务重复实现 PhaseNet 已有的网络、optimizer、loss 或 checkpoint 框架；
17. 为追求表面上的更优结果，给 baseline 和 neural 方法使用不同数据、geometry、mask 或 normalization；
18. 只报告最佳超参数/seed，隐藏失败或不稳定配置。

---

## 15. HPC 与运行规则

实施前必须阅读：

```text
ai_skills/HPC_Running_Guides.md
```

长时间实验必须：

```text
在 GPU node 上运行
使用 tmux
stdout/stderr 同时写入 task-specific runtime log
先检查 tmux ls 和现有 process
记录 CUDA_VISIBLE_DEVICES
记录 nvidia-smi
```

推荐环境：

```bash
cd /dtu/3d-imaging-center/projects/2022_DANFIX_08_BioComFert/raw_data_extern/Chengpeng/PycharmProjects/holotomocupy
. /zhome/64/c/214423/projects/BioComfert/codes/Tomocupy/Job_Scripts/miniconda_chengpeng.sh
conda activate dl_torch
```

实际运行前验证（测试路径以最终最小整合位置为准）：

```bash
python -c "import torch, cupy, h5py; print(torch.__version__)"
python -m pytest <neural-nfp parity/smoke test> -q
```

如发生 OOM：

- 先确认旧进程已退出；
- 保留 checkpoint 和日志；
- 优先降低 batch、启用 gradient accumulation/checkpointing；
- 不得通过退化成 independent-frame recovery 绕过显存问题。

非 OOM 错误按 HPC guide 停止并报告，不得盲目重复提交。

---

## 16. 预期代码和科学交付物

预期轻量代码交付原则：

```text
优先扩展/调用现有模块；
新增 1 个 integration/adapter；
新增 1 个 launcher/config；
新增或扩展 1 个紧凑测试入口；
只有现有评估脚本无法覆盖时才新增小型 evaluation adapter。
```

实际文件名在 Phase 0 组件审计后确定，不预设新增完整 package tree。

运行产物必须放在新的 task-specific 子目录，且不得覆盖现有结果：

```text
DanMAX:
  /zhome/64/c/214423/BioToBank/raw_data_extern/XHIST/output/

ID16B:
  /zhome/64/c/214423/BioToBank/raw_data_extern/XHIST/ID16B_output/
```

两处根目录下建议按 `dataset/method/config/seed` 分层，并包含 `configs/`、`checkpoints/`、`results/`、`figures/` 和 `runtime_logs/`。每次运行必须使用新目录或明确 resume 原目录，禁止覆盖 iterative-NFP baseline 和已有 neural 结果。

大型 `.pt`、HDF5 和逐帧 residual 不进入 git。

---

## 17. Expected output files

HPC-side Codex CLI 必须创建：

```text
ai_skills/logs/2026-07-23_nfp_shared_probe_neural_execution_log.md
ai_skills/summaries/2026-07-23_nfp_shared_probe_neural_summary.md
ai_skills/reports/2026-07-23_nfp_shared_probe_neural_report.md
```

Execution log 必须持续记录：

```text
时间
hostname/GPU
git 状态
检查的文件
设计决策
执行命令
测试结果
实验状态
错误和处理
生成文件的绝对路径
```

Summary 必须包含：

```text
task name
execution date
machine / hostname
current git branch
git status before and after
commands executed
files inspected
files modified
outputs generated
errors/warnings
completion status
recommended next steps
```

Report 必须包含：

```text
网络与物理模型
共享 probe 证明
global overlap consistency 证明
forward parity 结果
最小 synthetic recovery 结果
DanMAX Siemens Star 100-frame 结果
DanMAX coded-aperture 100-frame 结果
ID16B 64-position 结果
flat amplitude/residual 分析
与迭代 NFP 的公平对比
超参数搜索与 ablation（包括失败/不稳定配置）
多指标方法排序与 trade-off
显存和运行时间
失败实验
不可辨识性与局限
推荐配置
后续工作
```

追踪链必须保持：

```text
ai_skills/tasks/2026-07-23_nfp_shared_probe_neural_phase_retrieval.md
    ↓
ai_skills/logs/2026-07-23_nfp_shared_probe_neural_execution_log.md
    ↓
ai_skills/summaries/2026-07-23_nfp_shared_probe_neural_summary.md
    ↓
ai_skills/reports/2026-07-23_nfp_shared_probe_neural_report.md
```

---

## 18. 启动闸门

当前只生成任务描述。

用户已经确认：

```text
D1 DanMAX Siemens Star:
  dark/flat/sample = scan-0076/0097/0096
  Z1=0.12669 m, focus-to-detector distance=1.55669 m

D2 DanMAX coded aperture:
  dark/flat/sample = scan-0156/0187/0185
  Z1=0.14669 m, focus-to-detector distance=1.57669 m

D3 ID16B:
  使用现有 experimental/ID16B_NFP launcher 的数据与配置

代码策略:
  尽量复用 holotomocupy 和 PhaseNet，只做最小整合

输出根目录:
  DanMAX -> /zhome/64/c/214423/BioToBank/raw_data_extern/XHIST/output/
  ID16B -> /zhome/64/c/214423/BioToBank/raw_data_extern/XHIST/ID16B_output/
```

当前仍只生成任务描述。在用户明确回复批准启动任务前，不得开始实现、测试、GPU 实验、创建分支或修改 PhaseNet。启动时如尚未给出 GPU/时间预算，可以先完成只读审计和轻量 CPU parity test；任何长时间 GPU 实验前必须再次确认可用资源。
