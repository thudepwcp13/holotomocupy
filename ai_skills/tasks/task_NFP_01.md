# Task NFP-01：诊断 DanMAX Siemens Star 的 NFP 重建纵向拉伸问题

## 1. 任务目标

对当前 `holotomocupy` 中 DanMAX nano NFP 流程进行系统性诊断，找出为什么恢复出的 `proj_delta` 和 `proj_beta` 呈现明显纵向拉长的 Siemens Star，而原始 detector hologram 中 Siemens Star 的几何比例看起来基本正常。

本任务不能只靠调参猜测。必须通过：

1. 代码级审计；
2. 原始数据定量分析；
3. 合成数据单元测试；
4. 真实数据消融实验；
5. 前向预测残差分析；

最终给出有证据支持的根因排序，并提交可复现的修复方案。

---

## 2. 仓库与主要文件

仓库：

```text
holotomocupy
```

重点检查：

```text
experimental/DanMAX_nano/step0.py
experimental/DanMAX_nano/config_step0.conf
experimental/DanMAX_nano/run_step0.sh
src/holotomocupy/rec_nfp_mpi.py
src/holotomocupy/shift.py
src/holotomocupy/propagation.py
src/holotomocupy/chunking.py
src/holotomocupy/mpi_functions.py
src/holotomocupy/config.py
```

在开始工作前：

```bash
git status
git log --oneline -10
```

新建调试分支，禁止直接覆盖当前可运行版本：

```bash
git switch -c debug/nfp-elongation-01
```

---

## 3. 数据位置与 HDF5 布局

### 3.1 数据目录

```text
/dtu/3d-imaging-center/projects/2026_DANFIX_XHIST/raw_data_3DIM/DanMAX April 2026/NTT_multi_dist/
```

已确认文件对应关系：

```text
scan-0076.h5  dark
scan-0097.h5  flat
scan-0096.h5  Siemens Star sample / NFP scan
```

### 3.2 HDF5 路径

```text
Detector intensity: /entry/measurement/orca
Motor x:            /entry/measurement/tom_sam_x
Motor y:            /entry/measurement/tom_y
```

### 3.3 数据尺寸

```text
dark:   (100, 2592, 3712)
flat:   (50, 2592, 3712)
sample: (100, 2592, 3712)
```

扫描大致为规则的 `10 × 10` lateral grid。

---

## 4. 当前几何参数

```ini
energy=19.55
z1=0.12669
focustodetectordistance=1.55669
detector_pixelsize=5.5e-7
position_unit=mm
pos_row_sign=-1.0
pos_col_sign=1.0
center_positions=true
flat_correct=false
```

由当前参数得到：

```text
magnification ≈ 12.2874
object-plane voxel size ≈ 44.761 nm
```

当前位置范围：

```text
position pix row = [-674.483, 679.364]
position pix col = [-693.255, 680.699]
```

对应跨度：

```text
row span ≈ 1353.847 px
col span ≈ 1373.954 px
```

从原始图像中手动估计 Siemens Star 中心移动跨度：

```text
row span ≈ 1324 px
col span ≈ 1338 px
```

粗略 position scale：

```text
row scale ≈ 1324 / 1353.847 ≈ 0.978
col scale ≈ 1338 / 1373.954 ≈ 0.974
```

两个轴的相对比例差只有约 0.4%，不足以单独解释目前明显的纵向拉伸，但需要进一步做逐帧回归验证。

---

## 5. 当前已观察到的现象

### 5.1 原始数据

从 frame 0 到 frame 99：

```text
Siemens Star 在 detector 图像中从左向右、从下向上移动。
```

根据当前 `pos` 和 `Shift` 约定，global object window 在 canvas 上从右上向左下移动，方向正好相反。方向和符号目前看起来合理。

### 5.2 NFP 输出

在以下两种设置中，`proj_delta` 和 `proj_beta` 均呈现明显纵向拉长的 Siemens Star：

#### 设置 A：完整横向 FOV + padding

```ini
n=3712
use_valid_detector_mask=false/true
```

真实 detector 为 `2592 × 3712`，上下各 padding 560 行。

启用 `use_valid_detector_mask=true` 后，padding 区域已在 loss、gradient 和 Hessian 中设为零权重，但结果基本不变。

#### 设置 B：无 padding

```ini
n=2592
```

此时使用：

```text
rows[0:2592]
cols[560:3152]
```

即不 padding，但横向左右各裁掉 560 像素。运行 10 次迭代后，纵向拉长仍明显存在。

### 5.3 Probe

`prb_amp` 和 `prb_phase` 看起来与 KB mirror illumination 的横纵网格结构一致，整体形态较合理。但不能仅凭视觉认定 probe/object 已正确分离。

### 5.4 Position refinement

10 次迭代后的 `pos_err` 很小：

```text
row correction 约在 ±0.2 px 内
col correction 约在 ±0.12 px 内
```

没有明显漂移或周期性大跳变。因此当前纵向拉伸不像是 position optimizer 自己把 object 拉长。

### 5.5 误差变化示例

`n=2592, niter=10`：

```text
Initial err = 1.22709e-02
iter 0      = 3.81358e-03
iter 9      = 3.40611e-03
```

误差仍在下降，但 object 中仍存在明显 detector-plane hologram/fringe 特征。

---

## 6. 高优先级疑点

必须逐项验证，不允许直接假设。

### H1. MPI 输出拼接错误

当前 DanMAX `step0.py` 可能包含类似：

```python
projects = comm.gather(cl.vars["proj"].get(), root=0)
proj = np.concatenate(projects, axis=0)
```

但 `RecNFP` 中 `proj` 在各 rank 上是 replicated，而不是按 row 分块。若使用多 MPI rank，沿 axis 0 拼接会把相同或近似相同的 object 纵向重复，直接造成输出纵向拉长。

必须：

1. 记录实际运行时 `MPI.COMM_WORLD.Get_size()`；
2. 打印每个 rank 的 `proj.shape`；
3. 检查最终 HDF5 中 `proj_delta.shape`；
4. 若 `ngpus > 1`，验证是否发生 `ngpus × nobj` 的纵向拼接；
5. replicated object 应只取 rank 0，或先 allreduce 后只保存一次，不能 concatenate。

这是第一优先级检查项。

### H2. `Shift` 中 row/column 顺序或符号与 `step0.py` 不一致

必须审计：

```text
pos[:, 0]
pos[:, 1]
Shift.curlySc
Shift.dcurlySc
Shift.dcurlySadjc
Shift.coeff
```

确认：

```text
pos[:,0] 是否始终表示 row/y
pos[:,1] 是否始终表示 col/x
```

需要构建 impulse 单元测试，不仅阅读代码。

### H3. 原始 detector 位移与 motor position 存在二维仿射关系

当前只使用两个独立 scale 和 sign，假设 motor axes 与 detector axes 完全正交且对齐。

真实关系可能为：

```text
[detector_row]   [a11 a12] [tom_y    ]   [b1]
[detector_col] = [a21 a22] [tom_sam_x] + [b2]
```

必须从原始 100 帧中估计 Siemens Star 中心，拟合完整 2×2 affine matrix，而不是只比较首尾跨度。

### H4. 输出可视化或数组 shape 问题

必须检查：

```python
print(proj_delta.shape)
print(proj_beta.shape)
```

并使用：

```python
plt.imshow(proj, aspect="equal", origin="upper")
```

同时保存不经过任何 resize 的 TIFF/PNG。

验证拉伸是数组内部真实几何失真，而不是：

- MPI 拼接；
- plotting `aspect="auto"`；
- 非等比例 resize；
- transpose；
- HDF5 viewer 的显示比例。

### H5. Probe–object ambiguity / raster-grid pathology

数据为规则约 `10 × 10` grid，probe 本身又有很强的水平/垂直网格。这可能导致：

```text
probe pattern 泄漏到 object
object structure 泄漏到 probe
规则 raster artifact
```

需要做 staged optimization 和合成数据验证。

### H6. 迭代次数和初始化不足

当前仅测试 5 或 10 次迭代，而默认流程通常约 129 次。

必须判断拉伸是否：

- 随迭代逐渐收缩；
- 固定保持；
- 继续恶化；
- error 下降但几何不改善。

### H7. Fresnel scaling geometry

审计：

```python
magnification = focus_to_detector / z1
z2 = focus_to_detector - z1
distance = z1 * z2 / focus_to_detector
voxelsize = detector_pixelsize / magnification
```

确认该 Fresnel scaling theorem 对当前 DanMAX geometry 和数据定义正确。

检查：

- `z1` 是否真的是 focus-to-sample distance；
- `focusToDetectorDistance` 定义是否一致；
- detector pixel size 是否已经包含显微光学放大；
- 是否存在 binning；
- x/y pixel size 是否相同；
- HDF5 数据在写入前是否被 resize 或 binning。

### H8. 物体模型和输出命名

当前 forward model：

```text
D(prb * exp(i * shifted_proj))
```

`proj.real` 和 `proj.imag` 实际是 phase/absorbance exponent，不一定是直接的 material `delta/beta`。

这不会直接导致拉伸，但必须确认 complex object 参数化、符号和约束没有轴相关 bug。

---

## 7. 强制执行流程

## Phase 0：建立可复现 baseline

创建独立输出目录：

```text
outputs/NFP_01/baseline_3712_masked/
outputs/NFP_01/baseline_2592/
```

保存：

```text
完整 config
Git commit SHA
Python/CuPy/CUDA/MPI 版本
GPU 型号
MPI rank 数
完整 stdout/stderr log
conv_nfp.csv
最终 HDF5
每个 checkpoint 的 probe/object
```

Baseline 命令示例：

```bash
OUT_DIR=$PWD/outputs/NFP_01/baseline_3712_masked \
N=3712 \
NITER=10 \
USE_VALID_DETECTOR_MASK=true \
FLAT_CORRECTION=false \
NGPUS=1 \
bash experimental/DanMAX_nano/run_step0.sh nfp
```

必须先用 `NGPUS=1` 建立 baseline，避免 MPI 输出拼接干扰。

随后再单独测试 `NGPUS=2`，只用于检查 MPI 一致性。

---

## Phase 1：定量确认“拉伸”

编写：

```text
experimental/DanMAX_nano/debug_nfp_01_measure.py
```

功能要求：

1. 读取原始 frame 0–99；
2. 自动或半自动估计 Siemens Star 中心；
3. 输出每帧：

```text
frame, motor_x, motor_y, center_row, center_col, confidence
```

4. 对 detector center 与 motor position 做 2D affine 拟合；
5. 输出 affine matrix、残差、R²；
6. 绘制：

```text
motor-predicted center vs measured center
row residual vs frame
col residual vs frame
measured scan trajectory
```

7. 对最终 `proj_delta/proj_beta` 定量计算几何比例：

可用方法：

- 阈值/梯度能量 bounding box；
- 二阶矩椭圆；
- 径向对称性优化；
- x/y Fourier power spectrum；
- 与一个圆形/已知 Siemens Star 模板做 anisotropic scale registration。

输出：

```text
estimated object width
estimated object height
height/width ratio
best anisotropic correction scale
```

不要仅靠截图判断。

---

## Phase 2：MPI 与输出 shape 审计

必须在 `step0.py` 中临时打印：

```python
rank
world_size
cl.vars["proj"].shape
cl.vars["prb"].shape
local projection checksum
```

验证：

```text
NGPUS=1 输出 shape
NGPUS=2 输出 shape
NGPUS=2 各 rank object 是否 replicated
最终 HDF5 是否错误 concatenate
```

如果确认 bug，修复为：

```python
# replicated object: save rank-0 copy only
proj_np = cl.vars["proj"].get() if rank == 0 else None
```

或正确的 allreduce/一致性检查，但不能沿 object row 直接 concatenate。

修复后必须比较单 GPU 与双 GPU 输出：

```text
relative L2 difference
shape
visual geometry
```

---

## Phase 3：Shift 单元测试

新增测试脚本：

```text
experimental/DanMAX_nano/test_shift_axes_nfp01.py
```

至少包含：

### Test A：单 impulse

在 `nobj × nobj` object 中放一个中心 impulse，分别设置：

```text
pos = [0, 0]
pos = [+10, 0]
pos = [-10, 0]
pos = [0, +10]
pos = [0, -10]
```

记录 `curlySc` 输出中 impulse 的 row/col 移动。

### Test B：非对称标记

构建带有 `L` 形、数字或两个不同距离点的 object，避免 transpose/flip 后仍难以辨认。

### Test C：adjoint check

验证：

```text
<Shift(x), y> ≈ <x, Shift^T(y)>
```

### Test D：finite-difference gradient

对 row 和 col position 分别做 finite difference，验证 `dcurlySc` 和 `dcurlySadjc`。

输出所有误差和通过/失败状态。

---

## Phase 4：Propagation 单元测试

新增：

```text
experimental/DanMAX_nano/test_propagation_isotropy_nfp01.py
```

构建：

- 中心 delta；
- 圆形 aperture；
- 圆对称 Gaussian；
- 水平和垂直等宽线。

经过 `Propagation.D` 后检查：

```text
x/y radial symmetry
horizontal/vertical fringe spacing
D 和 DT 的 adjoint consistency
FFT axis order
pixel size 是否在两轴一致使用
```

若一个圆形输入经过 propagation 后变成椭圆，说明 propagation 或数据 reshape 存在轴相关问题。

---

## Phase 5：端到端合成 NFP 测试

这是必须完成的核心实验。

新增：

```text
experimental/DanMAX_nano/simulate_debug_nfp01.py
```

使用与真实数据一致的：

```text
energy
z1
focus-to-detector distance
detector pixel size
100 个 scan positions
n / nobj
```

生成已知 truth：

```text
object: 圆形 Siemens Star 或非对称 test target
probe amplitude: 使用 flat-derived pattern 或合成 KB grid
probe phase: 平滑低频 + 横纵条纹
```

前向生成 detector intensities，再使用当前 `RecNFP` 重建。

至少做以下 case：

### S1

```text
known probe
fixed exact positions
only reconstruct object
```

### S2

```text
unknown probe
fixed exact positions
joint object + probe
```

### S3

```text
unknown probe
positions allowed to refine
```

### S4

```text
3712 square + valid detector mask
```

### S5

```text
2592 centered crop
```

### S6

```text
position scale artificially set to 0.97 / 1.03
```

### S7

```text
motor axes 加入 1–3 degree rotation/shear
```

要求：

- 若合成 truth 也被纵向拉长，优先定位代码/solver bug；
- 若合成数据恢复正常，问题更可能在真实数据 geometry、probe stability、position metadata 或模型不匹配。

输出 truth、reconstruction、difference、aspect ratio 和相对误差。

---

## Phase 6：真实数据消融实验

所有实验使用独立输出目录，不得覆盖。

### E1：固定 position

禁止 position 更新，确认 position optimizer 不是原因。

如果当前 solver 不支持 freeze variable，请增加明确参数，例如：

```ini
update_position=false
```

不要用极小 `rho_pos` 代替真正 freeze，除非证明数学上完全等价。

### E2：固定 probe，先只恢复 object

probe 初值建议至少测试：

```text
P0 = 1
P0 amplitude = sqrt(flat-dark) normalized, phase=0
P0 = flat-derived amplitude back-propagated to sample plane
```

先固定 probe 运行 object-only，再释放 probe。

### E3：staged optimization

建议：

```text
stage 1: object only, 16–32 iter
stage 2: object + weak probe, 32 iter
stage 3: object + probe, positions fixed, 32–64 iter
stage 4: optional position refinement, 16 iter
```

### E4：迭代数

至少保存：

```text
iter 0, 1, 2, 4, 8, 16, 32, 64, 128
```

分析 aspect ratio 随迭代变化，而不是只看最终图。

### E5：RHO 网格

至少测试：

```text
1,2,0.1
1,1,0
1,0.5,0
1,0.2,0
1,0.5,0.1
```

必须记录：

```text
error
object aspect ratio
probe-object leakage metric
position correction
```

### E6：flat correction

比较：

```text
flat_correct=false
flat_correct=true
```

### E7：position calibration

比较：

```text
raw config positions
row_scale=0.978, col_scale=0.974
full 2×2 affine-calibrated positions
```

### E8：geometry sensitivity

只做小范围 sweep：

```text
z1 × [0.97, 0.985, 1.0, 1.015, 1.03]
detector_pixelsize × [0.97, 0.985, 1.0, 1.015, 1.03]
```

注意 detector pixel size 同时影响 propagation 和 position conversion。为了分离影响，需要额外支持：

```ini
position_row_scale
position_col_scale
```

不要只通过修改 detector pixel size 来校准 position。

---

## Phase 7：前向残差诊断

对每个关键实验输出：

```text
predicted sqrt intensity
measured sqrt intensity
residual
relative residual
```

至少检查 frame：

```text
0, 9, 45, 90, 99
```

分别计算：

```text
whole valid detector RMSE
central Siemens Star area RMSE
background/probe area RMSE
horizontal residual spectrum
vertical residual spectrum
```

如果整体 error 下降但 Siemens Star 区域 residual 仍有方向性结构，说明 solver 主要通过 probe/background 拟合降低了全局 loss。

同时生成 residual mean map：

```text
mean(abs(residual), axis=frame)
```

检查是否存在横纵方向系统误差。

---

## 8. 建议增加的诊断指标

每个 checkpoint 记录：

```text
iteration
loss
proj aspect ratio
proj support width/height
probe amplitude dynamic range
probe phase standard deviation
probe/object normalized cross-correlation
position correction max/mean
forward residual RMSE
```

建议实现：

```text
outputs/NFP_01/<experiment>/metrics.csv
```

Probe/object leakage 可用以下近似指标：

1. 将恢复 object 的中心窗口与 probe amplitude/phase 做 normalized cross-correlation；
2. 比较二者 Fourier power spectrum 的横纵峰位置；
3. 检查 object 是否含有与 flat/probe 相同的固定横纵 grid frequency。

---

## 9. 判定逻辑

### 情况 A：多 GPU 才拉长，单 GPU 正常

结论优先指向 MPI replicated-object 拼接或聚合 bug。

### 情况 B：合成数据也拉长

结论优先指向：

```text
Shift axis/sign
Propagation axis scaling
solver gradient/Hessian
output gather/reshape
```

### 情况 C：合成数据正常，真实数据拉长

结论优先指向：

```text
position metadata affine mismatch
sample/probe instability
geometry metadata错误
regular raster ambiguity
single-distance model mismatch
flat/dark preprocessing
```

### 情况 D：object-only + fixed known probe 正常，joint recovery 拉长

结论优先指向 probe–object ambiguity 或更新策略。

### 情况 E：长迭代后逐渐恢复正常

主要原因是初始化/未收敛，但仍需说明需要的 staged schedule 和稳定参数。

### 情况 F：长迭代 error 降低但 aspect ratio 不改善

说明低 loss 并不能唯一约束正确 object，需要引入：

```text
better probe initialization
regularization
probe constraints
position calibration
multi-distance data
non-raster scan diversity
```

---

## 10. 不允许的做法

禁止：

1. 只看 1–2 张图片后给出猜测；
2. 只调 `rho` 不做代码审计；
3. 只改 detector pixel size，同时混淆 propagation 与 position scale；
4. 覆盖原始 HDF5；
5. 覆盖 baseline 输出；
6. 在多 GPU 下直接 concatenate replicated object；
7. 只报告 loss，不报告 object aspect ratio；
8. 修复后不跑合成回归测试；
9. 未记录 Git commit/config/environment 就宣称问题解决。

---

## 11. 最终交付物

必须提交以下文件：

```text
analysis_NFP_01.md
experimental/DanMAX_nano/debug_nfp_01_measure.py
experimental/DanMAX_nano/test_shift_axes_nfp01.py
experimental/DanMAX_nano/test_propagation_isotropy_nfp01.py
experimental/DanMAX_nano/simulate_debug_nfp01.py
experimental/DanMAX_nano/run_debug_matrix_nfp01.sh
outputs/NFP_01/summary_metrics.csv
outputs/NFP_01/figures/
```

如需修改生产代码，提交清晰独立的 commit，并在报告中说明：

```text
修改前行为
根因
修改内容
为什么该修改数学上正确
单元测试
真实数据对比
是否影响其他 experimental pipelines
```

`analysis_NFP_01.md` 至少包含：

1. 根因候选排序；
2. 每个候选的支持/反对证据；
3. 最终最可能根因；
4. 是否存在多个共同原因；
5. 修复前后 aspect ratio；
6. 修复前后 loss 和 residual；
7. probe/object leakage 对比；
8. 单 GPU/多 GPU 一致性；
9. 推荐生产参数；
10. 尚未解决的风险。

---

## 12. 最低验收标准

任务不能以“可能是未收敛”结束。至少必须完成：

- [ ] 检查最终 HDF5 的真实 shape 和 MPI world size；
- [ ] 排查 replicated object 的 gather/concatenate；
- [ ] 完成 Shift impulse/adjoint/finite-difference 测试；
- [ ] 完成 propagation isotropy 测试；
- [ ] 完成至少一个端到端合成 NFP recovery；
- [ ] 对真实 100 帧拟合 motor-to-detector affine transform；
- [ ] 量化 reconstruction aspect ratio；
- [ ] 完成 fixed-position、fixed-probe、joint-recovery 三组对照；
- [ ] 生成 forward residual maps；
- [ ] 给出可复现的根因证据和修复 patch，或明确证明当前模型对该数据不可辨识。

---

## 13. 当前优先级建议

建议按以下顺序开展：

```text
1. 检查 MPI 输出拼接和最终 dataset shape
2. Shift axis impulse test
3. 量化原始 frame center trajectory + affine fit
4. 合成数据端到端 recovery
5. fixed probe / fixed position 消融
6. 64–128 iter convergence trajectory
7. geometry 与 RHO sensitivity
```

第一项非常重要：如果曾使用超过一个 MPI rank，而 `proj` 是 replicated 后沿 axis 0 concatenate，那么它可以直接产生纵向拉长或重复输出，必须最先排除。
