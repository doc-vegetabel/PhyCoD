

# Full-Order Corrected Student 项目代码说明书（阶段 1 起始版）

## 2026-05-10 init-checkpoint continuation update

| 修改时间 | 涉及脚本/文件 | 需增改说明 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-10 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 新增续训初始化入口 | 新增 `--init-checkpoint` 参数，用于在训练开始前加载已有 `model_state_dict`、`encoder_state_dict` 或原始 state dict；只初始化模型权重，不恢复 optimizer state，便于在已有 best/last checkpoint 基础上更换学习率、gate 权重和 loss 权重继续训练。 | phase-gated fast residual 训练效率优化 |

## 2026-05-10 generalization training update

| 修改时间 | 涉及脚本/文件 | 需增改说明 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-10 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 新增状态驱动 no-regression window guard | 新增 `--use-state-window-no-regression-guard` 及 `--w-state-no-regression-*` 参数；训练时不依赖 case name，而是使用 static-vs-teacher 局部窗口质量自动识别 static-good 窗口，并约束 pred 在这些窗口内的响应误差、相关性和幅值不要相对 static 退化。 | 工况泛化训练 |
| 2026-05-10 Asia/Shanghai | `scripts/generate_random_continuous_loads.py` | 新增随机连续载荷生成脚本 | 新脚本读取参考 complex case 的前三行表头格式，生成随机连续频率、相位、幅值、空间分布、多频/包络/chirp 混合的 train/valid/test dat 载荷组，并输出 manifest 与 load file list。 | 工况泛化数据构造 |

## 2026-05-11 high-frequency phase-drift update

| 修改时间 | 涉及脚本/文件 | 需增改说明 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-11 Asia/Shanghai | `src/student/transformer/frequency_losses.py` | 新增高频相位漂移率损失 | 新增 `phase_drift_rate_loss(...)`，在局部窗口上按 teacher 高频频带能量自动加权，惩罚 soft lag 及相邻窗口 lag 增量，用于约束高频持续激励下的后期相位累积漂移。 | high-frequency phase retention |
| 2026-05-11 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 接入 phase drift-rate loss | 新增 `--use-phase-drift-rate-loss`、`--w-phase-drift-*` 和 `--phase-drift-*` 参数，并将 `phase_drift_loss` 及相关诊断指标写入 `training_history.csv`。 | high-frequency phase retention |
| 2026-05-11 Asia/Shanghai | `scripts/generate_random_continuous_loads.py` | 扩展高频持续周期数据生成能力 | 新增 `--families`、`--chirp-weight-max`、`--burst-weight-max` 参数，可生成仅包含 single/multi 的持续高频周期载荷族；已用于构造 `data/load/high_periodic_alpha_v1`，该数据目录不纳入 git。 | high-frequency replay data |

本说明书仅涵盖当前项目的主线代码、核心脚本、叶片参数文件及验证规划。旧版的 `physics_difference` / 模态 (modal) / 缩减阶 (reduced) 相关脚本已移出主线，不在此列。

---

## 1. 当前项目主线数据流

项目的核心计算路径如下：

```text
blade_master.csv
  -> StudentBeamModel
  -> build_fem_matrices_6dof(...)
  -> 生成 M, K 矩阵
  -> 计算阻尼 C = beta_damp * K
  -> 加载 F_time (时历载荷)
  -> Full-order Newmark rollout (数值积分)
  -> 输出 u_full, v_full, a_full (全场响应)
```

**当前唯一基线 (Baseline):**
通过 `run_student_case(...)` 函数生成的响应。

**验证状态：**
新建的 `FullOrderCorrectedCoreTorch` 在零修正状态下，已通过 `test_complex_case.dat`、`train_complex_case.dat` 及零载荷工况验证，确认能够完美回退至 `direct student baseline`。

---

## 2. 推荐目录结构

建议将项目主线相关文件按以下结构组织：

```text
PhyCoD_gpt/
├── data/
│   ├── raw/
│   │   └── nrel5mw/
│   │       └── blade_master.csv       # 叶片结构参数
│   └── load/
│       ├── test_complex_case.dat      # 测试集载荷
│       └── train_complex_case.dat     # 训练集载荷
├── src/
│   └── student/
│       ├── model.py                   # 数据容器类
│       ├── io.py                      # 数据读取
│       ├── fem_builder.py             # FEM 矩阵构造
│       ├── dynamic_solver.py          # Numpy 版动力学求解器
│       ├── load_adapter.py            # 载荷映射适配
│       └── full_order_corrected_core_torch.py  # Torch 版修正核心 (核心文件)
├── scripts/
│   └── run_student_cases.py           # 生成基线数据的工程脚本
└── tests/
    └── compare_full_corrected_core_vs_student.py # 一致性/回归测试脚本
```

---

## 3. 主线核心脚本说明

### 3.1 `src/student/full_order_corrected_core_torch.py`

> **一句话说明：**
> 新建的 Full-order Torch 动力学核心，直接在张量层级处理 `u / v / a / F_time`。在零修正时，严格复刻 Direct Student 的 Newmark-beta 数值积分过程。

* **调用/依赖：** `numpy`, `torch`, `torch.nn`
* **不依赖：** 旧版的 `ModalAdapter`, `NominalStudentCore`, `q/qdot` 等模态空间变量。
* **核心功能：**
    * `compute_initial_acceleration(...)`: 计算初始加速度。
    * `step(...)`: 单步 Newmark 推进。
    * `rollout(...)`: 全时历响应求解。
* **当前 Correction 接口：** 已预留 `build_delta_M`, `build_delta_C`, `build_delta_K_mat`, `build_delta_K_geo`, `build_force_correction` 等接口，目前默认返回 0。
* **项目角色：** 阶段 1 的核心文件。未来所有的物理修正逻辑都将在此 Core 或配套的 Builder 中实现。

### 3.2 `tests/compare_full_corrected_core_vs_student.py`

> **一句话说明：**
> 用于验证 Torch 版 Core 与 Numpy 版 Baseline 在相同输入下的一致性。

* **核心流程：**
    1. 调用 `run_student_case` 获取基线响应。
    2. 构造 Full-order M/K/C 矩阵并初始化 `FullOrderCorrectedCoreTorch`。
    3. 执行 Zero-correction Rollout。
    4. 比较误差并输出 `full_x/full_y/tip_x/tip_y` 指标。
    5. 保存 JSON 格式的对比报告。
* **项目角色：** 阶段 1 的一致性测试脚本，未来将作为回归测试 (Regression Test) 长期保留。

---

## 4. Direct Student Baseline 依赖脚本说明

### 4.1 `scripts/run_student_cases.py`
* **说明：** 基线数据的唯一入口，负责从 CSV 和载荷文件生成标准的 `u/v/a` 响应。
* **输出：** 包含时间、位移、速度、加速度、载荷、固有频率及所用参数。

### 4.2 `src/student/dynamic_solver.py`
* **说明：** 提供 Numpy 版 Newmark-beta 求解器，是项目的“公式金标准”。
* **核心类：** `NewmarkBetaSolver`, `WindBladeDynamicSystem`。

### 4.3 `src/student/fem_builder.py`
* **说明：** 构造 FEM 质量矩阵 `M` 和刚度矩阵 `K`。
* **核心细节：** 每个节点 6 DOF (`ux, uy, uz, theta_x, theta_y, theta_z`)；固定根部节点（即截取 `M[6:, 6:]`）。

### 4.4 `src/student/load_adapter.py`
* **说明：** 读取多点时历载荷文件，并按 `eta` (展向位置) 线性分配到 Student 节点。

### 4.5 `src/student/io.py` & `src/student/model.py`
* **说明：** 负责读取 `blade_master.csv` 并定义 `StudentBeamModel` 数据容器（包含质量分布、EI、GJ 等参数）。

---

## 5. 叶片参数与载荷文件

| 文件路径 | 说明 | 角色 |
| :--- | :--- | :--- |
| `data/raw/nrel5mw/blade_master.csv` | 叶片结构参数（质量、刚度分布） | 结构参数标准来源 |
| `data/load/test_complex_case.dat` | 复杂验证工况 | 阶段 1 标准回归测试工况 |
| `data/load/train_complex_case.dat` | 复杂训练工况 | 阶段 1 一致性测试工况 |

---

## 6. 当前测试脚本规划

建议通过以下命令进行单工况对齐测试：

```bash
# 1. 零载荷自检
python tests/compare_full_corrected_core_vs_student.py --no-time-series-load --case-name zero_load_check

# 2. 测试集复杂工况对齐
python tests/compare_full_corrected_core_vs_student.py --time-series-load-file data/load/test_complex_case.dat --case-name test_complex_case_zero_correction

# 3. 训练集复杂工况对齐
python tests/compare_full_corrected_core_vs_student.py --time-series-load-file data/load/train_complex_case.dat --case-name train_complex_case_zero_correction
```

---

## 7. 当前不进入主线的旧脚本

以下脚本属于旧版的 **Reduced-order / Modal correction** 路线，当前 Full-order 主流框架不依赖它们：
* `src/physics_difference/` 下的 `modal_adapter.py`, `nominal_student_core.py` 等。
* `scripts/train_phycod_int.py` (旧版训练脚本)。

---

## 8. 后续代码说明书维护规则

每次新增、修改或删除脚本时，请在开发者日志中更新以下部分：

```md
### 代码说明书需增改的地方
- 新增：[文件名]，一句话说明作用与依赖。
- 修改：[文件名]，更新职责说明或所属阶段。
- 删除：[文件名]，说明退出主线的原因。
```

---

## 9. 2026-05-08 phase-gated late-phase 诊断更新

| 修改时间 | 涉及脚本/文件 | 需增改说明 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-08 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修改训练日志职责说明 | `compute_response_loss` 返回的 phase-gated 分解指标现在会在 `train_cases_grad_accum(...)`、`evaluate_cases(...)` 和 `training_history.csv` 中完整记录；新增记录 `phase_gate_mean/max/active_ratio`、`theta_fast_abs_max`、`theta_gated_fast_rms/abs_max` 等诊断列。 | phase-gated fast residual late-phase 诊断 |
| 2026-05-08 Asia/Shanghai | `scripts/analyze_phase_gated_results.py` | 新增结果诊断脚本说明 | 新增独立脚本读取 `training_history.csv`、`post_test` 下的 `metrics.csv` 与 `selected_timeseries.csv`，汇总 x/y 相对 static baseline 比例、phase gate 激活、`alpha_x/alpha_xy_gated_fast` RMS/max，并在存在 tip pred/teacher 响应列时计算 5s 后 late-window lag/corr/RMS 幅值比例；输出 `phase_gated_diagnosis_summary.csv` 和 `.json`。 | phase-gated fast residual late-phase 诊断 |

---

## 10. 2026-05-08 adaptive phase-window 与 complex phase loss 更新

| 修改时间 | 涉及脚本/文件 | 需增改说明 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-08 Asia/Shanghai | `src/student/transformer/frequency_losses.py` | 新增相位漂移自适应训练 loss | 新增 `adaptive_phase_window_loss(...)`，在整条响应上滑动扫描局部窗口，用 detached local lag/corr score 自动挖掘 phase-drift hard windows；对高分窗口施加可微 local lag loss、complex spectrum phase loss 和 RMS 幅值 guard，并在提供 `g_phase` 时返回 gate alignment loss。 | phase-gated fast residual 自主相位修正训练 |
| 2026-05-08 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修改训练入口与日志说明 | 新增 `--use-adaptive-phase-window-loss`、`--phase-window-*`、`--w-adaptive-phase-*`、`--w-complex-phase-*`、`--w-complex-amp-guard-*` 和 `--w-phase-gate-align`；新 loss 默认关闭，开启后并入 `freq_loss`，并将 adaptive window score、被选窗口起点、complex phase loss、gate alignment 等指标写入 `training_history.csv`。 | phase-gated fast residual 自主相位修正训练 |

---

## 11. 2026-05-09 adaptive phase gate target 标定更新

| 修改时间 | 涉及脚本/文件 | 需增改说明 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-09 Asia/Shanghai | `src/student/transformer/frequency_losses.py` | 修改 adaptive gate alignment 说明 | `adaptive_phase_window_loss(...)` 新增 `gate_target_score_ref`，将局部相位漂移 score 按可配置参考值映射到 `g_phase` 目标，避免 gate target 被压缩到过低导致 fast branch 难以局部打开。 | phase-gated fast residual 自主相位修正训练 |
| 2026-05-09 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修改训练入口与打印说明 | 新增命令行参数 `--phase-window-gate-score-ref`，并在 adaptive phase-window loss 调用和启动配置打印中使用；默认值为 `0.12`，用于增强 hard window 对 phase gate 的监督强度。 | phase-gated fast residual 自主相位修正训练 |

---

## 12. 2026-05-09 phase gate 诊断阈值与训练日志加速更新

| 修改时间 | 涉及脚本/文件 | 需增改说明 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-09 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修改 phase gate 诊断配置 | 新增 `--phase-gate-active-threshold`，用于控制 `phase_gate_active_ratio` 的统计阈值；默认仍为 `0.2` 以兼容旧日志，新一轮可显式设为 `0.10` 观察 gate 是否已在 high/complex 工况局部抬升。 | phase-gated fast residual 自主相位修正诊断 |
| 2026-05-09 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修改训练日志统计实现 | `train_cases_grad_accum(...)` 将 per-case 日志指标的累加、max/min 统计保留在当前 torch device 上，epoch 汇总时再转出，减少每个 case 后大量 `.cpu()` 标量同步；不改变 forward/backward、loss 权重、优化器 step 或 Newmark/core 数值路径。 | 训练速度优化 |

---

## 13. 2026-05-09 Newmark hot-path 安全加速更新

| 修改时间 | 涉及脚本/文件 | 需增改说明 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-09 Asia/Shanghai | `src/student/transformer/dynamic_physical_core_torch.py` | 修改 Newmark core hot path | 缓存固定的 Newmark 常数 `a0~a5`，新增 `newmark_step_fast(...)`，要求调用方已完成 device/dtype 转换，从而跳过每步重复 `torch.as_tensor`、shape 检查和 registry 拆分；线性求解仍使用原 `linear_solve_mode`，默认训练命令仍使用 `solve`。 | 训练速度优化 |
| 2026-05-09 Asia/Shanghai | `src/student/transformer/transformer_rollout_torch.py` | 修改 rollout 调用路径 | static rollout 中将整条 `theta_seq` 一次性转换到 core dtype/device，然后逐步调用 `physical_core.newmark_step_fast(...)`；不改变 `theta=[alpha_x_total, alpha_xy_total]` 接口、不改变 Newmark 方程、不改变 `core_dtype=float64` 要求。 | 训练速度优化 |

---

## 14. 2026-05-09 response no-regression guard 更新

| 修改时间 | 涉及脚本/文件 | 需增改说明 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-09 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 新增响应层 no-regression guard | 新增 `--use-no-regression-guard` 及 `--no-regression-*` 配置，仅对匹配关键词的 low/simple case 生效；guard 直接约束响应 ratio、局部 lag 和 RMS 幅值偏差，不约束 `g_phase` 本身，允许 fast branch 自由使用但不能恶化受保护工况。 | 高频/复杂相位强化与低频/simple 防退化 |
| 2026-05-09 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修改 best checkpoint 选择 | `--best-score-mode` 新增 `guarded_freq`，使用 `freq_loss + best_score_guard_weight * no_regression_guard_loss` 选择 best checkpoint，避免只看频域目标时保存到 low/simple 有副作用的模型。 | 高频/复杂相位强化与低频/simple 防退化 |

---

## 15. 2026-05-09 训练分段计时诊断更新

| 修改时间 | 涉及脚本/文件 | 需增改说明 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-09 Asia/Shanghai | `src/student/transformer/transformer_rollout_torch.py` | 新增 rollout 内部耗时 metadata | `TransformerRolloutConfig` 新增 `profile_timing` 与 `profile_timing_sync_cuda`；开启后在 `TransformerRolloutOutput.metadata` 中写入 `encoder_seconds`、`core_prepare_seconds`、`newmark_loop_seconds`、`state_stack_seconds`，用于判断 encoder、Newmark 积分和状态堆叠的相对耗时。 | 训练速度诊断 |
| 2026-05-09 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 新增训练计时 CLI 与 history 列 | 新增 `--profile-train-timing` 和 `--profile-timing-sync-cuda`；`training_history.csv` 新增 `train_/valid_timing_*` 列，覆盖 total、model forward、encoder、core prepare、Newmark loop、loss、backward、metric accumulation、grad clip、optimizer step。默认不启用，不影响既有训练结果。 | 训练速度诊断 |
| 2026-05-09 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修复训练 loss 返回链路 | 补回 `compute_response_loss(...)` 末尾的 `return result`，保证训练和验证循环拿到完整 loss/diagnostic 字典。 | 训练速度诊断 |
| 2026-05-09 Asia/Shanghai | `src/student/transformer/dynamic_physical_core_torch.py` | 新增 Newmark 内部细分计时接口 | 新增 `newmark_step_fast_timed(...)`，只在 profiling 路径使用，拆分记录 effective-matrix assembly、RHS build、linear solve、state update 耗时；不改变默认 fast Newmark 数值路径。 | Newmark 瓶颈细分诊断 |
| 2026-05-09 Asia/Shanghai | `src/student/transformer/transformer_rollout_torch.py`, `scripts/train_transformer_physical_params_torch.py` | 扩展训练计时列 | `training_history.csv` 新增 `train_/valid_timing_newmark_assemble_seconds`、`timing_newmark_rhs_seconds`、`timing_newmark_solve_seconds`、`timing_newmark_update_seconds`，用于判断 Newmark loop 具体瓶颈。 | Newmark 瓶颈细分诊断 |

---

## 16. 2026-05-10 auxiliary guard-only case 更新

| 修改时间 | 涉及脚本/文件 | 需增改说明 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-10 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 新增 simple/low 防退化辅助训练入口 | 新增 `--guard-load-files`、`--guard-case-paths`、`--w-guard-case-loss`。guard cases 可使用 simple/low 载荷，只反传 no-regression guard 与 theta/phase 正则，不进入 high/complex 的 adaptive phase 主目标，从而保护 simple/low 幅值与 x/y response。 | strong phase balanced v2 |
| 2026-05-10 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 扩展训练记录 | `training_history.csv` 新增 `train_guard_*` 指标，包括 guard case 数量、guard loss、x/y/tip_y/last5_y ratio 与 phase gate 统计，便于判断辅助保护是否真正生效。 | strong phase balanced v2 |

---

## 17. 2026-05-10 static-quality gate suppression 更新

| 修改时间 | 涉及脚本/文件 | 需增改说明 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-10 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 新增自动 gate 选择性训练项 | 新增 `--use-static-quality-gate-suppression`、`--w-static-good-gate-l1` 和 `--static-quality-*` 参数。训练时对局部窗口计算 static student 相对 teacher 的 corr、best-lag 和 RMS 幅值比；若 static 已经满足 good-window 条件，则对该窗口 `g_phase` 施加 L1 抑制，促使模型自动识别“不需要 fast phase correction”的窗口。 | automatic gate selectivity |
| 2026-05-10 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 扩展日志指标 | `training_history.csv` 新增 `static_quality_gate_loss`、`static_good_gate_mean`、`static_bad_gate_mean`、`static_gate_selectivity_gap`、`static_good_window_ratio` 等指标，用于检查 gate 是否从 case-name 保护转向基于响应质量的自动选择。 | automatic gate selectivity |
---

## 18. 2026-05-11 from-scratch loss curriculum update

| Time | Files | Required Note | Content | Stage |
|---|---|---|---|---|
| 2026-05-11 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Add configurable from-scratch curriculum | Add `--use-loss-curriculum` and `--curriculum-*` options. When enabled, the training loop builds an epoch-local config that ramps local lag/peak-time loss, adaptive/complex phase loss, high-frequency phase-drift loss, phase-gate regularization, static-quality gate suppression, and state-window no-regression weights without changing the model architecture, physical registry, or `theta=[alpha_x_total, alpha_xy_total]` core interface. | from-scratch frequency/phase curriculum |
| 2026-05-11 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Add curriculum diagnostics | `training_history.csv` now records curriculum scales and effective epoch weights such as `curriculum_phase_scale`, `curriculum_lag_scale`, `curriculum_guard_scale`, `effective_w_phase_drift_*`, `effective_w_phase_gate_l1`, and `effective_w_state_no_regression_*`; checkpoints also store the base config, effective epoch config, and curriculum metrics. | from-scratch frequency/phase curriculum |
---

## 19. 2026-05-12 amplitude-aware high-frequency phase update

| Time | Files | Required Note | Content | Stage |
|---|---|---|---|---|
| 2026-05-12 Asia/Shanghai | `src/student/transformer/frequency_losses.py` | Add amplitude-aware phase weighting | Add detached response-amplitude emphasis for local phase windows. `phase_drift_rate_loss(...)` can now multiply high-frequency window weights by an amplitude multiplier, and `adaptive_phase_window_loss(...)` can include amplitude in hard-window scoring, so large-amplitude high-frequency windows receive stronger phase alignment pressure without changing physical parameters. | high-amplitude high-frequency phase correction |
| 2026-05-12 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Expose amplitude-aware training controls | Add `--phase-window-amplitude-*` and `--phase-drift-amplitude-*` CLI/config fields, including optional x/y-specific amplitude references, and log `adaptive_*_amplitude_weight_*`, `phase_drift_*_amplitude_weight_mean`, and `phase_drift_*_combined_weight_mean` to `training_history.csv`. Defaults preserve previous behavior. | high-amplitude high-frequency phase correction |
| 2026-05-12 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Fix amplitude-aware phase-drift diagnostics | Add the missing `phase_drift_x/y_amplitude_weight_mean` and `phase_drift_x/y_combined_weight_mean` entries to the per-case loss dictionary, and make diagnostic metric aggregation fall back to zero for absent optional metrics. This only affects logging robustness and does not change the physical parameter interface or loss values. | high-amplitude high-frequency phase correction |

---

## 20. 2026-05-12 static-failure-aware phase weighting update

| Time | Files | Required Note | Content | Stage |
|---|---|---|---|---|
| 2026-05-12 Asia/Shanghai | `src/student/transformer/frequency_losses.py` | Add static-failure-aware phase weighting | Add a detached local-window weight based on whether the static corrected student fails against teacher in lag, correlation, or RMS amplitude. `adaptive_phase_window_loss(...)` and `phase_drift_rate_loss(...)` can now emphasize phase correction only where the static baseline is locally bad, instead of relying only on teacher response amplitude. Defaults preserve previous behavior. | static-failure-driven phase correction |
| 2026-05-12 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Expose static-failure-aware controls | Add `--phase-window-static-failure-weight`, `--phase-window-static-failure-max-weight`, `--phase-drift-static-failure-weight`, and `--phase-drift-static-failure-max-weight`. The thresholds reuse `static_quality_*` settings and history now records `adaptive_*_static_failure_weight_*` and `phase_drift_*_static_failure_weight_mean`. The model architecture and `theta=[alpha_x_total, alpha_xy_total]` interface are unchanged. | static-failure-driven phase correction |

---

## 21. 2026-05-12 slow-only branch diagnosis update

| Time | Files | Required Note | Content | Stage |
|---|---|---|---|---|
| 2026-05-12 Asia/Shanghai | `src/student/transformer/transformer_rollout_torch.py` | Add supplied-theta rollout helper | Add `rollout_with_theta_sequence(...)` so training diagnostics can run the physical Newmark core with an externally supplied theta sequence such as `theta_slow`. This keeps the dynamic core interface unchanged and still uses `theta[:,t,:]` for each Newmark step. | slow/fast phase-gate separation |
| 2026-05-12 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Add slow-only branch diagnosis loss | Add optional `--use-slow-only-branch-diagnosis`. When enabled, training runs a detached slow-only rollout with `theta_slow`, uses slow-vs-teacher local lag/corr/amplitude quality to decide whether the fast branch should be suppressed or allowed, and logs `slow_*` diagnostics. The total physical theta passed to the main rollout remains `[alpha_x_total, alpha_xy_total]`. | slow/fast phase-gate separation |

---

## 22. 2026-05-13 remove case-name guard paths

| Time | Files | Required Note | Content | Stage |
|---|---|---|---|---|
| 2026-05-13 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Remove case-name/manual guard controls | Remove the case-name keyword no-regression guard, explicit guard-only case inputs, and simple-case gate L1 regularizer. Deleted CLI/config entries include `--use-no-regression-guard`, `--no-regression-guard-case-keywords`, `--w-no-regression-*`, `--guard-load-files`, `--guard-case-paths`, `--w-guard-case-loss`, and `--w-phase-gate-simple-l1`. | state-driven phase/frequency generalization |
| 2026-05-13 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Preserve automatic protection mechanisms | Keep state-window no-regression, static-quality gate suppression, and slow-only branch diagnosis. `guarded_freq` checkpoint scoring now uses `state_no_regression_guard_loss`, so protection is driven by local response quality rather than case names or manually selected guard cases. | state-driven phase/frequency generalization |

---

## 23. 2026-05-13 local-band phase loss update

| Time | Files | Required Note | Content | Stage |
|---|---|---|---|---|
| 2026-05-13 Asia/Shanghai | `src/student/transformer/frequency_losses.py` | Add local narrow-band phase objective | Add `local_band_phase_loss(...)`, a sliding-window FFT phase-alignment loss. It compares target-dominant high-frequency band phase using `1 - cos(phase_pred - phase_teacher)` and weights windows by detached teacher high-band content, so activation is based on local spectral state rather than case names. | high-frequency x/y phase correction |
| 2026-05-13 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Expose local-band phase controls | Add `--use-local-band-phase-loss`, `--w-local-band-phase-x/y`, and `--local-band-phase-*` CLI/config fields. The loss is added to `freq_loss`, logs `local_band_phase_*` diagnostics to `training_history.csv`, and leaves the dynamic core and final `theta=[alpha_x_total, alpha_xy_total]` interface unchanged. | high-frequency x/y phase correction |

---

## 24. 2026-05-14 phase-gate bootstrap and delayed-best update

| Time | Files | Required Note | Content | Stage |
|---|---|---|---|---|
| 2026-05-14 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Add delayed best-checkpoint selection | Add `--best-start-epoch`. Validation still runs and is logged before this epoch, but `best_transformer_physical_params.pt` cannot be updated and LR/early-stop patience is not consumed before the allowed epoch. This avoids saving an early closed-gate checkpoint during phase/gate curricula. | phase-gated fast branch startup |
| 2026-05-14 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Add optional phase-gate bootstrap regularizer | Add `--w-phase-gate-bootstrap`, `--phase-gate-bootstrap-target`, and `--phase-gate-bootstrap-end-epoch`. When enabled, early epochs can softly penalize `mean(g_phase)` below a low target so the fast residual branch can receive gradient before later automatic gate suppression/guard terms take over. Defaults are zero, so existing commands keep prior behavior and the final physical core still receives only total `theta=[alpha_x_total, alpha_xy_total]`. | phase-gated fast branch startup |
| 2026-05-14 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Extend diagnostics | `training_history.csv` now records `best_start_epoch`, `best_epoch_allowed`, `effective_w_phase_gate_bootstrap`, `phase_gate_bootstrap_loss`, and `phase_gate_bootstrap_deficit` for train/valid diagnostics. | phase-gated fast branch startup |

---

## 25. 2026-05-14 local phase-correlation loss update

| Time | Files | Required Note | Content | Stage |
|---|---|---|---|---|
| 2026-05-14 Asia/Shanghai | `src/student/transformer/frequency_losses.py` | Add amplitude-invariant phase objective | Add `local_phase_correlation_loss(...)`. It scans local response windows, standardizes prediction/teacher signals, maximizes zero-lag correlation, and penalizes the gap between best-lag correlation and zero-lag correlation. All windows can contribute while high-frequency and static-failure windows receive detached automatic emphasis; no case-name logic is used. | frequency/phase-first alignment |
| 2026-05-14 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Expose local phase-correlation controls | Add `--use-local-phase-corr-loss`, `--w-local-phase-corr-x/y`, and `--local-phase-corr-*` controls. The loss is added to `freq_loss` and is intended for phase/frequency alignment when amplitude accuracy is secondary. The physical registry and final `theta=[alpha_x_total, alpha_xy_total]` Newmark interface remain unchanged. | frequency/phase-first alignment |
| 2026-05-14 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Extend diagnostics | `training_history.csv` now records `local_phase_corr_*` train/valid diagnostics, including zero-lag correlation, best-lag correlation, correlation gap, local lag, high-frequency weight, static-failure weight, and window counts. | frequency/phase-first alignment |

---

## 26. 2026-05-14 local phase-increment consistency update

| Time | Files | Required Note | Content | Stage |
|---|---|---|---|---|
| 2026-05-14 Asia/Shanghai | `src/student/transformer/frequency_losses.py` | Add cumulative-drift phase objective | Add `local_phase_increment_loss(...)`. For each local window it selects the target-dominant frequency bin in the configured band, compares prediction/teacher complex phase, then compares adjacent-window phase advances. The absolute phase term helps local alignment, and the increment term directly targets accumulated phase drift. The loss uses state/spectrum weights and does not rely on case names. | cumulative high-frequency phase correction |
| 2026-05-14 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Expose phase-increment controls | Add `--use-local-phase-increment-loss`, `--w-local-phase-absolute-x/y`, `--w-local-phase-increment-x/y`, and `--local-phase-increment-*` CLI/config fields. The weighted terms are added to `freq_loss` without changing model architecture, physical registry, or the final `theta=[alpha_x_total, alpha_xy_total]` interface consumed by Newmark. | cumulative high-frequency phase correction |
| 2026-05-14 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Extend diagnostics | `training_history.csv` now records `local_phase_increment_*` train/valid diagnostics, including absolute/increment losses, phase cosine, increment cosine, high-frequency weight, static-failure weight, combined weight, target-frequency mean, and window/increment counts. | cumulative high-frequency phase correction |

---

## 27. 2026-05-15 continuous phase-lag update

| Time | Files | Required Note | Content | Stage |
|---|---|---|---|---|
| 2026-05-15 Asia/Shanghai | `src/student/transformer/frequency_losses.py` | Add time-shift-aware phase objective | Add `local_continuous_phase_lag_loss(...)`. It projects each local window onto a dense continuous frequency grid instead of coarse FFT bins, uses teacher-band energy to weight frequencies automatically, and converts phase mismatch to an equivalent time-shift loss scaled by `--continuous-phase-time-shift-scale-seconds`. This directly targets residual high-frequency lag such as 0.02-0.06 s. | time-shift-aware high-frequency phase correction |
| 2026-05-15 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Expose continuous phase-lag controls | Add `--use-continuous-phase-lag-loss`, `--w-continuous-phase-absolute-x/y`, `--w-continuous-phase-time-shift-x/y`, and `--continuous-phase-*` CLI/config fields. The loss is added to `freq_loss` and can be limited to high-frequency/state-failure windows through automatic spectral/static weights, with no case-name logic and no change to `theta=[alpha_x_total, alpha_xy_total]`. | time-shift-aware high-frequency phase correction |
| 2026-05-15 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Extend diagnostics | `training_history.csv` now records `continuous_phase_*` diagnostics, including absolute/time-shift losses, phase cosine, equivalent absolute lag in seconds, high-frequency/static-failure/combined weights, target frequency mean, and window counts. | time-shift-aware high-frequency phase correction |

---

## 28. 2026-05-15 phase-slope and fast-mean separation update

| Time | Files | Required Note | Content | Stage |
|---|---|---|---|---|
| 2026-05-15 Asia/Shanghai | `src/student/transformer/frequency_losses.py` | Add phase-slope drift objective | Add `local_phase_slope_loss(...)`. It projects adjacent local windows onto a dense frequency grid, forms the prediction-vs-teacher phase-error unit for each window, and penalizes the window-to-window phase-error increment as an equivalent time-lag slope. This is intended to target small equivalent-frequency errors that accumulate into high-frequency late-time phase drift. Window emphasis remains automatic from local spectrum/static-failure state and does not use case names. | cumulative high-frequency frequency/phase correction |
| 2026-05-15 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Expose phase-slope controls | Add `--use-local-phase-slope-loss`, `--w-local-phase-slope-x/y`, and `--local-phase-slope-*` CLI/config fields. The x/y slope terms are added to `freq_loss`, can be curriculum-scaled by the existing phase curriculum, and write `local_phase_slope_*` train/valid diagnostics to `training_history.csv`. | cumulative high-frequency frequency/phase correction |
| 2026-05-15 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Add fast-branch mean separation regularizer | Add optional sliding-window mean regularization for raw and gated fast branches: `--w-theta-fast-window-mean`, `--w-theta-gated-fast-window-mean`, `--theta-fast-window-mean-seconds`, and `--theta-fast-window-mean-stride-seconds`. These losses discourage `alpha_fast(t)` / `g_phase(t)*alpha_fast(t)` from carrying cycle-scale mean stiffness changes, so average frequency calibration is biased toward `alpha_slow(t)` while the full-order Newmark core still receives only total `theta=[alpha_x_total, alpha_xy_total]`. | slow/fast phase-gate separation |

---

## 29. 2026-05-15 alpha-stage best-result baseline

| Time | Files | Required Note | Content | Stage |
|---|---|---|---|---|
| 2026-05-15 Asia/Shanghai | `developer_log.md`, `code_doc.md` | Record best alpha-class baseline | Record `result/train_gateopen_v2_fastnewmark` as the current best alpha-only physical-parameter result for frequency/phase alignment. If later attempts still cannot reduce the high-frequency late-time phase error while protecting simple/low/mid cases, this run should be treated as the best result for the first parameter class (`alpha_x`, `alpha_xy`). | alpha_x/alpha_xy stage baseline |
| 2026-05-15 Asia/Shanghai | `result/train_gateopen_v2_fastnewmark` | Best-result metrics | Best checkpoint is epoch 160 with `best_score=2.286574e-02`. On `freq_high_test_01_fullspan_xy_holdout_f0p650`, the transformer reached `full_x_mse_ratio_to_static=0.0379`, `full_y_mse_ratio_to_static=0.0676`, and late-window best-correlation lag about `+0.030 s` for both x and y. `simple_tip_fx_case` remained acceptable with x ratio `0.9756`, and `simple_tip_fy_case` y ratio was `1.0089`. | alpha_x/alpha_xy stage baseline |
| 2026-05-15 Asia/Shanghai | `result/train_gateopen_v2_fastnewmark/train_config.yaml` | Reproducible training setup | The run used 24 train cases and 4 validation cases from `tc_adaptphase_gateboost_complex_v1`: low/mid/high frequency-sweep cases plus `train_complex_case_3-8`, with low/mid/high holdout and `val_complex_case_1` for validation. Key settings were `enabled_params=alpha_x,alpha_xy`, phase-gated decomposition, `temporal_window_size=192`, `d_model=64`, 4 spatial heads, 4 temporal heads, 2 temporal layers, `epochs=160`, `lr=1e-4`, `linear_solve_mode=solve`, `fast_core_precompute_newmark=true`, `w_spec_x/y=0.5`, `w_peak_x/y=0.2`, `w_lag_x/y=0.05`, `lag_window_seconds=2.56`, `lag_stride_seconds=1.28`, `max_lag_seconds=0.8`, `use_adaptive_phase_window_loss=true`, `w_adaptive_phase_x/y=0.06/0.10`, `w_complex_phase_x/y=0.05/0.10`, `w_phase_gate_align=0.20`, `phase_fast_scale=1.0`, and `phase_gate_init_bias=-2.5`. | alpha_x/alpha_xy stage baseline |

---

## 30. 2026-05-18 beta damping parameter and four-curve evaluation update

| Time | Files | Required Note | Content | Stage |
|---|---|---|---|---|
| 2026-05-18 Asia/Shanghai | `src/student/transformer/physical_parameter_registry.py` | Add beta damping physical parameters | Add `beta_damp_x` and `beta_damp_y` after the existing `alpha_x,alpha_xy` registry entries. Both target `C` rather than `K`, use bounded scalar heads, and map to `C_x_template` / `C_y_template`, preserving the old alpha slices for partial checkpoint loading. | alpha+beta amplitude correction |
| 2026-05-18 Asia/Shanghai | `src/student/transformer/physical_templates.py`, `src/student/transformer/dynamic_physical_core_torch.py` | Add dynamic damping templates and C-effective Newmark support | Build directional damping templates as `C_x/C_y = damping_scale * K_x/K_y` and extend the torch Newmark core so dynamic damping contributes to both `A = K_eff + a0*M + a1*C_eff` and the RHS damping term. `theta=0` still exactly recovers the static alpha baseline core. | physically meaningful beta damping |
| 2026-05-18 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Add alpha checkpoint warm-start for wider beta heads | Replace strict-only initialization with partial checkpoint merging. Exact-shape tensors load normally; widened theta-head tensors copy the leading alpha rows and leave new beta rows at fresh near-zero initialization. Add `--freeze-alpha-backbone-for-beta` to freeze the loaded encoder/Transformer/alpha rows and train only C-target beta output rows. | conservative beta smoke training |
| 2026-05-18 Asia/Shanghai | `scripts/evaluate_transformer_vs_baselines.py` | Add alpha-only versus alpha+beta comparison plots | Add `--alpha-checkpoint`, `--alpha-label`, and `--checkpoint-label`. When an alpha checkpoint is supplied, evaluation runs both alpha-only and alpha+beta checkpoints and saves four-curve plots for teacher, base student, alpha-only, and alpha+beta on tip x/y and last5 x/y responses and errors. | beta result interpretation |
| 2026-05-18 Asia/Shanghai | `.gitignore` | Exclude local data/results from version control | Add ignore rules for `data/`, `result/`, `results/`, Python caches, virtual environments, and local editor files so Git commits on `add_beta` contain code and documentation but not large benchmark artifacts. | repository hygiene |
