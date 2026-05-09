

# Full-Order Corrected Student 项目代码说明书（阶段 1 起始版）

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
