# 幅值类参数项目对接说明

> 目标：把当前 PhyCoD alpha 阶段项目拆出一个副本，用于在另一个对话框中开发幅值类 / beta 类物理参数。当前对话框继续负责 `alpha_benchmark` 主线训练和后续消融实验。

---

## 1. 当前状态

当前主项目路径：

```text
D:\phycod_codex
```

建议幅值类参数副本路径：

```text
D:\phycod_amp_project
```

当前正在运行的正式主线 benchmark：

```text
MAIN-ALPHA / main_full_v1
```

对应输出目录：

```text
D:\phycod_codex\result\alpha_benchmark\main_full_v1
```

统一 benchmark 数据集：

```text
D:\phycod_codex\data\load\alpha_benchmark
D:\phycod_codex\result\alpha_benchmark\training_cases
```

重要提醒：

```text
当前 MAIN-ALPHA benchmark 尚未运行完成。
因此现在不要把 main_full_v1 的 best checkpoint 当作最终 alpha baseline。
另一个幅值类项目可以先复制代码、数据集和 prepared npz，用于代码开发；
等 MAIN-ALPHA 完成后，再同步 best checkpoint、training_history、training_summary 和 post_test 结果。
```

---

## 2. 两个项目的分工

### 2.1 当前对话框继续负责

当前对话框继续做：

```text
alpha_benchmark 主线训练结果分析
MAIN-ALPHA 结果写入 experiment_record.md
alpha 类参数消融实验
phase gate / loss / branch / physical core ablation
论文中 alpha 阶段实验结果整理
```

不要在当前对话框中开始大规模修改幅值类参数结构，以免影响正在进行的消融实验。

### 2.2 新对话框负责

新对话框负责：

```text
幅值类 / beta 类物理参数设计
参数注册表扩展
physical templates 扩展
dynamic physical core 接口扩展
训练脚本支持从 alpha best checkpoint 继续训练
幅值类参数的 quick probe / formal training
```

建议新项目输出目录不要与 alpha 消融混用：

```text
result/alpha_benchmark_amp/
```

或：

```text
result/amplitude_benchmark/
```

---

## 3. 当前可以先复制的文件

在 `MAIN-ALPHA` 未完成前，可以先复制以下内容到新项目，用于幅值类参数代码开发。

### 3.1 代码目录

必须复制：

```text
src/
scripts/
configs/
cases/
```

原因：

```text
src/      包含 student、transformer、physical core、parameter registry 等核心代码
scripts/  包含训练、评估、数据准备脚本
configs/  包含 blade 和路径配置
cases/    包含部分结构/气动 case 配置
```

### 3.2 原始物理数据

必须复制：

```text
data/raw/nrel5mw/blade_master.csv
data/raw/reference_cases/beamdyn/nrel5mw/
```

用途：

```text
blade_master.csv 用于构建 full-order blade student
reference_cases/beamdyn/nrel5mw/ 用于准备或复现 BeamDyn teacher cases
```

### 3.3 统一 benchmark 数据

必须复制：

```text
data/load/alpha_benchmark/
result/alpha_benchmark/training_cases/
```

说明：

```text
data/load/alpha_benchmark/ 是统一载荷数据
result/alpha_benchmark/training_cases/ 是已经准备好的 train/valid npz
幅值类参数后续应基于同一个 benchmark 继续训练，避免换数据集导致结果不可比
```

### 3.4 文档

建议复制：

```text
developer_log.md
code_doc.md
paper_outline_refined.md
ablation_experiment_plan.md
paper_draft_sections_3_4_5_cn.md
amplitude_project_handoff.md
```

用途：

```text
developer_log.md  记录代码演化和历史最佳机制
code_doc.md       帮助另一个对话框理解代码结构
paper_outline_refined.md 论文结构和方法定位
ablation_experiment_plan.md 统一 benchmark 和消融实验规划
paper_draft_sections_3_4_5_cn.md 当前中文论文初稿
amplitude_project_handoff.md 本对接说明
```

---

## 4. 当前不要复制或不要作为最终依据的内容

因为 `MAIN-ALPHA` benchmark 还在运行中，以下文件现在可能不完整：

```text
result/alpha_benchmark/main_full_v1/best_transformer_physical_params.pt
result/alpha_benchmark/main_full_v1/last_transformer_physical_params.pt
result/alpha_benchmark/main_full_v1/training_history.csv
result/alpha_benchmark/main_full_v1/training_summary.json
result/alpha_benchmark/main_full_v1/post_test/
```

当前不要把它们作为最终 alpha baseline。

可以等训练结束后再复制。

---

## 5. 如何判断 MAIN-ALPHA 已经跑完

在当前主项目中执行：

```powershell
cd D:\phycod_codex

Get-Process python | Select-Object Id,CPU,StartTime,Path

Test-Path result/alpha_benchmark/main_full_v1/training_summary.json
Test-Path result/alpha_benchmark/main_full_v1/best_transformer_physical_params.pt
Test-Path result/alpha_benchmark/main_full_v1/training_history.csv
```

如果训练进程已经结束，并且以下文件存在，说明可以同步结果：

```text
training_summary.json
training_history.csv
train_config.yaml
best_transformer_physical_params.pt
last_transformer_physical_params.pt
post_test/
```

还需要检查训练日志最后是否出现：

```text
[6/6] Done
```

以及 post-test 是否正常执行。

---

## 6. 复制到新项目的具体命令

### 6.1 当前阶段：先复制代码、数据和文档

在 PowerShell 中执行：

```powershell
$SRC = "D:\phycod_codex"
$DST = "D:\phycod_amp_project"

New-Item -ItemType Directory -Force $DST | Out-Null

robocopy "$SRC\src" "$DST\src" /E /XD __pycache__
robocopy "$SRC\scripts" "$DST\scripts" /E /XD __pycache__
robocopy "$SRC\configs" "$DST\configs" /E
robocopy "$SRC\cases" "$DST\cases" /E

New-Item -ItemType Directory -Force "$DST\data\raw\nrel5mw" | Out-Null
Copy-Item "$SRC\data\raw\nrel5mw\blade_master.csv" "$DST\data\raw\nrel5mw\blade_master.csv" -Force

robocopy "$SRC\data\raw\reference_cases\beamdyn\nrel5mw" "$DST\data\raw\reference_cases\beamdyn\nrel5mw" /E

robocopy "$SRC\data\load\alpha_benchmark" "$DST\data\load\alpha_benchmark" /E
robocopy "$SRC\result\alpha_benchmark\training_cases" "$DST\result\alpha_benchmark\training_cases" /E

Copy-Item "$SRC\developer_log.md" "$DST\developer_log.md" -Force
Copy-Item "$SRC\code_doc.md" "$DST\code_doc.md" -Force
Copy-Item "$SRC\paper_outline_refined.md" "$DST\paper_outline_refined.md" -Force
Copy-Item "$SRC\ablation_experiment_plan.md" "$DST\ablation_experiment_plan.md" -Force
Copy-Item "$SRC\paper_draft_sections_3_4_5_cn.md" "$DST\paper_draft_sections_3_4_5_cn.md" -Force
Copy-Item "$SRC\amplitude_project_handoff.md" "$DST\amplitude_project_handoff.md" -Force
```

说明：

```text
robocopy 返回码非 0 不一定代表失败。
如果复制文件数正常，通常可以继续。
```

### 6.2 MAIN-ALPHA 跑完后：同步最终 alpha baseline

等当前 benchmark 结束后再执行：

```powershell
$SRC = "D:\phycod_codex"
$DST = "D:\phycod_amp_project"

New-Item -ItemType Directory -Force "$DST\result\alpha_benchmark\main_full_v1" | Out-Null

Copy-Item "$SRC\result\alpha_benchmark\main_full_v1\best_transformer_physical_params.pt" "$DST\result\alpha_benchmark\main_full_v1\" -Force
Copy-Item "$SRC\result\alpha_benchmark\main_full_v1\last_transformer_physical_params.pt" "$DST\result\alpha_benchmark\main_full_v1\" -Force
Copy-Item "$SRC\result\alpha_benchmark\main_full_v1\train_config.yaml" "$DST\result\alpha_benchmark\main_full_v1\" -Force
Copy-Item "$SRC\result\alpha_benchmark\main_full_v1\training_history.csv" "$DST\result\alpha_benchmark\main_full_v1\" -Force
Copy-Item "$SRC\result\alpha_benchmark\main_full_v1\training_summary.json" "$DST\result\alpha_benchmark\main_full_v1\" -Force
Copy-Item "$SRC\result\alpha_benchmark\main_full_v1\experiment_record.md" "$DST\result\alpha_benchmark\main_full_v1\" -Force

robocopy "$SRC\result\alpha_benchmark\main_full_v1\post_test" "$DST\result\alpha_benchmark\main_full_v1\post_test" /E
```

可选复制 spectral disk cache：

```powershell
robocopy "$SRC\result\alpha_benchmark\cache_fs_main_full_v1" "$DST\result\alpha_benchmark\cache_fs_main_full_v1" /E
```

是否复制 cache 的建议：

```text
如果硬盘空间充足，建议复制，可以节省新项目后续启动时间。
如果空间紧张，可以不复制，后续训练会重新生成 cache。
```

---

## 7. 新项目检查命令

复制完成后，在新项目执行：

```powershell
cd D:\phycod_amp_project

Get-ChildItem src
Get-ChildItem scripts

(Get-ChildItem result/alpha_benchmark/training_cases/train -Recurse -Filter "*_training_case.npz").Count
(Get-ChildItem result/alpha_benchmark/training_cases/valid -Recurse -Filter "*_training_case.npz").Count

Test-Path data/raw/nrel5mw/blade_master.csv
Test-Path data/raw/reference_cases/beamdyn/nrel5mw
Test-Path data/load/alpha_benchmark
```

等 MAIN-ALPHA 结束后，同步结果后再检查：

```powershell
Test-Path result/alpha_benchmark/main_full_v1/best_transformer_physical_params.pt
Test-Path result/alpha_benchmark/main_full_v1/training_summary.json
Test-Path result/alpha_benchmark/main_full_v1/training_history.csv
Test-Path result/alpha_benchmark/main_full_v1/post_test
```

---

## 8. 另一个对话框需要优先阅读的文件

建议按照以下顺序阅读：

```text
1. amplitude_project_handoff.md
2. code_doc.md
3. developer_log.md
4. ablation_experiment_plan.md
5. paper_outline_refined.md
```

然后重点查看以下代码文件：

```text
src/student/transformer/physical_parameter_registry.py
src/student/transformer/physical_templates.py
src/student/transformer/physical_parameter_heads.py
src/student/transformer/dynamic_physical_core_torch.py
src/student/transformer/transformer_rollout_torch.py
src/student/transformer/spatiotemporal_physics_encoder.py
scripts/train_transformer_physical_params_torch.py
scripts/evaluate_transformer_vs_baselines.py
```

---

## 9. 幅值类参数开发的建议方向

当前 alpha 类参数：

```text
alpha_x(t)
alpha_xy(t)
```

当前物理接口：

```text
K_eff(t) = K0
         + alpha_x(t)  K_x_template
         + alpha_xy(t) K_xy_template
```

当前 alpha 类参数主要用于：

```text
频率对齐
相位对齐
x-y coupling response 修正
```

幅值类参数建议不要直接破坏 alpha 机制，而是在 alpha best checkpoint 基础上新增 beta 类参数。例如：

```text
beta_damp_x(t)
beta_damp_y(t)
beta_force_scale_x(t)
beta_force_scale_y(t)
beta_modal_amp(t)
beta_rot_coupling(t)
```

具体采用哪一种，需要先结合现有响应误差判断：

```text
如果频率/相位已经较好，但峰值幅值偏小或偏大，可以考虑 damping / force-scale / modal amplitude residual。
如果 y 方向耦合响应幅值异常，可以考虑 coupling amplitude residual。
如果相位和幅值都受影响，需要谨慎区分刚度、阻尼和外力等效误差。
```

强烈建议：

```text
不要一开始加入很多 beta 参数。
先做一个最小 beta 参数，验证它是否能改善幅值且不破坏 alpha 已取得的频率/相位对齐。
```

---

## 10. 新增幅值类参数时必须注意的代码问题

### 10.1 Checkpoint 兼容

幅值类参数会改变模型 head 或 registry 维度，因此旧 alpha checkpoint 很可能不能 strict load。

需要实现或使用 partial loading：

```text
复用 encoder / Transformer / alpha head 中能对齐的权重
新增 beta 参数相关权重从 0 或小值初始化
不要 strict=True 强制加载全部参数
```

建议另一个项目优先检查：

```text
scripts/train_transformer_physical_params_torch.py 中 init-checkpoint 加载逻辑
```

### 10.2 Physical core 接口

如果 beta 参数进入：

```text
C_eff(t)
F_eff(t)
modal amplitude correction
rotational coupling residual
```

则需要明确：

```text
beta 参数作用在哪里
是否仍保持 full-order physical rollout
是否会破坏当前 alpha 的 K_eff 接口
是否能在 evaluate 脚本中正确保存 timeseries
```

### 10.3 训练目标

幅值类参数不能只看 MSE。需要同时保留：

```text
frequency / phase guard
alpha no-regression check
amplitude metrics
```

否则 beta 参数可能通过牺牲频率和相位来降低幅值误差。

---

## 11. 给另一个对话框的开场说明

可以直接复制下面这段发给另一个对话框：

```text
这是 PhyCoD 项目的幅值类 / beta 类参数开发副本。

当前主项目 D:\phycod_codex 正在运行 MAIN-ALPHA / main_full_v1 benchmark，尚未完成。
因此现在副本中可能还没有最终 alpha best checkpoint。

当前副本可以先做幅值类参数的代码设计和实现。
等主项目 MAIN-ALPHA 跑完后，我会把以下文件同步过来：

result/alpha_benchmark/main_full_v1/best_transformer_physical_params.pt
result/alpha_benchmark/main_full_v1/last_transformer_physical_params.pt
result/alpha_benchmark/main_full_v1/train_config.yaml
result/alpha_benchmark/main_full_v1/training_history.csv
result/alpha_benchmark/main_full_v1/training_summary.json
result/alpha_benchmark/main_full_v1/experiment_record.md
result/alpha_benchmark/main_full_v1/post_test/

幅值类参数必须基于统一 benchmark：

data/load/alpha_benchmark
result/alpha_benchmark/training_cases

当前 alpha 机制：
- alpha_x(t), alpha_xy(t)
- phase-gated slow-fast decomposition
- K_eff(t) = K0 + alpha_x*K_x_template + alpha_xy*K_xy_template
- response 由 full-order MCK/Newmark physical core 生成

开发目标：
在不破坏 alpha 已取得的频率/相位修正基础上，添加最小幅值类 beta 参数，改善幅值响应。

注意：
新增 beta 参数后，旧 alpha checkpoint 不能 strict load。
需要 partial checkpoint loading，复用 encoder/Transformer/已有 alpha 权重，新 beta 参数从 0 或小值初始化。
```

---

## 12. MAIN-ALPHA 完成后的同步提醒

当当前 benchmark 跑完后，需要做三件事：

1. 在当前对话框中分析 `MAIN-ALPHA` 结果，并更新：

```text
result/alpha_benchmark/main_full_v1/experiment_record.md
```

2. 将最终 alpha baseline 文件复制到幅值类项目：

```text
best_transformer_physical_params.pt
last_transformer_physical_params.pt
train_config.yaml
training_history.csv
training_summary.json
post_test/
```

3. 通知另一个对话框：

```text
MAIN-ALPHA 已完成，最终 alpha baseline 已同步，可以开始基于 best checkpoint 做幅值类参数训练。
```

