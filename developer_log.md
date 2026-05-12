| 项目   | 内容                                                                                                                                                                                                                                            |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 修改时间 | 2026-04-25 11:41 Asia/Singapore                                                                                                                                                                                                               |
| 涉及脚本 | `src/student/full_order_corrected_core_torch.py`                                                                                                                                                                                              |
| 增改类型 | 新增                                                                                                                                                                                                                                            |
| 修改内容 | 新增 full-order torch corrected core；直接处理 `u_t / v_t / a_t / F_t1`；复刻 full-order Newmark-beta 推进；预留 `delta_M / delta_C / delta_K_mat / delta_K_geo / force_correction` 接口但当前全部返回零；不依赖 `ModalAdapter`、`CorrectedPhysicsCoreTorch`、`q/qdot/qddot` |
| 所属阶段 | 阶段 1：搭建 Full-Order Corrected Core 骨架                                                                                                                                                                                                          |
| 当前目的 | 建立可与 direct student baseline 做零修正一致性验证的新 core                                                                                                                                                                                                 |
| 项目   | 内容                                                                                                                                                    |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 修改时间 | 2026-04-25                                                                                                                                            |
| 涉及脚本 | `tests/compare_full_corrected_core_vs_student.py`                                                                                                     |
| 增改类型 | 移动                                                                                                                                                    |
| 修改内容 | 建议将原 `scripts/compare_full_corrected_core_vs_student.py` 移动到 `tests/compare_full_corrected_core_vs_student.py`，作为单工况 zero-correction 对齐测试脚本；脚本逻辑不需要改变 |
| 所属阶段 | 阶段 1：搭建 Full-Order Corrected Core 骨架                                                                                                                  |
| 当前目的 | 按新的项目约定，把测试脚本从 `scripts/` 归档到与 `src/`、`scripts/` 同级的 `tests/` 文件夹                                                                                     |
| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 12:18 Asia/Singapore |
| 涉及脚本 | `src/student/full_order_corrected_core_torch.py` |
| 增改类型 | 修改 |
| 修改内容 | 新增 `theta_full` 显式字典接口解析函数；新增 batch vector/matrix/diag matrix 校验与转换函数；将 `build_delta_K_mat(...)` 从固定返回零扩展为支持 `delta_K_mat`、`delta_K_diag`、`delta_K_relative_diag` 三种显式刚度修正输入；零修正 `theta_full=None` 仍返回零矩阵 |
| 所属阶段 | 阶段 1：搭建 Full-Order Corrected Core 骨架 |
| 当前目的 | 验证 full-order core 的刚度修正接口能真实进入 PDE 单步推进，同时保持零修正回退 direct baseline |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 12:18 Asia/Singapore |
| 涉及脚本 | `tests/test_full_order_delta_k_sanity.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 delta_K 接口 sanity test；先验证 `theta_full=None` 时与 `run_student_case(...)` 对齐，再传入 `theta_full={"delta_K_relative_diag": ...}` 检查非零刚度修正是否导致响应发生可检测变化 |
| 所属阶段 | 阶段 1：搭建 Full-Order Corrected Core 骨架 |
| 当前目的 | 在不引入网络、不引入 modal/reduced 变量的前提下，验证 full-order 修正接口可用性 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 Asia/Singapore |
| 涉及脚本 | `tests/test_full_order_delta_k_sanity.py` |
| 增改类型 | 修改 |
| 修改内容 | 新增 `make_json_safe(...)` 工具函数，并修改 `save_json(...)`，在保存 report 前递归转换 `np.bool_`、`np.integer`、`np.floating`、`np.ndarray`、`torch.Tensor`、`Path` 等对象，避免 JSON 序列化失败 |
| 所属阶段 | 阶段 1：搭建 Full-Order Corrected Core 骨架 |
| 当前目的 | 修复 delta_K sanity <br/test 已通过但保存 JSON report 时因 numpy 类型不可序列化而报错的问题 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 12:43 Asia/Singapore |
| 涉及脚本 | `src/student/section_parameterization.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 student 截面参数管理模块；定义 `SpanwiseSectionField` 与 `StudentSectionParameters`；新增 `build_baseline_section_parameters(...)`，用于复现当前 `fem_builder.py` 的 frozen baseline 参数语义；新增 `build_element_section_table(...)` 与 `summarize_section_parameters(...)` |
| 所属阶段 | 阶段 2：把 student 截面参数从“占位版”升级到“正式 baseline 版” |
| 当前目的 | 在不改变现有 baseline 的前提下，将质量、EA、EI、GJ、J_rho 等截面参数集中管理，为后续 `corrected_fem_builder.py` 做准备 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 12:43 Asia/Singapore |
| 涉及脚本 | `tests/test_section_parameterization_baseline_equivalence.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增截面参数 baseline 等价性测试脚本；显式复刻当前 `fem_builder.py` 的单元参数取值方式，并与 `section_parameterization.py` 输出的 element table 对比 |
| 所属阶段 | 阶段 2：把 student 截面参数从“占位版”升级到“正式 baseline 版” |
| 当前目的 | 验证新截面参数管理模块能够复现当前 frozen baseline，不引入新的参数语义偏差 |


| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 13:08 Asia/Singapore |
| 涉及脚本 | `src/student/corrected_fem_builder.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 corrected FEM builder；基于 `section_parameterization.py` 输出的 element section table 重新装配 6DOF full-order 梁单元质量矩阵和刚度矩阵；实现单元级 12x12 矩阵构造、全局矩阵装配、根部 6DOF 固定边界条件、M/K 摘要输出；当前无修正时目标是严格复现旧 `fem_builder.py` |
| 所属阶段 | 阶段 2：把 student 截面参数从“占位版”升级到“正式 baseline 版” |
| 当前目的 | 建立新的 corrected FEM 装配入口，为后续 `delta_EA / delta_EI / delta_GJ / delta_J_rho` 等物理参数修正做准备，同时保持 frozen baseline 不变 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 13:08 Asia/Singapore |
| 涉及脚本 | `tests/test_corrected_fem_builder_baseline_equivalence.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 corrected FEM builder baseline 等价性测试；读取同一 `StudentBeamModel`，分别调用旧 `build_fem_matrices_6dof(...)` 和新 `build_corrected_fem_matrices_6dof(...)`，比较 M/K 的 MAE、max_abs 和 relative Frobenius error，并保存 JSON report |
| 所属阶段 | 阶段 2：把 student 截面参数从“占位版”升级到“正式 baseline 版” |
| 当前目的 | 验证新 corrected FEM builder 在无修正时不会引入任何 M/K baseline 偏差 |


| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 13:31 Asia/Singapore |
| 涉及脚本 | `src/student/section_parameterization.py` |
| 增改类型 | 修改 |
| 修改内容 | 新增 `SectionParameterCorrection` 数据结构；新增 `apply_section_parameter_corrections(...)`、`build_corrected_section_parameters_from_model(...)`、`make_uniform_section_correction(...)`；支持对 `EA / EI_flap / EI_edge / GJ / J_rho` 施加 relative/absolute station-level 修正 |
| 所属阶段 | 阶段 2 向阶段 3 过渡：从 baseline 截面参数管理推进到物理截面参数修正 |
| 当前目的 | 将修正项从任意 `delta_K` 矩阵推进为具有物理含义的截面参数修正，并保持 correction=None 时复现 frozen baseline |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 13:31 Asia/Singapore |
| 涉及脚本 | `tests/test_section_correction_sanity.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增截面参数修正 sanity test；验证 zero section correction 不改变 M/K；验证 `delta_EI_flap / delta_EI_edge / delta_GJ` 只改变 K 不改变 M；验证 `delta_J_rho` 改变 M 且不改变 K |
| 所属阶段 | 阶段 2 向阶段 3 过渡：从 baseline 截面参数管理推进到物理截面参数修正 |
| 当前目的 | 确认基于 `section_parameterization.py` 的物理截面修正能够通过 `corrected_fem_builder.py` 正确影响 full-order FEM 矩阵 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 14:32 Asia/Singapore |
| 涉及脚本 | `src/teacher/beamdyn_teacher_adapter.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增独立 BeamDyn teacher adapter；封装 `BeamDynTeacherCaseConfig`、`run_teacher_case_beamdyn(...)`、`load_teacher_6dof_response(...)`、`resample_response_to_time_grid(...)` 和 student-teacher 6DOF 指标计算函数；不依赖 `pipeline_hybrid_transformer.py` |
| 所属阶段 | 阶段 3：无网络 teacher 对齐实验准备 |
| 当前目的 | 建立干净的 teacher 调用与解析入口，为后续手动物理修正 teacher alignment 实验做准备 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 14:32 Asia/Singapore |
| 涉及脚本 | `tests/test_teacher_student_io_alignment_smoke.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 teacher/student IO 对齐 smoke test；调用 BeamDyn teacher、解析 6DOF fullfield `.out`、运行 direct student、将 student 响应重采样到 teacher 时间网格，并输出 baseline student-vs-teacher 指标 |
| 所属阶段 | 阶段 3：无网络 teacher 对齐实验准备 |
| 当前目的 | 在进入手动截面修正 teacher alignment 之前，确认 teacher 输出、student 输出、时间网格和 6DOF 排列均能正确对齐 |


| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 15:04 Asia/Singapore |
| 涉及脚本 | `tests/test_manual_section_correction_teacher_alignment.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增无网络 teacher 对齐手动扫描脚本；运行 BeamDyn teacher，读取 6DOF fullfield，运行 direct student baseline，并分别扫描 `delta_EI_flap_relative` 与 `delta_EI_edge_relative` 的 1D uniform 修正；每个候选修正通过 `corrected_fem_builder.py` 生成 M/K/C，再用 `FullOrderCorrectedCoreTorch` rollout，与 teacher 计算 `full_x/full_y/tip_x/tip_y/last5_y` 等误差，并按目标指标排序输出 best candidate |
| 所属阶段 | 阶段 3：无网络 teacher 对齐实验 |
| 当前目的 | 在引入任何网络训练前，先判断手动物理截面刚度修正是否能降低 student-teacher 误差，从而确定后续哪些物理参数值得学习 |


| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 15:38 Asia/Singapore |
| 涉及脚本 | `tests/test_xy_coupling_diagnostic_teacher_student.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 x/y 耦合诊断脚本；分别运行 x-only 与 y-only 单方向载荷工况下的 BeamDyn teacher 和 direct student，读取并对齐 6DOF full-order 响应，计算主方向响应、交叉方向响应、cross/primary ratio，以及 student cross response 相对 teacher cross response 的比例 |
| 所属阶段 | 阶段 3：无网络 teacher 对齐实验 / x-y 耦合误差诊断 |
| 当前目的 | 验证用户经验判断：teacher 存在明显 x-y 耦合响应，而当前 student 的 x/y 响应基本解耦，从而为后续 flap-edge coupling 修正项设计提供证据 |

在 simple_tip_fx_case 与 simple_tip_fy_case 下，teacher 表现出显著 x-y 交叉响应：
Fx-only 工况下 teacher 的 y/x cross-to-primary RMS ratio 约为 13%～14%；
Fy-only 工况下 teacher 的 x/y cross-to-primary RMS ratio 约为 35%～39%。
而当前 direct student 的交叉响应严格为 0。
因此，当前 student 与 teacher 的主要差异之一不是单纯 EI 幅值偏差，而是 full-order FEM 中缺少 flap-edge bending coupling。



| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 16:10 Asia/Singapore |
| 涉及脚本 | `src/student/coupled_fem_builder.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增基于截面主轴旋转角 `phi` 的 coupled FEM builder；实现 12DOF 单元坐标转换矩阵、刚度矩阵主轴旋转、全局 M/K 装配、degree/radian 两种入口，以及 `K_xy` bending coupling 范数诊断函数；`phi=0` 时应严格回退 `corrected_fem_builder.py` baseline |
| 所属阶段 | 阶段 3：x-y coupling 物理修正建模 |
| 当前目的 | 将 x/y 耦合缺失从诊断推进到可解释物理修正项，用 `phi(s)` 表示截面主弯曲轴相对 student x/y 坐标的旋转 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 16:10 Asia/Singapore |
| 涉及脚本 | `tests/test_principal_axis_rotation_coupling_sanity.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 principal-axis rotation coupling sanity test；验证 `phi=0` 时 M/K 与 baseline 一致，`phi≠0` 时 K 出现 x/y bending coupling，并在 x-only / y-only 单方向载荷下产生非零交叉响应 |
| 所属阶段 | 阶段 3：x-y coupling 物理修正建模 |
| 当前目的 | 在不接 teacher alignment 和网络训练前，确认 `phi` 耦合项确实进入 full-order FEM 与 Newmark 响应链路 |

在 simple_tip_fx_case 与 simple_tip_fy_case 下，direct student 的交叉方向响应严格为 0，而 BeamDyn teacher 存在显著交叉响应。引入截面主轴旋转角 phi 后，phi=0 能严格回退 baseline，phi=5° 能在 full-order FEM 刚度矩阵中产生非零 K_xy coupling，并在 x-only / y-only 工况下产生非零交叉响应。其中 y-only 工况下 student 的 cross/primary ratio 已达到 teacher 的约 89%～91%，说明主轴旋转型 flap-edge coupling 是解释 teacher/student 差异的有效物理修正方向。

## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 16:36 Asia/Singapore |
| 涉及脚本 | `src/student/spanwise_phi_parameterization.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增展向主轴旋转角 `phi(s)` 参数化模块；定义 `SpanwisePhiProfile`，支持 uniform、linear、piecewise constant、control-point interpolation 四类 station-level phi profile，并提供 element-level 平均、profile 摘要和浮点列表解析工具 |
| 所属阶段 | 阶段 3：x-y coupling 物理修正建模 |
| 当前目的 | 将单一常数 `phi` 扩展为 spanwise-varying `phi(s)`，为后续每个截面/分段不同主轴旋转角的 teacher alignment 实验做准备 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 16:36 Asia/Singapore |
| 涉及脚本 | `tests/test_spanwise_phi_parameterization.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 spanwise phi 参数化测试；构造 zero/uniform/linear/piecewise/control-point profiles，并调用 `build_coupled_fem_matrices_6dof_degrees(...)` 验证 `phi(s)=0` 回退 baseline，非零 profile 产生非零 `K_xy` bending coupling |
| 所属阶段 | 阶段 3：x-y coupling 物理修正建模 |
| 当前目的 | 在进入 teacher alignment 扫描前，确认展向变化 `phi(s)` 能正确接入 coupled FEM builder |

## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 17:03 Asia/Singapore |
| 涉及脚本 | `tests/test_spanwise_phi_teacher_alignment.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增展向 `phi(s)` teacher 对齐扫描脚本；运行 x-only 与 y-only BeamDyn teacher 和 direct student，构造多种 uniform/linear/piecewise/control-point `phi(s)` profile，通过 `coupled_fem_builder.py` 和 `FullOrderCorrectedCoreTorch` rollout，比较 corrected student 与 teacher 的 cross/primary ratio 差异，并按综合目标排序 |
| 所属阶段 | 阶段 3：x-y coupling 物理修正建模 / teacher 对齐扫描 |
| 当前目的 | 从机制验证推进到 teacher 对齐，判断哪类展向主轴旋转角分布最能复现 BeamDyn teacher 的 x-y 交叉响应 |


在 simple_tip_fx_case 与 simple_tip_fy_case 的 teacher 对齐扫描中，展向变化的主轴旋转角 phi(s) 能够有效复现 teacher 的 x-y 交叉响应比例。当前最佳 profile 为 root-heavy 线性分布，即叶根处 |phi| 最大、向叶尖逐渐衰减至 0°。这说明 x-y 耦合主要受根部/中根部高刚度截面影响，而不是单纯由叶尖局部扭转决定。与此同时，+phi 与 -phi 在当前 RMS ratio 指标下结果完全相同，说明当前指标只能确定耦合强度，不能确定耦合方向。下一步需要引入 signed correlation 或 signed time-series error，以判断 phi 的正确符号和相位一致性。

## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 17:31 Asia/Singapore |
| 涉及脚本 | `tests/test_spanwise_phi_teacher_alignment_signed.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增展向 `phi(s)` signed teacher 对齐扫描脚本；在原有 cross/primary RMS ratio 对齐基础上，增加 cross-response 时间序列的 MAE、MSE、RMSE、normalized RMSE、signed correlation 和 student/teacher RMS ratio，用于区分 `+phi` 与 `-phi` 的物理方向 |
| 所属阶段 | 阶段 3：x-y coupling 物理修正建模 / signed teacher 对齐扫描 |
| 当前目的 | 解决 RMS ratio 指标无法区分正负号的问题，判断 `phi(s)` 的正确符号、相位一致性和时间序列误差 |

当前应停止把 EI_edge 作为第一批学习参数，因为它之前带来的改善很可能是在间接补偿 x/y 耦合缺失。下一步应把训练目标收敛到 κ_phi，即每个梁单元一个主轴旋转角 phi_e，共 48 个 element-level 参数。phi_e 是响应无关的截面耦合刚度参数，先通过 simple x/y 单方向载荷训练，使 student 产生与 teacher 同符号、同相位的交叉响应。训练稳定后，将 phi_e 固定为已知结构参数，再进入基于 student 响应与外载荷时历的 Transformer encoder 阶段，学习更高阶的响应相关物理参数。
## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 18:05 Asia/Singapore |
| 涉及脚本 | `src/student/trainable_phi_correction.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 48 个 element-level `phi_e` 的训练参数工具模块；提供 bounded raw-to-phi 映射、初始化 profile、平滑/曲率/幅值正则、raw clip、保存/读取 `phi` profile 等工具 |
| 所属阶段 | 阶段 3：x-y coupling 物理修正建模 / `κ_phi` 训练准备 |
| 当前目的 | 将 `phi(s)` 从手动 profile 扫描推进到可训练的 48 维 element-level 结构参数 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-25 18:05 Asia/Singapore |
| 涉及脚本 | `scripts/train_phi_correction_spsa.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增基于 SPSA 的 `phi_e` 无梯度训练脚本；只使用 `simple_tip_fx_case.dat` 与 `simple_tip_fy_case.dat` 两个单方向载荷文件，训练 48 个 element-level `phi_e`，loss 使用 cross-response normalized RMSE、signed correlation penalty 和 `phi` 平滑正则 |
| 所属阶段 | 阶段 3：x-y coupling 物理修正建模 / 第一版 `κ_phi` 训练 |
| 当前目的 | 在不训练 `EI_edge`、不引入 Transformer、不直接修正响应的前提下，先得到稳定的已知 `κ_phi` 结构耦合参数 |


| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `tests/test_uniform_phi_plot_compare.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增一个直接绘图版本的 teacher / student(phi=0) / student(phi=-7.5) 对比脚本，默认绘制叶尖 tip 在 x、y 两个方向的位移时历，并分别输出两张 PNG 图 |
| 所属阶段 | 阶段3：`κ` 组中的截面耦合修正（当前聚焦 `phi` 参数化验证） |
| 当前目的 | 直接观察在单一工况下，统一 `phi=0°` 与 `phi=-7.5°` 时，student 相对于 teacher 的 x/y 位移时历差异 |

phi=-7.5° 使 y 方向误差大幅降低、x 方向误差小幅增加，说明 x/y 主轴耦合修正确实命中了当前 student 的主要误差来源。它不是单独修 y 的参数，而是改变 full-order FEM 的弯曲耦合结构，因此会对 x/y 两个方向同时产生影响。后续应保留 κ_phi 作为主线修正，但在训练和筛选时加入 x 方向误差保护，避免为了降低 y 误差而过度牺牲 x 方向响应。

## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `src/student/torch_phi_parameterization.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 PyTorch 版 48 个 element-level `phi_e` 参数模块，支持有界角度映射、初始化、平滑/曲率/幅值正则以及 profile 保存读取 |
| 所属阶段 | 阶段 3：`κ_phi` 可微训练 |
| 当前目的 | 将 `phi_e` 从 SPSA 无梯度优化参数升级为 PyTorch 可训练参数 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `src/student/torch_coupled_fem_builder.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 Torch 可微 `K(phi)` builder；通过 NumPy coupled FEM builder 预计算 per-element trig basis，并在 Torch 中用 `cos(2phi)` / `sin(2phi)` 构造可反传刚度矩阵 |
| 所属阶段 | 阶段 3：`κ_phi` 可微训练 |
| 当前目的 | 保证 `phi -> K(phi) -> rollout -> loss` 链路可反传，避免继续使用 SPSA |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `tests/test_torch_coupled_fem_builder_equivalence.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 Torch FEM builder 等价性测试，验证 `phi=0` 回退 baseline、`phi=-7.5` 接近 NumPy coupled builder，并确认 autograd 梯度存在 |
| 所属阶段 | 阶段 3：`κ_phi` 可微训练保护测试 |
| 当前目的 | 在正式训练前确认 Torch 版 `K(phi)` 没有破坏原有 full-order FEM 语义 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `scripts/prepare_phi_training_cases.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增训练数据缓存脚本，提前运行 BeamDyn teacher 与 direct student，保存 teacher 对齐响应、baseline 响应、F_time、初始状态和 baseline x/y MSE |
| 所属阶段 | 阶段 3：`κ_phi` 可微训练数据准备 |
| 当前目的 | 避免训练过程中反复运行 BeamDyn，提高 PyTorch 训练效率 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `scripts/train_phi_correction_torch.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 PyTorch 可微训练脚本，使用 Adam 优化 48 个 `phi_e`，loss 包含全局 y 误差降低、全局 x 误差 guard、phi 平滑/曲率/幅值正则 |
| 所属阶段 | 阶段 3：`κ_phi` 可微训练 |
| 当前目的 | 训练出既能降低全局 y 误差、又不显著恶化全局 x 误差的 `phi_e` 展向分布 |

## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `src/student/torch_phi_parameterization.py` |
| 增改类型 | 修改 |
| 修改内容 | 将 `TrainableElementPhiConfig` 和 `TrainableElementPhi.__init__` 的默认初始化方式从 `uniform_neg7p5` 改为 `linear_root_to_tip`，默认 `root_phi=-15°`、`tip_phi=0°` |
| 所属阶段 | 阶段 3：`κ_phi` 可微训练 |
| 当前目的 | 使默认 `phi_e` 初始化更符合当前物理判断：叶根主轴旋转角接近 -15°，叶尖接近 0° |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `scripts/train_phi_correction_torch.py` |
| 增改类型 | 修改 |
| 修改内容 | 将训练脚本默认初始化从 uniform `-7.5°` 改为 root-heavy `-15° -> 0°`；建议同步将 `w_smooth` 从 `0.1` 降为 `0.01`，将 `w_curvature` 从 `0.01` 提高到 `0.05` |
| 所属阶段 | 阶段 3：`κ_phi` 可微训练 |
| 当前目的 | 加速训练早期形成合理的 root-to-tip 展向 `phi(s)` 分布，避免从常数 `-7.5°` 缓慢爬升 |

## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `scripts/train_phi_correction_torch.py` |
| 增改类型 | 修改 |
| 修改内容 | 新增 `--train-case-names` 与 `--valid-case-names`，将缓存工况显式划分为训练集和验证集；默认使用 `train_complex_case` 训练、`test_complex_case` 验证 |
| 所属阶段 | 阶段 3：`κ_phi` 可微训练稳定化 |
| 当前目的 | 避免继续只追训练 loss，通过验证集选择更稳健的 `phi_e` checkpoint |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `scripts/train_phi_correction_torch.py` |
| 增改类型 | 修改 |
| 修改内容 | 新增验证集 `valid_score`，使用 `valid_y_ratio + x_guard` 作为 checkpoint 选择标准；只有验证集改善超过 `early_stop_min_delta` 才更新 `best_phi_profile.npz` |
| 所属阶段 | 阶段 3：`κ_phi` 可微训练稳定化 |
| 当前目的 | 防止训练集 loss 长期缓慢下降导致无意义微调或过拟合 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `scripts/train_phi_correction_torch.py` |
| 增改类型 | 修改 |
| 修改内容 | 新增 early stopping 参数：`--early-stop-patience`、`--early-stop-min-delta`；验证集长期无显著改善时自动停止 |
| 所属阶段 | 阶段 3：`κ_phi` 可微训练稳定化 |
| 当前目的 | 建立停止准则，不再无限追逐训练 loss 的微小下降 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `scripts/train_phi_correction_torch.py` |
| 增改类型 | 修改 |
| 修改内容 | 新增学习率平台衰减参数：`--lr-plateau-patience`、`--lr-decay-factor`、`--min-lr`；验证集平台期自动降低学习率 |
| 所属阶段 | 阶段 3：`κ_phi` 可微训练稳定化 |
| 当前目的 | 后期由大步搜索切换为小步微调，减少震荡与过度追训练误差 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `scripts/train_phi_correction_torch.py` |
| 增改类型 | 修改 |
| 修改内容 | 新增 `--phi-quant-step-deg`，默认 `0.01°`；通过 straight-through estimator 实现前向量化、反向保留梯度；保存时同时输出 continuous 与 quantized phi |
| 所属阶段 | 阶段 3：`κ_phi` 可微训练稳定化 |
| 当前目的 | 将 `phi_e` 有效精度限制到 0.01°，减少后期极小尺度连续微调 |

## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `tests/test_initial_twist_phi_teacher_alignment.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 initial_twist_deg 构型先验验证脚本，从 `blade_master.csv` 读取 49 个 `initial_twist_deg`，构造 48 个 element-level `phi`，并比较 `phi=0 student`、`phi=initial_twist student` 与 BeamDyn teacher 的响应误差 |
| 所属阶段 | 阶段 3：`κ_phi` 截面主轴旋转修正验证 |
| 当前目的 | 判断叶片构型中的结构扭转角是否已经足以作为固定 `Phi(s)`，从而避免继续训练 `Phi` |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `tests/test_initial_twist_phi_teacher_alignment.py` |
| 增改类型 | 新增 |
| 修改内容 | 支持 `--phi-sign-mode both/direct/negative`，用于同时验证 `+initial_twist_deg` 与 `-initial_twist_deg` 两种符号约定 |
| 所属阶段 | 阶段 3：`κ_phi` 截面主轴旋转修正验证 |
| 当前目的 | 解决 BeamDyn/student 坐标系符号可能不一致的问题，先通过结果确定正确符号 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `tests/test_initial_twist_phi_teacher_alignment.py` |
| 增改类型 | 新增 |
| 修改内容 | 输出 `full_x/full_y/tip/lastk` 等 teacher 对齐误差，并计算相对 `phi=0 student` 的改善比例 |
| 所属阶段 | 阶段 3：`κ_phi` 截面主轴旋转修正验证 |
| 当前目的 | 为是否冻结 `Phi(s)` 提供定量依据 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `tests/test_initial_twist_phi_teacher_alignment.py` |
| 增改类型 | 结论性说明 |
| 修改内容 | 当前脚本不进行 `Phi` 训练，只做 teacher / student 对齐验证；如果 `initial_twist_deg(s)` 已经显著改善 teacher 对齐结果，则后续可将 `Phi(s)` 固定为叶片构型参数 |
| 所属阶段 | 阶段 3：`κ_phi` 截面主轴旋转 / x-y 弯曲耦合修正验证 |
| 当前目的 | 判断是否结束 `Phi` 训练阶段，并进入后续 `ζ` 阻尼修正、`f` 载荷映射修正或响应相关物理参数修正阶段 |

## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `src/student/base_phi.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 base student 固定构型 `Phi_base(s)` 构造工具，从 `blade_master.csv` 的 `initial_twist_deg` 读取 49 个 station-level 初始扭转角，并以负号相邻平均构造 48 个 element-level `Phi_base(s)` |
| 所属阶段 | 阶段 3：固定 `κ_phi` 截面主轴旋转构型先验 |
| 当前目的 | 将已经验证有效的 `Phi(s)=-initial_twist_deg(s)` 从可训练参数转为 base student 的固定构型参数 |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `src/student/base_full_order_builder.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 base student full-order M/K 统一构造函数 `build_base_student_full_order_mk(...)`，默认使用 `Phi_base(s)=-initial_twist_deg(s)` 调用 `build_coupled_fem_matrices_6dof_degrees(...)` 装配耦合刚度矩阵 |
| 所属阶段 | 阶段 3：固定 `κ_phi` 截面主轴旋转构型先验 |
| 当前目的 | 统一后续 direct student、teacher-student 对齐测试和训练脚本中的 base M/K 构造逻辑，避免不同脚本使用不同 baseline |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `scripts/run_student_cases.py` |
| 增改类型 | 修改 |
| 修改内容 | 将 direct student 的 M/K 构造路径从原始 uncoupled FEM baseline 改为带固定 `Phi_base(s)=-initial_twist_deg(s)` 的 coupled FEM baseline |
| 所属阶段 | 阶段 3：固定 `κ_phi` 截面主轴旋转构型先验 |
| 当前目的 | 使 base student 默认具备由叶片初始扭转角引入的 x-y 弯曲耦合，后续训练不再学习 `Phi` |

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及内容 | 后续训练参数定义 |
| 增改类型 | 设计调整 |
| 修改内容 | 将 `Phi` 从待学习的 `κ` 修正参数中移除，改为 base student 固定构型参数；后续 `κ` 若继续使用，应解释为 `κ_residual`，即在固定 `Phi_base` 之后的响应相关刚度/内力残差 |
| 所属阶段 | 阶段 4 准备：响应与外载荷相关物理参数训练 |
| 当前目的 | 明确后续训练只学习与 `u, u_dot, u_ddot, F_ext` 相关的物理参数，暂不学习质量 `ψ` 与阻尼 `ζ` |


| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `scripts/prepare_physical_training_cases_base_phi.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增物理训练缓存准备脚本，用于从载荷文件列表自动运行 BeamDyn teacher 与当前新 base student，完成时间对齐、初始偏置处理、baseline 指标计算，并保存包含 `F_raw`、`u_teacher`、`u_base`、`v_base`、`a_base`、`base_phi_element_deg` 等字段的 `.npz` 训练缓存 |
| 所属阶段 | 阶段 4 准备：响应与外载荷相关物理参数训练数据构造 |

## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `scripts/train_physical_correction_torch.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增统一物理参数训练脚本，当前第一版仅启用 `LoadMappingCorrection`，用于训练 `f` 载荷映射参数组；脚本支持自动调用训练缓存准备流程、构造固定 `Phi_base(s)=-initial_twist_deg(s)` 的 base full-order core、通过 Newmark/PDE rollout 得到响应，并以降低 y 方向误差且限制 x 方向恶化为训练目标 |
| 所属阶段 | 阶段 4：训练外载荷映射参数 `f` |

## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `tests/test_y_frequency_teacher_student_diagnostic.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 y 方向频率诊断脚本，用于在 y-only 初始载荷/时历载荷工况下分别运行 BeamDyn teacher 与当前 base student，提取 tip y 与 last-k mean y 响应，通过 FFT 估计主频并输出 teacher/student 频率比值，同时保存时历图、频谱图和 JSON 报告 |
| 所属阶段 | 阶段 4：静态物理参数训练后的频率诊断；为后续 `κ_y_residual(s)` 静态刚度残差设计提供依据 |

## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `tests/test_kappa_y_stiffness_scale_frequency_scan.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 y 主导刚度缩放频率扫描脚本。脚本在固定 `Phi_base(s)=-initial_twist_deg(s)` 的当前 base student 上构造 M/K，并通过 `K_scaled = S K S` 对 y bending 相关自由度进行对称刚度缩放，扫描不同 y stiffness scale 下的 student y 响应主频，与 BeamDyn teacher 的 tip y 和 last-k mean y 主频进行比较。 |
| 所属阶段 | 阶段 4：静态物理参数训练路线中的 `κ_y_residual(s)` 前置诊断，用于判断降低 y 主导刚度是否能对齐 teacher y 主频。 |

## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `tests/test_kappa_y_scale_response_compare.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 y-only 载荷下的三模型时历对比脚本。脚本分别运行 BeamDyn teacher、当前 base student，以及通过 `K_scaled = S K S` 对 y-bending<br/> 相关自由度施加 `kappa_y_scale` 的 corrected student；输出 tip y、last-k mean y 和 tip x guard 三类三曲线时历图，并计算 base/corrected 相对 teacher 的 full/tip/last-k x/y 误差指标。 |
| 所属阶段 | 阶段 4：静态物理参数训练路线中的 `κ_y_residual(s)` 前置验证；用于确认 y 主导刚度缩放不仅能修正频率，也能改善 y-only 载荷下的时历响应。 |

## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `scripts/train_kappa_y_global_torch.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增全局 `κ_y_scale` 静态刚度缩放训练脚本。脚本在固定 `Phi_base(s)=-initial_twist_deg(s)` 的 base student 上，只训练一个全局 y-bending stiffness scale 参数，通过 `K_scaled = S K_base S` 对 y-bending 相关自由度进行对称刚度缩放，并使用 y-only 载荷下的 teacher/student 时历误差作为训练目标。当前脚本不训练 `f_static`。 |
| 所属阶段 | 阶段 4：静态物理参数训练中的 `κ_y_residual(s)` 第一阶段，即全局 y 向刚度缩放参数训练。 |

## 开发者日志

| 项目 | 内容 |
|---|---|
| 修改时间 | 2026-04-26 |
| 涉及脚本 | `scripts/train_kappa_y_control_points_torch.py` |
| 增改类型 | 新增 |
| 修改内容 | 新增 6 个展向控制点形式的 `κ_y_scale(s)` 静态刚度缩放训练脚本。脚本在固定 `Phi_base(s)=-initial_twist_deg(s)` 的 base student 上，通过控制点线性插值得到 48 个自由节点上的 y-bending stiffness scale，并用 `K_scaled = S K_base S` 对 y-bending 相关自由度进行对称刚度缩放。训练目标包含 y 向 full/tip/last-k 时历误差、x 向 guard、控制点先验、平滑与曲率正则。当前脚本不包含 `f_static`，不包含 Transformer 动态参数。 |
| 所属阶段 | 阶段 4：静态物理参数训练中的 `κ_y_residual(s)` 第二阶段，即低维展向控制点形式的 y 向刚度缩放训练。 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-26 | `src/student/base_full_order_builder.py` | 修改 | 在固定 `Phi_base(s)=-initial_twist_deg(s)` 的 full-order base builder 中新增全局 `kappa_y_global_scale` 接口，默认启用 `0.952`；新增 `selected_kappa_y_dof_indices(...)`、`build_kappa_y_global_scale_vector(...)`、`apply_global_kappa_y_scale_to_k(...)`，通过 `K_base = S_y K_phi S_y` 将 y-bending 等效刚度缩放并入静态 base student。 | 阶段 5：Transformer 动态物理参数学习准备 |
| 2026-04-26 | `tests/test_base_full_order_builder_kappa_y_global.py` | 新增 / 测试 | 新增 base builder 的全局 `kappa_y_scale` 注入测试脚本；验证 `M` 不变、`K` 发生变化、`K_scaled` 与手动 `S K S` 一致、`K` 保持对称且有限，并保存 JSON report。 | 阶段 5：Transformer 动态物理参数学习准备 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-26 | `src/student/physical_parameter_registry.py` | 新增 | 新增 Transformer 可学习物理参数注册表，定义 `PhysicalParameterSpec` 与 `PhysicalParameterRegistry`；第一版默认支持并启用 `alpha_y` 与 `alpha_xy`，同时预注册 `alpha_bt`、载荷映射、陀螺耦合、速度二次惯性力和应变率阻尼等后续候选参数。 | 阶段 5：Transformer 动态物理参数学习准备 |
| 2026-04-26 | `src/student/physical_templates.py` | 新增 | 新增动态物理刚度模板构造模块；基于 static corrected student 的 `K0=S_y(0.952)K_phiS_y(0.952)` 构造 `K_y_template` 与 `K_xy_template`，用于后续 `K_eff=K0+alpha_y*K_y_template+alpha_xy*K_xy_template`。 | 阶段 5：Transformer 动态物理参数学习准备 |
| 2026-04-26 | `tests/test_physical_parameter_registry_and_templates.py` | 新增 / 测试 | 新增物理参数注册表与刚度模板 sanity test；验证参数注册、theta 拆分、模板矩阵形状、有限性、对称性、非零性，以及 `theta=0` 回退 `K0`、`theta!=0` 改变 `K_eff`。 | 阶段 5：Transformer 动态物理参数学习准备 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-26 | `src/student/transformer/physical_parameter_registry.py` | 新增 / 路径调整 | 将原计划中的 `src/student/physical_parameter_registry.py` 调整到 `src/student/transformer/physical_parameter_registry.py`，用于集中管理 Transformer 可学习物理参数注册表。 | 阶段 5：Transformer 动态物理参数学习准备 |
| 2026-04-26 | `src/student/transformer/physical_templates.py` | 新增 / 路径调整 | 将原计划中的 `src/student/physical_templates.py` 调整到 `src/student/transformer/physical_templates.py`，用于集中构造 Transformer 动态物理参数所需的刚度模板。 | 阶段 5：Transformer 动态物理参数学习准备 |
| 2026-04-26 | `tests/test_physical_parameter_registry_and_templates.py` | 修改 / 测试 | 测试脚本保留在 `tests/` 目录下，仅将 import 路径改为 `src.student.transformer.*`，以适配新的 Transformer 子目录结构。 | 阶段 5：Transformer 动态物理参数学习准备 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-26 | `src/student/transformer/dynamic_physical_core_torch.py` | 新增 | 新增 Transformer 动态物理 core；支持将 registry 中启用的低维物理参数 `theta_t` 映射为 `K_eff=K0+alpha_y*K_y_template+alpha_xy*K_xy_template`，并提供可反传的 full-order Newmark 单步推进。当前第一版只支持刚度类参数进入 `K_eff`，暂不启用载荷映射、陀螺项、速度二次惯性力或阻尼修正。 | 阶段 5：Transformer 动态物理参数学习准备 |
| 2026-04-26 | `tests/test_dynamic_physical_core_torch.py` | 新增 / 测试 | 新增 dynamic physical core sanity test；验证 `theta=0` 时 `K_eff` 回退 `K0`，`theta!=0` 时 `K_eff` 和单步响应发生可检测变化，并检查 loss 能通过 Newmark 单步反向传播到物理参数 `theta`。 | 阶段 5：Transformer 动态物理参数学习准备 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-26 | `src/student/transformer/blade_geometry_features.py` | 新增 | 新增叶片节点级几何/结构特征构造模块，从 `blade_master.csv`、`Phi_base(s)=-initial_twist_deg(s)` 和 baseline section parameters 中构造 48 个自由节点的 geometry features，用于 Transformer geometry branch。 | 阶段 5：Transformer 第一阶段网络输入分支构建 |
| 2026-04-26 | `src/student/transformer/spatial_attention_encoders.py` | 新增 | 新增 response / load / geometry 三个空间注意力分支和可开关 gated fusion 模块；response branch 处理每节点 `u,v,a`，load branch 处理每节点 `F`，geometry branch 处理叶片静态构型特征，fusion 支持任意分支组合开启。 | 阶段 5：Transformer 第一阶段网络输入分支构建 |
| 2026-04-26 | `tests/test_transformer_attention_branches.py` | 新增 / 测试 | 新增 Transformer 三分支 attention 测试脚本；验证几何特征构造、response/load/geometry 分支输入输出形状、attention 权重归一化、所有分支组合的 gated fusion 输出是否正确。 | 阶段 5：Transformer 第一阶段网络输入分支构建 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-26 | `src/student/transformer/physical_parameter_heads.py` | 新增 | 新增有界物理参数输出头 `BoundedPhysicalParameterHead`，根据 `PhysicalParameterRegistry` 将 encoder hidden state 映射为有界 `theta_t`；第一阶段支持输出 `alpha_y(t)`，后续可扩展到 `alpha_y,alpha_xy`。 | 阶段 5：Transformer 第一阶段网络主体构建 |
| 2026-04-26 | `src/student/transformer/spatiotemporal_physics_encoder.py` | 新增 | 新增时空物理参数编码器，串联 response/load/geometry 三个可开关分支、gated fusion、causal temporal Transformer 和有界物理参数 head；支持从 full-order `u,v,a,F` 自动 reshape 为节点级特征，并输出 `theta_t`。 | 阶段 5：Transformer 第一阶段网络主体构建 |
| 2026-04-26 | `tests/test_spatiotemporal_physics_encoder_and_heads.py` | 新增 / 测试 | 新增时空编码器与物理参数 head 测试脚本；验证 `alpha_y` 或 `alpha_y,alpha_xy` 输出 shape、有界性、分支开关组合、fusion gate 归一化以及梯度反传。 | 阶段 5：Transformer 第一阶段网络主体构建 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-26 | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增 Transformer 动态物理参数第一阶段训练主脚本；当前默认使用 static-conditioning，只训练第一优先级全局参数 `alpha_y(t)`，并通过 `K_eff=K0+alpha_y*K_y_template` 进入 full-order Newmark 物理 core；teacher response 只作为 label 计算 loss，不进入网络输入。脚本支持 response/load/geometry 三个分支开关、训练/验证 cache 读取、可选自动准备 BeamDyn teacher cache、短序列 smoke training、checkpoint 保存、history CSV 与 summary JSON 输出。 | 阶段 5：Transformer 第一阶段 teacher-supervised static-conditioning 训练 |

第一阶段 alpha_y(t) static-conditioning smoke training 已跑通。
response-only、response+load、response+load+geometry 三种配置均能反传并降低 y 方向误差。
其中 response+load 在当前 120-step smoke setting 下表现最好：
valid_y_ratio 从 static corrected baseline 的 1.0 降至约 0.642，
且 valid_x_ratio 保持在约 0.980，没有出现 x 方向恶化。

Transformer 第一阶段 alpha_y(t) static-conditioning 训练已在 240-step response+load 设置下跑通。
模型将 valid_y_ratio 从 1.0 降至 0.598656，且 valid_x_ratio 保持在 0.995347，没有造成 x 方向恶化。
当前 theta_max=0.036788，未达到 alpha_y 输出上限 0.05，说明参数仍处于可解释动态刚度残差范围内。

alpha_y(t) + response/load + static conditioning 在 240-step 训练中有效。
最佳验证点为 epoch 63，valid_y_ratio=0.518847，valid_x_ratio=0.996679。
相比 static corrected baseline，验证集 y 方向误差约降低 48.1%，且 x 方向没有恶化。
训练后期出现轻微过拟合，后续应使用 best checkpoint，而不是 last checkpoint。

## 开发者日志

| 修改时间 | 涉及脚本/文件                                        | 增改类型 | 修改内容 | 所属阶段 |
|---|------------------------------------------------|---|---|---|
| 2026-04-27 | `scripts/evaluate_transformer_vs_baselines.py` | 新增 | 新增独立 Transformer 评估脚本；该脚本在不修改基础脚本的前提下，显式计算 BeamDyn teacher、no-scale Phi-base student、static kappa-y student 以及 trained Transformer 四条响应时历，并保存指标、时历 CSV、响应 npz 和对比图。 | 阶段 5：Transformer 第一阶段模型评估与 baseline 对比 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-27 | `scripts/evaluate_transformer_vs_baselines.py` | 修改 | 修改 `plot_timeseries(...)` 的绘图样式，加入不同 linestyle、marker 和 zorder，避免 `tip_x` 等分量中多条曲线重合时看起来像只有两条线。 | 阶段 5：Transformer 第一阶段评估与时历对比 |
| 2026-04-27 | `scripts/evaluate_transformer_vs_baselines.py` | 新增 | 新增 `plot_timeseries_error_to_teacher(...)`，用于绘制 no-scale baseline、static kappa-scale baseline 和 Transformer 相对于 teacher 的误差时历，更清楚判断各模型是否真正改善响应。 | 阶段 5：Transformer 第一阶段评估与时历对比 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-27 | `scripts/train_transformer_physical_params_torch.py` | 配置调整 | 明确后续 `alpha_y + alpha_xy` 联合训练中，x 方向作为叶片挥舞主响应方向，不允许相对 static kappa-y baseline 恶化；建议使用 `--x-guard-tol 0.0`、更大的 `--w-x-guard` 和更大的 `--w-x` 进行 hard constraint 风格训练。 | 阶段 5：Transformer 动态物理参数 alpha_y + alpha_xy 联合训练 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-27 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 在 `TransformerPhysicalTrainConfig` 中新增 `use_x_constraint_for_best` 和 `x_best_constraint_max`，用于控制 best checkpoint 是否必须满足 x 方向不恶化约束。 | 阶段 5：Transformer 动态物理参数 alpha_y + alpha_xy 联合训练 |
| 2026-04-27 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 在 argparse 中新增 `--use-x-constraint-for-best`、`--no-x-constraint-for-best` 和 `--x-best-constraint-max` 参数，使训练命令可以启用 `valid_x <= 1.0` 的 best checkpoint 硬约束。 | 阶段 5：Transformer 动态物理参数 alpha_y + alpha_xy 联合训练 |
| 2026-04-27 | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增 `can_update_best_checkpoint(...)` 辅助函数，根据验证集 `valid_x/x_ratio` 判断当前 epoch 是否允许参与 best checkpoint 更新。 | 阶段 5：Transformer 动态物理参数 alpha_y + alpha_xy 联合训练 |
| 2026-04-27 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 修改 best checkpoint 更新逻辑，使其从单纯 `score < best_score` 改为 `eligible_for_best and score < best_score`，从而实现 `min score subject to valid_x <= x_best_constraint_max`。 | 阶段 5：Transformer 动态物理参数 alpha_y + alpha_xy 联合训练 |
| 2026-04-27 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 建议在 epoch 日志、history CSV 和 summary JSON 中记录 `best_gate`、`best_gate_reason`、`use_x_constraint_for_best` 和 `x_best_constraint_max`，方便判断哪些 epoch 因 x 恶化被排除。 | 阶段 5：Transformer 动态物理参数 alpha_y + alpha_xy 联合训练 |

在 x 方向不允许恶化的约束下，alpha_y-only 模型明显优于 alpha_y + alpha_xy。当前 alpha_y-only 最优结果为 valid_y=0.763813、valid_x=0.967845，说明其能够改善 y 方向响应，同时保持并改善 x 挥舞方向响应。由于 theta_max 已接近 0.05，下一阶段应优先将 alpha_y 从全局标量扩展为展向控制点形式，而不是继续使用当前 alpha_xy 模板。

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-27 | `src/student/transformer/alpha_y_control_points.py` | 新增 | 新增 alpha_y 展向控制点工具模块，支持解析 `alpha_y_cpN` 参数名、构造线性 hat 控制点权重、生成 y-bending DOF 权重，以及通过有限差分构造 `K_y_cp_tpl` 控制点刚度残差模板。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |
| 2026-04-27 | `src/student/transformer/physical_parameter_registry.py` | 修改 | 在物理参数注册表中新增对 `alpha_y_cpN` 的支持，例如 `alpha_y_cp3` 自动注册为 3 维物理参数，供 Transformer head 输出。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |
| 2026-04-27 | `src/student/transformer/physical_templates.py` | 修改 | 在物理模板构造流程中新增 `K_y_cp_tpl`，当启用 `alpha_y_cpN` 时自动构造 shape 为 `(N, n_dofs, n_dofs)` 的控制点刚度残差模板。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |
| 2026-04-27 | `src/student/transformer/dynamic_physical_core_torch.py` | 修改 | 修改动态物理 core 的 K 装配逻辑，支持 `alpha_y_cpN` 通过 `einsum` 将多个控制点参数装配进 `K_eff = K0 + sum_i alpha_y_cp_i K_y_cp_tpl_i`。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-27 | `src/student/transformer/physical_templates.py` | 修改 | 将物理模板构造脚本改为自包含兼容版：新增 enabled params 解析逻辑，优先读取 `cfg.enabled_params`，若为空则从命令行 `--enabled-params` 兜底解析。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |
| 2026-04-27 | `src/student/transformer/physical_templates.py` | 修改 | 当解析到 `alpha_y_cpN` 时，自动生成 `K_y_cp_tpl`、`alpha_y_cp_dof_weights` 和 `alpha_y_cp_n_control_points`，并通过 `PhysicalTemplateBundle.stiffness_template_dict()` 暴露给 `DynamicPhysicalCoreTorch`。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |
| 2026-04-27 | `src/student/transformer/physical_templates.py` | 修改 | 在 summary 和 metadata 中补充控制点模板的 shape、范数、对称性和物理含义，便于后续调试与代码说明。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-27 | `src/student/transformer/dynamic_physical_core_torch.py` | 修改 | 将 `DynamicPhysicalCoreTorch` 从仅支持二维 stiffness template 扩展为同时支持二维模板 `(D,D)` 和三维控制点模板 `(N,D,D)`。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |
| 2026-04-27 | `src/student/transformer/dynamic_physical_core_torch.py` | 修改 | 新增 `alpha_y_cpN -> K_y_cp_tpl` 的参数到模板映射逻辑，使 `alpha_y_cp3` 可通过 `K_eff = K0 + sum_i alpha_i K_y_cp_tpl[i]` 进入刚度矩阵。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |
| 2026-04-27 | `src/student/transformer/dynamic_physical_core_torch.py` | 修改 | 修改 stiffness template 注册逻辑，允许注册 `K_y_cp_tpl` 这类三维模板，同时保留 `K_y_template` 和 `K_xy_template` 的二维模板逻辑。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |
| 2026-04-27 | `src/student/transformer/dynamic_physical_core_torch.py` | 修改 | 修改 `assemble_stiffness(...)`，当模板为三维时使用 `torch.einsum("p,pij->ij", theta_part, template)` 装配控制点刚度贡献。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-27 | `src/student/transformer/dynamic_physical_core_torch.py` | 修改 | 将原 `newmark_step(...)` 拆分为 `_newmark_step_single(...)` 和支持 batch 的 `newmark_step(...)`，使 physical core 同时支持单样本 `(D,)` 与批量 `(B,D)` 状态输入。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |
| 2026-04-27 | `src/student/transformer/dynamic_physical_core_torch.py` | 修改 | 新增 batch theta 支持，允许 `theta_t=None`、`theta_t=(P,)` 或 `theta_t=(B,P)`，其中 `alpha_y_cp3` 对应 `P=3`。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |
| 2026-04-27 | `src/student/transformer/dynamic_physical_core_torch.py` | 修改 | batch 模式下逐样本调用 `_newmark_step_single(...)` 并 `torch.stack` 返回，保证梯度仍可从响应 loss 回传到每个样本的 `theta_t` 和 Transformer 参数。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 修改 | 在 `build_training_model(...)` 构造 `PhysicalTemplateConfig` 时新增 `enabled_params=str(cfg.enabled_params)`，使 `alpha_y_cp3` 等控制点参数能够被模板构造模块显式识别，不再依赖 `sys.argv` 兜底解析。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点扩展 |
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 修改 | 保证训练脚本和 `evaluate_transformer_vs_baselines.py` 通过 checkpoint 恢复模型时，都能根据 `cfg.enabled_params="alpha_y_cp3"` 正确生成 `K_y_cp_tpl`，避免评估阶段缺失三维控制点刚度模板。 | 阶段 5：Transformer 第一阶段模型评估与 baseline 对比 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-27 Asia/Singapore | `src/student/transformer/spatiotemporal_physics_encoder.py` | 修改 | 在 `SpatiotemporalPhysicsEncoderConfig` 中新增 `temporal_window_size` 参数；默认 `None` 保持原 full-history causal attention，不破坏旧实验。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点泛化稳定化 |
| 2026-04-27 Asia/Singapore | `src/student/transformer/spatiotemporal_physics_encoder.py` | 修改 | 将 temporal attention mask 从单纯 causal mask 扩展为 causal sliding-window mask；当 `temporal_window_size=W` 时，`theta_t` 只能 attend 到最近 W 个时间步。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点泛化稳定化 |
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 修改 | 在 `TransformerPhysicalTrainConfig` 和 argparse 中新增 `--temporal-window-size` 参数，并在构造 `SpatiotemporalPhysicsEncoderConfig` 时传入该参数。 | 阶段 5：Transformer 动态物理参数 alpha_y 控制点泛化稳定化 |
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 修改 | 在训练日志中打印 `temporal_window_size`，便于区分 full-history 模型和 fixed-window 模型。 | 阶段 5：Transformer 训练可复现实验管理 |
| 2026-04-27 Asia/Singapore | `tests/test_transformer_static_conditioning_rollout.py` | 测试 / 可选修改 | 可选新增 `--temporal-window-size` 测试参数，用于验证 `alpha_y_cp3 + fixed temporal window + Newmark rollout + gradient` 链路。 | 阶段 5：Transformer fixed-window 回归测试 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 修改 | 将默认 `TRAIN_LOAD_FILES` 从 1 个训练工况扩展为 8 个训练工况：`train_case_1.dat` 至 `train_case_8.dat`；验证工况仍保持 `valid_case_1.dat`。 | 阶段 5：Transformer 动态物理参数 alpha_y_cp3 多工况泛化训练 |
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增 `TEST_LOAD_FILES`，默认使用 `test_complex_case.dat` 作为训练后自动评估的测试工况。 | 阶段 5：Transformer 训练后独立测试评估 |
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 新增 | 在 `TransformerPhysicalTrainConfig` 中新增 `run_test_after_training`、`test_load_files`、`test_max_steps`、`test_output_dir`、`test_case_name_prefix` 等训练后测试配置。 | 阶段 5：Transformer 训练与测试流程整合 |
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 新增 | 在 argparse 中新增 `--run-test-after-training`、`--test-load-files`、`--test-max-steps`、`--test-output-dir`、`--test-case-name-prefix` 参数。 | 阶段 5：Transformer 训练脚本 CLI 扩展 |
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增 `run_post_training_tests(...)`，训练完成后自动调用 `scripts/evaluate_transformer_vs_baselines.py`，使用 best checkpoint 对 test 工况进行 teacher / no-scale / static-kappa / Transformer 四响应对比。 | 阶段 5：Transformer 训练后独立测试评估 |
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增 `_read_metrics_csv(...)` 与 `_print_post_train_test_summary(...)`，用于读取评估脚本输出的 `metrics.csv`，并在训练日志末尾打印各模型相对 teacher 的 full_x/full_y 误差及相对 static baseline 的比例。 | 阶段 5：Transformer 实验结果快速检查 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 修改 | 将默认载荷文件路径从单目录平铺的 `data/load/train_case_*.dat` 改为 train / val / test 分目录结构：`data/load/train/`、`data/load/val/`、`data/load/test/`。 | 阶段 5：Transformer 多工况训练数据管理 |
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 修改 | 将训练文件名前缀修正为单下划线版本：`train_complex_case_`、`val_complex_case_`、`test_complex_case_`。 | 阶段 5：Transformer 训练/验证/测试数据目录规范化 |
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增 `build_default_indexed_load_files(...)` 和 `build_load_files_from_cfg(...)`，支持通过默认 case 数量自动生成 train / valid / test 载荷文件列表。 | 阶段 5：Transformer 多工况训练数据管理 |
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 修改 | 在 `TransformerPhysicalTrainConfig` 中新增 train / valid / test 目录、文件名前缀、文件后缀、起始编号和 case 数量配置，使日常实验主要通过脚本默认配置而非命令行参数控制。 | 阶段 5：Transformer 实验配置管理 |
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 修改 | 修改 `parse_args()` 返回值和 main 接收逻辑，新增 `test_load_files`，为训练后自动测试阶段提供默认测试工况列表。 | 阶段 5：Transformer 训练后自动评估准备 |
| 2026-04-27 Asia/Singapore | `scripts/train_transformer_physical_params_torch.py` | 新增 | 在训练启动时打印最终解析出的 train / valid / test load files，方便确认当前实验实际使用的工况文件。 | 阶段 5：Transformer 实验可复现性与调试便利性 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-28 | `src/student/transformer/physical_parameter_heads.py` | 修改 | 在 `PhysicalParameterHeadConfig` 中新增 `zero_init_last_bias`、`init_last_weight_std`、`theta_output_scale` 配置，用于控制物理参数 head 的初始输出幅值。 | 阶段 5：Transformer 动态物理参数学习稳定化 |
| 2026-04-28 | `src/student/transformer/physical_parameter_heads.py` | 修改 | 对 `BoundedPhysicalParameterHead` 的最后一层 `Linear` 使用极小随机权重初始化和零 bias 初始化，使初始 `raw_theta≈0`、`theta≈0`，从而让初始 rollout 接近 `base_static_kappa_y`。 | 阶段 5：Transformer 动态物理参数学习稳定化 |
| 2026-04-28 | `src/student/transformer/physical_parameter_heads.py` | 修改 | 在 `theta = tanh(raw_theta) * max_abs` 后加入可选 `theta_output_scale`，保留 registry 物理上界的同时支持临时保守缩放实验。 | 阶段 5：Transformer 动态物理参数学习稳定化 |

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-28 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 将默认 `core_dtype` 从 `float64` 改为 `float32`，`encoder_dtype` 保持 `float32`，并将默认 device 改为优先使用 CUDA | Transformer 动态物理参数学习：训练性能优化 |
| 2026-04-28 | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增 `valid_every` 配置与 `--valid-every` 命令行参数，用于控制验证集评估频率 | Transformer 动态物理参数学习：训练性能优化 |
| 2026-04-28 | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增 `train_cases_grad_accum(...)`，实现逐 case backward 与梯度累积，每个 epoch 仍只执行一次 `optimizer.step()` | Transformer 动态物理参数学习：训练性能优化 |
| 2026-04-28 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 修改主训练循环，使训练阶段使用逐 case 梯度累积，验证阶段按 `valid_every` 间隔执行 | Transformer 动态物理参数学习：训练性能优化 |
| 2026-04-28 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 修改 best checkpoint 选择逻辑，使只有真正执行 validation 的 epoch 才允许更新 best checkpoint | Transformer 动态物理参数学习：训练性能优化 |




# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-29 | `src/student/transformer/physical_parameter_registry.py` | 完整替换 | 删除 Transformer 动态 `alpha_y`、`alpha_y_cpN` 注册逻辑；当前 registry 只支持 `alpha_xy`，`theta_dim=1`。 | 阶段 5：alpha_xy-only 频率/模态预训练 |
| 2026-04-29 | `src/student/transformer/physical_templates.py` | 完整替换 | 删除 `K_y_template`、`K_y_cp_tpl` 构造逻辑；当前只构造 `K_xy_template`，同时保留 static corrected base `K0=S_y(0.952)K_phiS_y(0.952)`。 | 阶段 5：alpha_xy-only 动态刚度模板 |
| 2026-04-29 | `src/student/transformer/dynamic_physical_core_torch.py` | 完整替换 | 删除 `alpha_y` 与控制点刚度模板装配逻辑；当前只支持 `K_eff=K0+alpha_xy*K_xy_template`。 | 阶段 5：alpha_xy-only full-order Newmark core |
| 2026-04-29 | `src/student/transformer/frequency_losses.py` | 新增 | 新增归一化频谱 loss、soft peak frequency loss 和方向 DOF 索引工具，用于复杂工况 x/y 频率对齐。 | 阶段 5：频率/模态对齐 loss |
| 2026-04-29 | `scripts/train_transformer_physical_params_torch.py` | 完整替换 | 默认启用 `alpha_xy`；新增频域 loss 权重与 `best_score_mode=response/freq/mixed`；默认按 `freq` 选择 best checkpoint。 | 阶段 5：alpha_xy-only 训练入口 |
| 2026-04-29 | `scripts/evaluate_transformer_vs_baselines.py` | 修改 | 评估输出中的 theta 时历记录从旧 `alpha_y` 改为 `alpha_xy`。 | 阶段 5：alpha_xy-only 评估兼容 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-29 | `src/student/transformer/frequency_losses.py` | 完整替换 | 保留全局频谱 loss，并新增 `direction_observation_signal(...)`、teacher-anchored peak/trough time alignment loss、local cross-correlation lag alignment loss，以及 `peak_and_lag_alignment_loss(...)` 统一入口。 | 阶段 5：alpha_xy-only 峰值/相位对齐训练 |
| 2026-04-29 | `scripts/train_transformer_physical_params_torch.py` | 完整替换 | 在 alpha_xy-only 训练中加入 `spectrum_loss + peak_time_loss + lag_loss`，使 `freq_loss` 成为频谱、峰值时间和局部相位滞后的综合对齐目标。 | 阶段 5：alpha_xy-only 峰值/相位对齐训练 |
| 2026-04-29 | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增 CLI 参数：`--alignment-observations`、`--w-peak-time-x/y`、`--peak-time-*`、`--w-lag-x/y`、`--lag-*`，用于控制峰值时间对齐和局部 lag loss。 | 阶段 5：训练 loss 扩展 |
| 2026-04-29 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 将默认 `core_dtype` 改为 `float64`，后续 full-order physical core 默认双精度。 | 阶段 5：数值精度设置 |


# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-29 | `scripts/scan_alpha_xy_constant_phase.py` | 新增 | 新增 `alpha_xy` 常数敏感性扫描脚本；固定多个常数 `alpha_xy`，通过 `K_eff=K0+alpha_xy*K_xy_template` 做 10s full-order Newmark rollout，并输出 MSE、tip/last5 x/y local lag、zero-lag correlation 等指标。 | 阶段 5：alpha_xy 相位控制能力诊断 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-29 | `src/student/transformer/physical_parameter_registry.py` | 完整替换 | 新增 `alpha_x` 参数注册，默认启用 `alpha_x`；保留 `alpha_xy` 但默认关闭。`alpha_x` 的 `max_abs=0.15`，对应模板为 `K_x_template`。 | 阶段 5：alpha_x 相位修正能力验证 |
| 2026-04-29 | `src/student/transformer/physical_templates.py` | 完整替换 | 新增 `K_x_template` 构造逻辑，通过对 `ux/ry` 相关自由度进行对称刚度缩放有限差分得到；保留 `K_xy_template` 供后续联合训练。 | 阶段 5：alpha_x 动态刚度模板 |
| 2026-04-29 | `src/student/transformer/dynamic_physical_core_torch.py` | 完整替换 | 动态物理 core 支持 `alpha_x -> K_x_template` 和 `alpha_xy -> K_xy_template`，可单独启用或联合启用；当前推荐只启用 `alpha_x`。 | 阶段 5：alpha_x full-order Newmark core |
| 2026-04-29 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 默认 `enabled_params` 从 `alpha_xy` 改为 `alpha_x`，用于先验证 x 方向相位漂移修正能力；训练逻辑仍保持 full-order MCK/Newmark 求解。 | 阶段 5：alpha_x-only 训练入口 |
| 2026-04-29 | `scripts/evaluate_transformer_vs_baselines.py` | 修改 | selected timeseries CSV 中的 theta 输出改为泛化写法，可自动记录 `alpha_x`、`alpha_xy` 等任意启用参数。 | 阶段 5：评估兼容 alpha_x |
| 2026-04-29 | `scripts/scan_alpha_x_constant_phase.py` | 新增 | 新增 `alpha_x` 常数敏感性扫描脚本，用于固定不同 `alpha_x` 值，评估其对复杂工况 x/y lag、corr0 和 MSE 的影响。 | 阶段 5：alpha_x 相位控制能力诊断 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-04-29 | `scripts/analyze_load_dat_frequency.py` | 新增 | 新增 `.dat` 外载荷频域分析脚本，支持解析 48 节点载荷文件，提取 Fx/Fy 等分量在 tip、last-k、mean 等观测下的主频、频谱质心、周期性强度、频段能量占比和 top peaks，并输出 CSV 与频谱图。 | 阶段 5：外力频域诊断与 Load Spectral Branch 前置分析 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-04 | `scripts/analyze_load_dat_frequency_v2.py` | 新增 | 新增外力时域/频域联合分析脚本，在原 FFT 主频、频谱能量分析基础上，加入 active duration、zero crossing、effective cycles、autocorrelation、pulse-like score、sustained oscillation score 和 signal class，用于避免将短时脉冲误判为周期载荷。 | 阶段 5：外力频域诊断与 Load Spectral Branch 前置分析 |
| 2026-05-04 | `scripts/analyze_load_dat_frequency_v2.py` | 新增 | 支持 `--make-window-features` 生成与时间步对齐的滑动窗口外力频域特征，并保存为 CSV 和 NPZ，供后续训练时接入 Load Spectral Branch 或拼接到 Load Branch。 | 阶段 5：外力频域特征接入准备 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-04 | `src/student/transformer/spatiotemporal_physics_encoder.py` | 修改 | 在 Load branch 内新增可选外力频域/脉冲特征分支，支持从 `F_raw` 计算 causal 局部频域特征，并与原始 load time token 融合。 | 阶段 5：alpha_x 外力频域输入增强 |
| 2026-05-04 | `src/student/transformer/transformer_rollout_torch.py` | 修改 | 在 rollout forward 中新增可选 `load_spectral_features` 传递路径；物理 core 仍只使用原始 `F` 做 Newmark/MCK 积分。 | 阶段 5：alpha_x 外力频域输入增强 |
| 2026-05-04 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 在 training case loading 阶段直接由 `F_raw` 在线计算 load spectral features，并用 train set 统计 mean/std 后写入 encoder buffer。 | 阶段 5：alpha_x 外力频域输入增强 |
| 2026-05-04 | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增 `--use-load-spectral-features` 及相关频域窗口、频段、观测点、归一化参数；默认关闭，保持旧命令兼容。 | 阶段 5：alpha_x 外力频域输入增强 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-04 | `scripts/evaluate_transformer_vs_baselines.py` | 修改 | 在评估阶段新增从 `F_time` 在线计算 causal load spectral features 的逻辑，并将 `load_spectral_features` 显式传入 Transformer rollout，保证启用 spectral branch 的 checkpoint 在 evaluation 中输入一致。 | 阶段 5：alpha_x 外力频域输入增强 |
| 2026-05-04 | `scripts/evaluate_transformer_vs_baselines.py` | 修改 | 在评估输出中记录 `use_load_spectral_features`、`load_spectral_feature_dim`、窗口、频段和观测点信息，并在保存 NPZ 时附带 `load_spectral_features`。 | 阶段 5：alpha_x 外力频域输入增强 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-05 | `scripts/generate_frequency_sweep_loads.py` | 新增 | 新增频率扫描外力时历生成脚本，按 low/mid/high 三个频段生成 train/valid/test `.dat` 文件，格式兼容现有 48 节点载荷输入。 | 阶段 6：频率覆盖训练工况设计 |
| 2026-05-05 | `scripts/generate_frequency_sweep_loads.py` | 新增 | 默认每个频段生成 6 个训练工况、1 个验证工况、1 个测试工况，并覆盖 tip、last5、fullspan 以及 Fx/Fy 单向与耦合载荷形式。 | 阶段 6：频率泛化验证 |


# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-06 | `tests/diagnose_selected_timeseries.py` | 新增测试/诊断脚本 | 新增 selected_timeseries 诊断工具，用于计算 transformer 相对 teacher 的 MSE、幅值比例、互相关最优滞后、FFT 主频幅值比例与相位时间等效差，并输出时域叠加图、误差图和频谱图。 | STEP B：高频响应误差诊断 |


# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-06 | `tests/diagnose_selected_timeseries.py` | 新增测试/诊断脚本 | 新增 selected_timeseries 诊断工具，用于计算 transformer 相对 teacher 的 MSE、幅值比例、互相关最优滞后、FFT 主频幅值比例与相位时间等效差，并输出时域叠加图、误差图和频谱图。 | STEP B：高频响应误差诊断 |

## 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-07 | `scripts/train_transformer_physical_params_torch.py` | 新增 | 增加 `CachedNpzFile`、`fast_np_load()` 与 `_FAST_NPZ_CACHE`，用于将 `.npz` training/validation case 缓存在内存中，避免每个 epoch 重复磁盘读取。 | Transformer 动态物理参数训练加速 |
| 2026-05-07 | `scripts/train_transformer_physical_params_torch.py` | 新增 | 增加 `configure_torch_fast_runtime()`，支持 `torch.set_float32_matmul_precision`、`cudnn.benchmark` 和可选 TF32 encoder 加速。 | Transformer 动态物理参数训练加速 |
| 2026-05-07 | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增命令行参数：`--cache-npz-in-ram`、`--allow-tf32-encoder`、`--matmul-precision`。 | Transformer 动态物理参数训练加速 |
| 2026-05-07 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 建议将训练 case 读取处的 `np.load(...)` 替换为 `fast_np_load(...)`，保持原有 `.npz` 读取接口兼容。 | Transformer 动态物理参数训练加速 |
| 2026-05-07 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 建议 validation/test 阶段使用 `torch.inference_mode()` 包裹，减少无梯度评估开销。 | Transformer 动态物理参数训练加速 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-07 | `src/student/transformer/physical_parameter_heads.py` | 新增 | 新增 `PhaseGatedPhysicalParameterHeadConfig` 与 `PhaseGatedPhysicalParameterHead`，实现 `theta_total = theta_slow + g_phase * theta_fast`，并输出 `theta_aux` 保存 slow/fast/gate 分量 | Step 1：slow + phase-gated fast decomposition |
| 2026-05-07 | `src/student/transformer/spatiotemporal_physics_encoder.py` | 修改 | 新增 `use_phase_gated_decomposition`、`phase_slow_scale`、`phase_fast_scale`、`phase_gate_init_bias`、`phase_total_clip_scale` 配置；根据开关选择旧 bounded head 或新 phase-gated head | Step 1：slow + phase-gated fast decomposition |
| 2026-05-07 | `src/student/transformer/transformer_rollout_torch.py` | 修改 | `TransformerRolloutOutput` 新增 `theta_aux`，rollout 仍使用总 `theta` 进入 `DynamicPhysicalCoreTorch`，不改变 `K_eff` 接口 | Step 1：保持旧 K_eff 接口兼容 |
| 2026-05-07 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 新增 phase-gated decomposition 的命令行参数和训练配置；新增 slow smooth、fast amp、fast smooth、gate L1、gate TV、simple-case gate sparse 正则；`compute_response_loss` 接收 `theta_aux` 并把分解正则加入 `reg_loss` | Step 2：自动相位识别机制与正则约束 |
| 2026-05-07 | `scripts/evaluate_transformer_vs_baselines.py` | 修改 | 评估输出新增 `theta_aux` 统计；`selected_timeseries.csv` 和 `.npz` 保存 `alpha_x_slow/fast/gated_fast`、`alpha_xy_slow/fast/gated_fast`、`g_phase` 等分解结果 | Step 2：诊断 phase gate 是否局部激活 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-07 | `src/student/transformer/dynamic_physical_core_torch.py` | 修改 | 新增 `precompute_newmark_matrices` 配置；新增 Newmark 常量函数、`A_base` 预计算、动态刚度增量装配函数；每步积分可复用 `sym(K0)+a0M+a1C`，只叠加 `theta` 相关刚度残差 | 训练加速优化：优先级 3 |
| 2026-05-07 | `src/student/transformer/dynamic_physical_core_torch.py` | 修改 | 保留 `linear_solve_mode={solve,cholesky}`；默认建议使用 `solve`，`cholesky` 作为后续数值一致性验证后的可选加速项 | 训练加速优化：优先级 4 |
| 2026-05-07 | `src/student/transformer/frequency_losses.py` | 新增 | 新增 `FrequencyAlignmentCache`、`build_frequency_alignment_cache`、`frequency_alignment_loss_from_cache`，缓存 teacher 侧归一化频谱与 soft peak frequency | 训练加速优化：优先级 2 |
| 2026-05-07 | `src/student/transformer/frequency_losses.py` | 新增 | 新增 `PeakLagAlignmentCache`、`build_peak_lag_alignment_cache`、`peak_and_lag_alignment_loss_from_cache`，缓存 teacher 侧 peak/trough anchor、lag window 和归一化 teacher window | 训练加速优化：优先级 2 |
| 2026-05-07 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 新增 `F_spectral` 磁盘缓存，缓存 key 包含 case 路径、文件大小、mtime、`F_raw` shape、dt、频带、观测点、window size 等配置；rebuild case 后自动生成新 cache key | 训练加速优化：优先级 1 |
| 2026-05-07 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 新增 `loss_cache` 到 `TransformerTrainingCase`；case 加载后可预构建 teacher-side frequency / peak / lag cache；训练和验证阶段复用 cache | 训练加速优化：优先级 2 |
| 2026-05-07 | `scripts/train_transformer_physical_params_torch.py` | 修改 | 新增 CLI 参数：`--use-load-spectral-disk-cache`、`--load-spectral-cache-dir`、`--force-recompute-load-spectral-cache`、`--use-cached-alignment-loss`、`--fast-core-precompute-newmark`、`--linear-solve-mode` | 训练加速优化：命令行配置 |


# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-08 | `evaluate_transformer_vs_baselines.py` | 修改 | 将 `--max-steps` 默认值从 `500` 改为 `None`，使评估脚本在不显式传入 `--max-steps` 时默认使用完整时间序列，而不是截断到 5s | full-length 评估与诊断修正 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-08 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修改 | 补齐 phase-gated decomposition 训练/验证日志链路：`compute_response_loss` 中的 `phase_gate_mean`、`phase_gate_max`、`theta_fast_abs_max` 等指标现在会经 `train_cases_grad_accum(...)` 和 `evaluate_cases(...)` 聚合后写入 `training_history.csv`；同时新增 `phase_gate_active_ratio`、`theta_gated_fast_rms`、`theta_gated_fast_abs_max`，用于区分 raw fast 分支与实际 gated-fast 贡献。 | phase-gated fast residual late-phase 诊断 |
| 2026-05-08 Asia/Shanghai | `scripts/analyze_phase_gated_results.py` | 新增 | 新增 phase-gated 结果诊断脚本，读取 `training_history.csv`、`post_test/*/metrics.csv` 和 `post_test/*/selected_timeseries.csv`，汇总 case 名、x/y ratio_to_static、phase gate 激活、`alpha_x/alpha_xy_gated_fast` RMS/max，以及 tip_x/tip_y 后半段 lag/corr/RMS 幅值比例，并输出 `phase_gated_diagnosis_summary.csv/json`。 | phase-gated fast residual late-phase 诊断 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-08 Asia/Shanghai | `src/student/transformer/frequency_losses.py` | 新增 | 新增 `adaptive_phase_window_loss(...)` 及局部辅助函数：对全序列滑动窗口计算 detached local phase-drift score，自动选取 high-score windows；在这些窗口上返回可微 local lag loss、complex spectrum phase loss、RMS amp guard 和可选 `g_phase` alignment loss。 | phase-gated fast residual 自主相位修正训练 |
| 2026-05-08 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修改 | 将 adaptive phase-window hard mining 与 complex spectrum phase loss 接入 `compute_response_loss(...)`；新增 CLI 配置 `--use-adaptive-phase-window-loss`、`--phase-window-*`、`--w-adaptive-phase-*`、`--w-complex-phase-*`、`--w-complex-amp-guard-*`、`--w-phase-gate-align`，默认关闭以保持旧实验兼容。 | phase-gated fast residual 自主相位修正训练 |
| 2026-05-08 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修改 | 扩展训练/验证 history 指标，记录 `adaptive_phase_loss`、`complex_phase_loss`、`complex_amp_guard_loss`、`phase_gate_align_loss`、adaptive score、被选窗口起点范围、被选窗口 gate 均值与 gate target 均值，用于判断模型是否自主定位相位漂移窗口。 | phase-gated fast residual 自主相位修正诊断 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-09 Asia/Shanghai | `src/student/transformer/frequency_losses.py` | 修改 | `adaptive_phase_window_loss(...)` 新增 `gate_target_score_ref`，用可配置参考 score 缩放 gate alignment target，使 hard phase-drift windows 能给 `g_phase` 更明确的局部开启目标。 | phase-gated fast residual 自主相位修正训练 |
| 2026-05-09 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修改 | 新增 `--phase-window-gate-score-ref` 配置，训练时传入 adaptive phase-window loss，并在启动信息中打印该值；默认 `0.12`。 | phase-gated fast residual 自主相位修正训练 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-09 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修改 | 新增 `--phase-gate-active-threshold`，使 `phase_gate_active_ratio` 的统计阈值可由命令行配置；默认保持 `0.2`，下一轮 gate-open 训练建议设为 `0.10`。 | phase-gated fast residual 自主相位修正诊断 |
| 2026-05-09 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 优化 | 优化 `train_cases_grad_accum(...)` 日志指标累加方式：per-case backward 后不再立即对每个标量执行 CPU 同步，而是在 torch device 上完成 sum/max/min，epoch 汇总时再同步，减少训练日志统计开销且不改变训练目标和梯度路径。 | 训练速度优化 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-09 Asia/Shanghai | `src/student/transformer/dynamic_physical_core_torch.py` | 优化 | 缓存固定 Newmark 常数，新增 `newmark_step_fast(...)` 作为 rollout hot path；该路径要求输入已在 core device/dtype 上，避免每个时间步重复张量包装、shape 检查和 registry 拆分，仍保留原 `newmark_step(...)` 兼容路径。 | 训练速度优化 |
| 2026-05-09 Asia/Shanghai | `src/student/transformer/transformer_rollout_torch.py` | 优化 | static rollout 中一次性将 `theta_seq` 转为 core dtype/device，并调用 `physical_core.newmark_step_fast(...)`；未改变 dynamic core 外部物理接口、Newmark 方程、`linear_solve_mode` 或 `core_dtype=float64` 约定。 | 训练速度优化 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-09 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增 response-level no-regression guard：`--use-no-regression-guard`、`--no-regression-guard-case-keywords`、`--w-no-regression-response/lag/amp`、ratio/lag/amp 容忍参数；guard 只按 case 名称匹配 low/simple 等受保护工况，并直接惩罚响应退化、局部 lag 超限和幅值偏差。 | 高频/复杂相位强化与低频/simple 防退化 |
| 2026-05-09 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修改 | 训练/验证日志新增 `no_regression_*` 指标；`--best-score-mode` 新增 `guarded_freq`，用 `freq_loss + best_score_guard_weight * no_regression_guard_loss` 选择 best checkpoint。 | 高频/复杂相位强化与低频/simple 防退化 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-09 Asia/Shanghai | `src/student/transformer/transformer_rollout_torch.py` | 新增 | `TransformerRolloutConfig` 新增 `profile_timing` 与 `profile_timing_sync_cuda`；static rollout 在 metadata 中记录 encoder、core prepare、Newmark loop、state stack 的耗时，用于定位训练瓶颈，不改变物理接口和数值路径。 | 训练速度诊断 |
| 2026-05-09 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增 `--profile-train-timing` 与 `--profile-timing-sync-cuda`；训练/验证阶段聚合 total、model forward、encoder、core prepare、Newmark loop、loss、backward、metric accumulation、grad clip、optimizer step 等耗时，并写入 `training_history.csv`。 | 训练速度诊断 |
| 2026-05-09 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修复 | 修复计时诊断改动中 `compute_response_loss(...)` 组装 `result` 后未返回的问题，避免训练阶段出现 `loss_dict is None`。 | 训练速度诊断 |
| 2026-05-09 Asia/Shanghai | `src/student/transformer/dynamic_physical_core_torch.py` | 新增 | 新增 `newmark_step_fast_timed(...)`，仅在 profiling 开启时把 fast Newmark step 拆分计时为 effective-matrix assembly、RHS build、linear solve 和 state update；普通 `newmark_step_fast(...)` 路径不变。 | Newmark 瓶颈细分诊断 |
| 2026-05-09 Asia/Shanghai | `src/student/transformer/transformer_rollout_torch.py`, `scripts/train_transformer_physical_params_torch.py` | 修改 | rollout 汇总 Newmark 内部细分耗时并写入 metadata；训练脚本新增 `timing_newmark_assemble/rhs/solve/update_seconds` 的 train/valid history 列和控制台摘要。 | Newmark 瓶颈细分诊断 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-10 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增 auxiliary guard-only cases：`--guard-load-files`、`--guard-case-paths`、`--w-guard-case-loss`。guard case 在训练中只反传 no-regression guard 与 theta/phase 正则，不参与强相位 adaptive/complex 主 loss，用于保护 simple/low 工况不被 high/complex 相位强化牺牲。 | strong phase balanced v2 |
| 2026-05-10 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修改 | `--prepare-cases` 时会把 guard load files 一并准备到 train split，但不会加入主 train case 列表；训练 history 新增 `train_guard_*` 指标，用于检查 guard loss、guard case ratio 和 phase gate 激活。 | strong phase balanced v2 |

# 开发者日志

| 修改时间 | 涉及脚本/文件 | 增改类型 | 修改内容 | 所属阶段 |
|---|---|---|---|---|
| 2026-05-10 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 新增 | 新增 static-quality gate suppression：`--use-static-quality-gate-suppression`、`--w-static-good-gate-l1` 及 `--static-quality-*` 配置。训练时按局部窗口比较 static student 与 teacher 的 corr/lag/amp，自动识别 static 已经足够好的窗口，并在这些窗口压低 `g_phase`，不依赖 case 名称。 | automatic gate selectivity |
| 2026-05-10 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | 修改 | `training_history.csv` 新增 `static_quality_*`、`static_good_gate_mean`、`static_bad_gate_mean`、`static_gate_selectivity_gap` 等指标，用于判断 gate 是否学会在 static-good 窗口关闭、在 static-bad 窗口保持可用。 | automatic gate selectivity |

# 2026-05-10 init-checkpoint continuation update

| Time | Files | Change Type | Content | Stage |
|---|---|---|---|---|
| 2026-05-10 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Add | Add `--init-checkpoint` so a follow-up run can initialize model weights from an existing checkpoint without restoring optimizer state; this keeps LR and phase/gate loss weights adjustable for continuation experiments. | phase-gated fast residual training efficiency |

# 2026-05-10 generalization training update

| Time | Files | Change Type | Content | Stage |
|---|---|---|---|---|
| 2026-05-10 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Add | Add state-driven no-regression window guard controlled by `--use-state-window-no-regression-guard` and `--w-state-no-regression-*`; it uses local static-vs-teacher quality to protect already-good windows without relying on case-name keywords. | operating-condition generalization |
| 2026-05-10 Asia/Shanghai | `scripts/generate_random_continuous_loads.py` | Add | Add random continuous load generator that copies a reference complex-case header and produces train/valid/test `.dat` files with randomized frequency mixtures, phases, amplitudes, spatial profiles, envelopes, and chirp components plus manifest/list files. | operating-condition generalization data |

# 2026-05-11 high-frequency phase-drift update

| Time | Files | Change Type | Content | Stage |
|---|---|---|---|---|
| 2026-05-11 Asia/Shanghai | `src/student/transformer/frequency_losses.py` | Add | Add `phase_drift_rate_loss(...)`, a high-frequency-window weighted loss that penalizes both local soft lag and consecutive-window lag drift, targeting accumulated late-time phase error under sustained high-frequency excitation. | high-frequency phase retention |
| 2026-05-11 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Add | Expose `--use-phase-drift-rate-loss`, `--w-phase-drift-*`, and `--phase-drift-*` CLI/config fields; include `phase_drift_*` diagnostics in training history and add the weighted drift loss to `freq_loss`. | high-frequency phase retention |
| 2026-05-11 Asia/Shanghai | `scripts/generate_random_continuous_loads.py` | Modify | Add `--families`, `--chirp-weight-max`, and `--burst-weight-max` so the same generator can create sustained high-periodic replay families without burst/chirp dilution. | high-frequency replay data |
# 2026-05-11 from-scratch loss curriculum update

| Time | Files | Change Type | Content | Stage |
|---|---|---|---|---|
| 2026-05-11 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Add | Add optional `--use-loss-curriculum` training mode with epoch-local loss-weight scheduling for lag/peak-time, adaptive/complex phase, high-frequency phase-drift, phase-gate regularization, static-quality gate suppression, and state-window no-regression terms. Defaults remain disabled, so existing commands keep the same behavior. | from-scratch frequency/phase curriculum |
| 2026-05-11 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Modify | Train/valid evaluation now uses the scheduled epoch config when curriculum mode is enabled, and history/checkpoints record curriculum scales plus effective weights to diagnose whether phase/gate/guard ramps are active. | from-scratch frequency/phase curriculum |
# 2026-05-12 amplitude-aware high-frequency phase update

| Time | Files | Change Type | Content | Stage |
|---|---|---|---|---|
| 2026-05-12 Asia/Shanghai | `src/student/transformer/frequency_losses.py` | Add | Add amplitude-aware detached window weighting for high-frequency phase losses. High-frequency phase drift windows can now be emphasized when teacher response amplitude exceeds a configurable reference; adaptive phase hard-mining can also rank large-amplitude phase-drift windows higher. | high-amplitude high-frequency phase correction |
| 2026-05-12 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Modify | Add CLI/config fields `--phase-window-amplitude-reference/weight/power/max-weight` and `--phase-drift-amplitude-reference/weight/power/max-weight`, plus x/y-specific reference overrides; pass them into phase losses, print them at startup, and log amplitude/combined window weights in `training_history.csv`. | high-amplitude high-frequency phase correction |
| 2026-05-12 Asia/Shanghai | `scripts/train_transformer_physical_params_torch.py` | Fix | Include the missing phase-drift amplitude diagnostic keys in `compute_response_loss(...)` output and make optional diagnostic aggregation tolerate absent metric keys by logging zero. This fixes the `KeyError: phase_drift_x_amplitude_weight_mean` startup failure without changing training loss definitions. | high-amplitude high-frequency phase correction |
