# STATUS.md

## 当前角色
- 角色：资深电力算法专家（定义见 ROLE.md）
- 生效范围：本 session 全部任务（用户另行指定角色时覆盖）

## 当前目标
- 【进行中·用户专题】调参方案设计：Transformer-NILM（UK-DALE House1 Kettle，Seq2Point）全流程调参方案（未跑真实数据 → 数据制备 → baseline → 粗搜 → 细搜 → 锁定 → Test 一次）
- 本任务角色：实验/调参教练（REPORT_TEST.md 正式方案 + TUNING_GUIDE.md 人话版双文档交付）

## 已完成
- [x] 2026-09-08 开局仪式：git 现状核对 / 续接文件读取 / 环境检查
- [x] 修复台账一致性：`ROLE.md.md` → `ROLE.md`（git mv，内容不变）；新建 `STATUS.md` 骨架
- [x] （历史，仓库现状）Transformer-NILM 代码 / 配置 / smoke 产物齐备，`reports/smoke/result.json` 显示用户机器 smoke test 通过（device cuda，test MAE 61.1 / R² 0.906，合成信号，非真实数据集结果）
- [x] 2026-09-08 任务立项：确认 4 项口径（起点=未跑真实数据 / 算力=消费级 GPU 数小时 / 交付=REPORT_TEST.md+TUNING_GUIDE.md / 目标=业务口径优先 F1+能量误差）；盘点代码缺口 4 项（见决策记录）
- [x] 2026-09-08 交付 v1：`REPORT_TEST.md` 新增专题「Transformer-NILM 调参方案设计」；新建 `TUNING_GUIDE.md`（人话版方法论 v1.0）

## 进行中
- 方案已交付，等待用户评审 / 批准执行（执行需要：真实数据在用户机器、tune.py 等 4 项代码改造、sandbox 不可跑训练）

## 下一步（TODO）
1. 用户评审方案 → 批准后：任务 2「调参执行配套改造」立项（改 tune.py 按 val 综合分排名 / fit() 可选选择指标 / prepare_ukdale.py 数据制备脚本等 4 项改造，见 REPORT_TEST.md「代码改造清单」）
2. 用户机器：数据制备 → baseline ×3 seeds → 粗搜 → 细搜 → 锁定 → Test 一次，结果追加进 REPORT_TEST.md 同专题（执行实录节）并回报
3. 真实结果稳定后：KPI 口径与推荐配置进入 REPORT.md；TUNING_GUIDE.md 补「战史」

## 决策记录 / 踩坑
- 2026-09-08：`ROLE.md.md` 与台账文件名（`ROLE.md`）不一致，且最近 commit 意图即「上传ROLE.md」→ 执行 `git mv ROLE.md.md ROLE.md`，无损、可回退
- 2026-09-08：真实 UK-DALE 实验需要数据 + GPU/conda 环境，本 sandbox 两者皆无 → 涉及真实 KPI 的任务只能输出代码/配置/方案，结论须标注口径，不伪造数据
- 2026-09-08：BOOTSTRAP.md 指定 `STATUS.md` / `ROLE.md` / `session/NILM_AC_session_complete.md` / `REPORT_TEST.md` / `REPORT.md` 为事实来源与沉淀文件，所有结论只追加写入这些文件
- 2026-09-08（任务口径）：用户确认①尚未跑真实数据→方案覆盖全流程；②消费级 GPU（3060/4060 级）单卡、数小时预算→搜索规模按单 trial 10–60s 核算；③双文档交付；④验收业务口径优先：ON/OFF F1 + 能量误差为主，MAE 为辅（非纯 val_mae 单指标）
- 2026-09-08（盘点·代码缺口，均写入方案）：
  1. `tune.py` 按 **test MAE** 排名（数据泄漏口径，代码注释与 README 都声明应按 val）→ 需改按 val 综合分排名
  2. `fit()` 早停/存 checkpoint 固定按 val MAE → 业务口径需支持按「综合分」选最优 epoch（可选改造）
  3. 数据管道缺口：真实数据入口只有 `load_simple_npz`（npz{aggregate,target}），仓库无 ukdale.h5→npz 制备脚本；README 的 `--data-path ukdale.h5` 描述与实际 npz 入口不一致（run_real.ps1 用 `UKDALE_PREPARED_NPZ` 环境变量，方向正确）
  4. 产物与文档不一致：README 声称调参产物 `best_config.yaml`，代码实际写 `best_trial.json`；`tune.py` 的 `val_objective` 列为占位字符串
- 2026-09-08（方案要点）：KPI 综合分 S=0.4·(MAE/2000)+0.4·(1−F1)+0.2·|EE|，先硬门槛（F1≥0.75、recall≥0.7、|EE|≤0.15，待 baseline 摸底校准）再按 S 排序；Test 集冻结（House1 后 15%），全程只在阶段 0 与最终锁定各碰一次

## 关键文件路径
- 协议：`BOOTSTRAP.md`（v2.1）、`ROLE.md`（角色库，默认角色=资深电力算法专家）
- 续接：`STATUS.md`（本文件）、`session/NILM_AC_session_complete.md`（会话纪要，session 收尾追加）
- 报告：`REPORT_TEST.md`（专题，追加式）、`REPORT.md`（稳定结论）
- 代码：`src/`（data / model / metrics / trainer / experiment）、`scripts/`（train / evaluate / tune / inspect_h5 / run_smoke）、`configs/`（baseline.yaml / tuning.yaml）
- 运行：`run_baseline.ps1` / `run_tuning.ps1` / `run_real.ps1`（Windows + Conda）
- 产物：`reports/smoke/`（smoke test 基线）、`reports/`（实验输出目录）
- 测试：`tests/test_model.py`；依赖：`requirements.txt`
