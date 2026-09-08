# STATUS.md

## 当前角色
- 角色：资深电力算法专家（定义见 ROLE.md）
- 生效范围：本 session 全部任务（用户另行指定角色时覆盖）

## 当前目标
- 【进行中·用户专题】调参方案设计：Transformer-NILM（UK-DALE House1 Kettle，Seq2Point）全流程调参方案（未跑真实数据 → 数据制备 → baseline → 粗搜 → 细搜 → 锁定 → Test 一次）
- 本任务角色：实验/调参教练（REPORT_TEST.md 正式方案 + TUNING_GUIDE.md 人话版双文档交付）
- 【进行中·用户任务】调参执行配套改造（REPORT_TEST.md 代码改造清单 #1–#4）：tune.py 按 val 综合分排名 / fit() 判据参数化 / prepare_ukdale.py 数据制备脚本 / tuning.yaml 对齐 + README 一致性
- 本任务角色：工程实现工程师（小步提交；实际运行验证；不越界改动）

## 已完成
- [x] 2026-09-08 开局仪式：git 现状核对 / 续接文件读取 / 环境检查
- [x] 修复台账一致性：`ROLE.md.md` → `ROLE.md`（git mv，内容不变）；新建 `STATUS.md` 骨架
- [x] （历史，仓库现状）Transformer-NILM 代码 / 配置 / smoke 产物齐备，`reports/smoke/result.json` 显示用户机器 smoke test 通过（device cuda，test MAE 61.1 / R² 0.906，合成信号，非真实数据集结果）
- [x] 2026-09-08 任务立项：确认 4 项口径（起点=未跑真实数据 / 算力=消费级 GPU 数小时 / 交付=REPORT_TEST.md+TUNING_GUIDE.md / 目标=业务口径优先 F1+能量误差）；盘点代码缺口 4 项（见决策记录）
- [x] 2026-09-08 交付 v1：`REPORT_TEST.md` 新增专题「Transformer-NILM 调参方案设计」；新建 `TUNING_GUIDE.md`（人话版方法论 v1.0）
- [x] 2026-09-08 配套改造立项：角色=工程实现工程师；范围=改造清单 #1–#4
- [x] 2026-09-08 **配套改造 #1–#4 全部完成并验证通过**（详细见 REPORT_TEST.md「方案执行更新」）：
  - #2 `src/objective.py`（共享综合分模块）+ `trainer.fit()`/`experiment.train_experiment()` 支持 `select_metric: mae|composite`、`data.eval_test` 开关（默认 mae+eval_test=True，向后兼容）
  - #1 `scripts/tune.py` 重写：只按 val 决策、默认冻结 Test、留档列（seed/git/runtime/val 全套）、输出 best_config.yaml；`--trials` 覆盖
  - #3 `scripts/prepare_ukdale.py`（NILMTK 风格 h5→npz + data_spec.json 口径留痕 + --list-meters）；README 数据入口与调参口径章节修正
  - #4 `configs/tuning.yaml` 对齐方案（trials 32、composite 目标、业务门槛、搜索空间）
  - 仓库卫生：移除误入库的 `__pycache__/*.pyc`，新增 `.gitignore`
  - 验证：`pytest tests/` 10 passed；`run_smoke.py` 默认路径通过（CPU，MAE 59.8，与用户 GPU 61.1 同量级）；tune 复合口径烟雾通过（门槛过滤→按 val 排序、test 列为空、best_config.yaml 可复现）；sandbox 搭好 torch 2.14.0+cu130 CPU 环境（/tmp/tvenv，不入库）；用户 smoke 历史产物已备份恢复、未改动

## 进行中
- （无）配套改造收尾完成，等待用户评审 / 下一步指令

## 下一步（TODO）
1. 用户评审配套改造；批准后按 REPORT_TEST.md SOP 在用户机器执行：数据制备（prepare_ukdale.py）→ baseline ×3 seeds → 粗搜 32 trials → 细搜 top-3×3 seeds → 锁定 + Test 一次
2. 执行结果回填 REPORT_TEST.md 同专题（执行实录），门槛阈值按 baseline 校准
3. 真实结果稳定后：KPI 口径与推荐配置进入 REPORT.md；TUNING_GUIDE.md 补「战史」
4. （可选，后续）torch 2.14 的 enable_nested_tensor UserWarning 噪音清理（不影响结果）

## 决策记录 / 踩坑
- 2026-09-08：`ROLE.md.md` 与台账文件名（`ROLE.md`）不一致，且最近 commit 意图即「上传ROLE.md」→ 执行 `git mv ROLE.md.md ROLE.md`，无损、可回退
- 2026-09-08：真实 UK-DALE 实验需要数据 + GPU/conda 环境，本 sandbox 两者皆无 → 涉及真实 KPI 的任务只能输出代码/配置/方案，结论须标注口径，不伪造数据
- 2026-09-08：BOOTSTRAP.md 指定 `STATUS.md` / `ROLE.md` / `session/NILM_AC_session_complete.md` / `REPORT_TEST.md` / `REPORT.md` 为事实来源与沉淀文件，所有结论只追加写入这些文件
- 2026-09-08（任务口径）：用户确认①尚未跑真实数据→方案覆盖全流程；②消费级 GPU（3060/4060 级）单卡、数小时预算→搜索规模按单 trial 10–60s 核算；③双文档交付；④验收业务口径优先：ON/OFF F1 + 能量误差为主，MAE 为辅（非纯 val_mae 单指标）
- 2026-09-08（盘点·代码缺口）：见 REPORT_TEST.md 专题（4 项，全部已修复）
- 2026-09-08（方案要点）：KPI 综合分 S=0.4·(MAE/2000)+0.4·(1−F1)+0.2·|EE|，先硬门槛（F1≥0.75、recall≥0.7、|EE|≤0.15，待 baseline 摸底校准）再按 S 排序；Test 集冻结（House1 后 15%），全程只在阶段 0 与最终锁定各碰一次
- 2026-09-08（配套改造·实现决策）：综合分收敛到 `src/objective.py`（纯函数无 torch 依赖）；fit 默认 mae 向后兼容；tune 默认不评估 test（eval_test=false）；`--trials` 覆盖供烟雾；prepare_ukdale.py 契约= NILMTK 风格布局 + `--list-meters` 探表号，布局不符报错+键树提示不硬解
- 2026-09-08（踩坑·torch 依赖）：PyPI torch 2.14.0+cu130 Linux wheel 不在 wheel 内带 CUDA 运行库，需按 `nvidia-*` 包补齐；cu13 系 pip 包已改名（`nvidia-cuda-runtime-cu13` 等旧名报「请用不带后缀新名」）；cudnn/nccl/cusparselt/nvshmem 仍用 `-cu13` 后缀且版本由 torch METADATA 钉死；cufft/cusparse/cusolver/curand 用不带后缀新名（soname .12）；`nvidia-nccl`（新名）sdist 损坏 → 装 `nvidia-nccl-cu13==2.30.7`；小坑：nvidia-cuda-profiler-api 不含 libcupti，需 `nvidia-cuda-cupti`。安装时用 `--only-binary :all:` 避免 sdist 回退
- 2026-09-08（踩坑·工程）：①`reports/smoke/*` 是 git 跟踪的历史产物（commit 7824bb4），本地验证先备份、跑完恢复，勿覆盖；②仓库历史误提交 `__pycache__/*.pyc` → 本次清理出库并加 `.gitignore`；③prepare 脚本 `meter_groups` 曾对 h5py Group 对象二次索引报 TypeError → 已修（单测捕获）；④合成 h5 测试的 mains 需包含 kettle 事件才物理自洽

## 关键文件路径
- 协议：`BOOTSTRAP.md`（v2.1）、`ROLE.md`（角色库，默认角色=资深电力算法专家）
- 续接：`STATUS.md`（本文件）、`session/NILM_AC_session_complete.md`（会话纪要，session 收尾追加）
- 报告：`REPORT_TEST.md`（专题，追加式）、`REPORT.md`（稳定结论）、`TUNING_GUIDE.md`（调参教学）
- 代码：`src/`（data / model / metrics / objective / trainer / experiment）、`scripts/`（train / evaluate / tune / prepare_ukdale / inspect_h5 / run_smoke）、`configs/`（baseline.yaml / tuning.yaml）
- 运行：`run_baseline.ps1` / `run_tuning.ps1` / `run_real.ps1`（Windows + Conda）
- 产物：`reports/smoke/`（smoke 基线，git 跟踪勿覆盖）、`reports/`（实验输出目录）
- 测试：`tests/`（test_model / test_objective / test_prepare_ukdale）；依赖：`requirements.txt`
