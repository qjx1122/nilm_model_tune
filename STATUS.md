# STATUS.md

## 当前角色
- 角色：资深电力算法专家（定义见 ROLE.md）
- 生效范围：本 session 全部任务（用户另行指定角色时覆盖）

## 当前目标
- （空）等待用户下达任务。本 session 已按 BOOTSTRAP.md v2.1 完成开局仪式。

## 已完成
- [x] 2026-09-08 开局仪式：git 现状核对 / 续接文件读取 / 环境检查
- [x] 修复台账一致性：`ROLE.md.md` → `ROLE.md`（git mv，内容不变）；新建 `STATUS.md` 骨架
- [x] （历史，仓库现状）Transformer-NILM 代码 / 配置 / smoke 产物齐备，`reports/smoke/result.json` 显示用户机器 smoke test 通过（device cuda，test MAE 61.1 / R² 0.906，合成信号，非真实数据集结果）

## 进行中
- 依赖恢复受限：本 sandbox 为 Linux + Python 3.11，无 conda；`download.pytorch.org` CPU 源被网络阻断，PyPI 完整版 torch 体积过大（数 GB）暂未安装 → 本环境不可跑训练/调优真实实验；README 中 smoke 验证可等有任务需要时再装 CPU 版执行

## 下一步（TODO）
1. 等待用户下达本 session 任务；立项时登记本行并按任务协议执行
2. （可选）如需在本 sandbox 跑代码验证：pip 安装 CPU 版 torch 与 requirements（改走可用镜像/索引），跑 `python scripts/run_smoke.py` 对照已有产物

## 决策记录 / 踩坑
- 2026-09-08：`ROLE.md.md` 与台账文件名（`ROLE.md`）不一致，且最近 commit 意图即「上传ROLE.md」→ 执行 `git mv ROLE.md.md ROLE.md`，无损、可回退
- 2026-09-08：真实 UK-DALE 实验需要数据 + GPU/conda 环境，本 sandbox 两者皆无 → 涉及真实 KPI 的任务只能输出代码/配置/方案，结论须标注口径，不伪造数据
- 2026-09-08：BOOTSTRAP.md 指定 `STATUS.md` / `ROLE.md` / `session/NILM_AC_session_complete.md` / `REPORT_TEST.md` / `REPORT.md` 为事实来源与沉淀文件，所有结论只追加写入这些文件

## 关键文件路径
- 协议：`BOOTSTRAP.md`（v2.1）、`ROLE.md`（角色库，默认角色=资深电力算法专家）
- 续接：`STATUS.md`（本文件）、`session/NILM_AC_session_complete.md`（会话纪要，session 收尾追加）
- 报告：`REPORT_TEST.md`（专题，追加式）、`REPORT.md`（稳定结论）
- 代码：`src/`（data / model / metrics / trainer / experiment）、`scripts/`（train / evaluate / tune / inspect_h5 / run_smoke）、`configs/`（baseline.yaml / tuning.yaml）
- 运行：`run_baseline.ps1` / `run_tuning.ps1` / `run_real.ps1`（Windows + Conda）
- 产物：`reports/smoke/`（smoke test 基线）、`reports/`（实验输出目录）
- 测试：`tests/test_model.py`；依赖：`requirements.txt`
