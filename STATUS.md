# STATUS.md

## 当前目标
- 本 session 任务（已完成）：恢复开发环境 + 冒烟验证代码链路。下一 session 默认目标见「下一步」。

## 已完成
- [x] 2026-09-05 session（arena/01a06b72-nilm-model-tune）：
  - 开局仪式：确认分支/认证/环境；新建 `STATUS.md`（本文件）。
  - 建 `.venv`（Python 3.11.2）并安装依赖：torch 2.14.0+cu130（CPU 运行，cuda available=False）、numpy 2.4.6、pandas 3.0.5、h5py 3.16.0、scikit-learn 1.9.0 等。
  - `python scripts/run_smoke.py` **通过**（~11s，CPU）：best_epoch=3，test R²=0.909、MAE=59.8、RMSE=172.9、SAE=0.0349、P/R/F1=1.0。产物 `reports/smoke/{best.pt,history.json,result.json}` 已重新生成并提交。
  - `tests/test_model.py::test_forward` 逻辑验证通过（输出形状 (8,)；该文件为 pytest 风格，pytest 未装，用等价调用直跑）。
  - 工程卫生：新增 `.gitignore`（.venv/、__pycache__、data/、checkpoints/、logs/）；`git rm --cached` 移除原误提交的 `src/__pycache__/*.pyc`。
  - README 增补「Linux / macOS（无 Conda 沙箱/CI）」环境小节。

## 进行中
- 无。

## 下一步（TODO）
1. （默认）用户下达新任务：真实数据实验需 UK-DALE 数据（`data/` 为空；README §2 有 NILMbench/HF 下载源），跑通后依 README §9 流程出 KPI 并沉淀 REPORT_TEST.md。
2. 如需跑单元测试：`pip install pytest`（requirements.txt 未含）。
3. 可选：修复 `src/model.py:48` 的 UserWarning（`norm_first=True` 时 `enable_nested_tensor` 无效，来自新版 torch，仅提示无碍）。

## 决策记录 / 踩坑
- PyTorch CPU index（download.pytorch.org/whl/cpu）在本沙箱 TLS 报错（EOF）；回退默认 PyPI 安装成功，无 GPU 时 torch 自动 CPU 运行——无需专门装 CPU 版。
- `unittest discover` 对 pytest 风格用例（tests/test_model.py 裸函数）发现 0 个用例；pytest 不在 requirements.txt，故用直接调用验证。
- 冒烟指标（R²=0.91、F1=1.0）为合成信号结果，**不是** UK-DALE 科学结论，禁止写入 REPORT.md。
- 原提交误含 `.pyc` 字节码文件，已解除跟踪；未来提交前应 `git status` 确认无 `__pycache__`。

## 关键文件路径
- 会话协议：`BOOTSTRAP.md`；续接文件：`STATUS.md`
- 会话纪要：`session/NILM_AC_session_complete.md`
- 专题报告：`REPORT_TEST.md`（未创建，等首个专题）；稳定结论：`REPORT.md`（未创建）
- 冒烟测试：`scripts/run_smoke.py`；产物 `reports/smoke/`
- 配置：`configs/baseline.yaml`、`configs/tuning.yaml`；环境说明：`README.md` §3
