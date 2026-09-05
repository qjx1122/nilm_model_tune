# session 纪要（NILM_AC）

> 本文件为会话纪要台账：**只追加、不新建、不删除历史条目**。

---

## [2026-09-05] 会话纪要
- 目标：开局仪式 + 恢复开发环境（Linux 沙箱）+ 冒烟验证代码链路（用户选项 verify_env）。
- 完成项：
  - 开局仪式：分支 `arena/01a06b72-nilm-model-tune`（基于 main @ `858196b`），gh 认证 ✓；仓库为首次上传代码、无历史纪要；`STATUS.md` 缺失 → 按模板新建。
  - 依赖：`python3 -m venv .venv` + `pip install -r requirements.txt`；torch 2.14.0+cu130（CPU 运行）。
  - 冒烟：`python scripts/run_smoke.py` PASSED（~11s CPU；test R²=0.909，F1=1.0，合成数据）；产物已重新生成。
  - 模型单测等价验证：`test_forward` 输出形状 (8,) ✓。
  - 工程卫生：新增 `.gitignore`；`git rm --cached` 移除误入库的 `src/__pycache__/*.pyc`。
- 关键决策：
  - PyTorch CPU index TLS 失败 → 用默认 PyPI（无 GPU 自动 CPU），记录于 STATUS 决策记录。
  - 冒烟指标不视为 UK-DALE 结论，不写入 REPORT.md。
- 未决问题：
  - 真实 UK-DALE 数据实验待用户提供数据/下达任务；pytest 未列入 requirements。
- 相关文件/分支：`arena/01a06b72-nilm-model-tune`；改动见 commit「chore: 恢复 Linux 沙箱环境并冒烟验证代码链路」。
