# STATUS.md

## 当前角色
- 角色：资深电力算法专家（定义见 ROLE.md）
- 生效范围：本 session 全部任务（用户另行指定角色时覆盖）

## 当前目标
- 【进行中·用户任务】数据地基修复：表号已定（mains=meter1 单表、kettle=meter10，REPORT_TEST.md 执行实录 8）→ 待用户跑 prepare 生成 ukdale_prepared_v2.npz + diagnose 复验，回传三份输出
- 本任务角色：实验/调参教练（数据完整性核查，不猜表号）

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
- [x] 2026-09-08 **调参执行·粗搜完成（用户机器回传）+ 细搜设计交付**：32 trials、门槛 31/32；top-5 判读 + 细搜设计 v1（锚 trial13、批次 1 七变体 ×3 seeds、批次 2/锁定规则）已落盘 REPORT_TEST.md「执行实录 1」
- [x] 2026-09-08 **完整 top-5 参数行回传判读 → 细搜批次 v1.1 落盘**（REPORT_TEST.md 执行实录 1 补充）：锚 best_epoch=4 快收敛、do0.2 最强单因子证据（trial18 MAE/F1/P/R 全优但 EE +5.5% 反超失败——S 拆解算例）、P<R 假阳偏多、nhead8 独苗验证、ff128=2×d64 容量检查、lr 只测下行
- [x] 2026-09-08 **细搜批次 1 完成 → 判读与批次 2 落盘**（REPORT_TEST.md 执行实录 2）：V0 锚种子噪声极大（0.0645±0.0479）→ 粗搜单跑 0.0372 系优胜者偏差；v2_do00 均值/分量全面最优且更稳（新基准）；v1_do02 未复现 trial18 → dropout 非增益源；窗口族稳定但天花板低；批次 2 = C1(lr2e4)+C2(nhead4)+v2 加跑，每配置 10 fresh seeds 6000–6009
- [x] 2026-09-08 **批次 2 完成 → 锁定候选 c2 并交付收官 SOP**（REPORT_TEST.md 执行实录 3）：c2(nhead4, n=9) S 0.0381±0.0102 全场最优+分布最紧+P/R 拉平 0.920/0.920+EE −0.0025；差值 0.0053<2×合并SEM 0.0105 未达严格显著，但系第二次同构证据（更稳+分量全优）→ 判定性锁定；c1(lr2e4) 与 v2 无差 → lr 维持 3e-4；收官 = final_c2.yaml(epochs30/pat7) ×3 seeds 7000–7002 val 复核 → train.py 新增 `--test` 开关做 Test 一次
- [x] 2026-09-08 `train.py` 新增 `--test`（显式 eval_test 覆盖，仅在最终 Test 一步使用；py_compile 通过）
- [x] 2026-09-08 **fv2 配对对照完成 → 分支 1 命中 → 最终锁定 v2(nhead8)**（REPORT_TEST.md 执行实录 3 补充 3）：同 seeds 7000–7002 三方对照 fv2(0.0407/EE+0.005) vs final_c2(0.0476/−0.054) vs fc2_25(0.0560/−0.075) → 7000 系变差系 nhead4 特异性；v2 池化 n=18 S≈0.043 EE≈−0.006 跨族一致胜出；Test 预注册（seed 7000、25/5、`--test` 一次）已交付
- [x] 2026-09-08 **Test 回传 → 判定 v2 未通过验收**（REPORT_TEST.md 执行实录 4）：S_test 0.124（MAE 9.84/RMSE 131.7/R² 0.666/F1 0.811/R 0.729/EE −0.234）vs val 0.0407 → Δ≈0.084≫0.015；test 业务门槛 F1/recall 过、|EE| 0.234 FAIL → 「Train/Val 好+Test 差」时间漂移模式（漏报 ON 事件、非过拟合非随机）；校准不可行（val EE≈0 无恒定偏差）；Test 触碰计数=1，不做基于 test 的选型
- [x] 2026-09-08 新增 `scripts/diagnose_split.py`（纯数据分段诊断：事件数/ON 功率/能耗/日均，冒烟通过）
- [x] 2026-09-08 **方向 A 诊断完成并落盘**（REPORT_TEST.md 执行实录 5）：test 段真实漂移坐实（evt/day 5.31、on_frac 0.0088、kWh/天 0.515 vs train 0.344/val 0.359 → +43~50%）；**发现 aggregate 红旗**（agg 均值≈target 均值、agg p95=1.0W → aggregate 几乎只含 kettle，非真实总负荷；NILM 前提存疑，val MAE 3–4W 或为泄漏产物）→ 脚本升级（kWh /1000 修正 + 新增 agg_off_mean/corr 列）并冒烟通过，待用户重跑确认
- [x] 2026-09-08 diagnose_split.py 升级：kWh 单位修正（/1000）+ 新增 agg_off_mean_w / corr_agg_target 列（判别 aggregate 是否泄漏的探针）；合成对照验证（正常版 aggOffW≈350 vs 泄漏版 0/corr 1.0）
- [x] 2026-09-09 prepare tz 真因定位+修复：用户重跑 f10c795 版 --list-meters 暴露真实报错 Cannot interpret 'datetime64[ns, Europe/London]' → 根因 _normalize_ts_index 对 tz-aware dtype 调 np.issubdtype（沙箱复现同错）；修复= DatetimeIndex 先行分支 + try/except；顺带修 med_dt ns/us 单位陷阱（改 Timedelta 口径）；新增 tz-aware 回归测试，pytest（除 test_model）12 passed
- [x] 2026-09-09 --list-meters 54/54 通过（用户回传全文，落盘 REPORT_TEST.md 执行实录 7）：meter1/2/3 apparent 全程 10M 级；meter10 active 8.94M；meter54 为 1s 表 56.7M；无 meter0。重要修正：6s mains 无 active 列，aggregate 默认 apparent 路线（留痕）
- [x] 2026-09-09 metadata 全文回传→定表号（REPORT_TEST.md 执行实录 8）：mains=meter1 单表、kettle=meter10；meter2=锅炉回路（纠正 1,2 假设）；meter54=1s mains 备选

## 进行中
- （用户侧）跑 prepare 生成 ukdale_prepared_v2.npz（几分钟）+ diagnose_split 复验 + data_spec 关键字段，一次贴回三份输出
- （本侧）无阻塞；判读 aggOffW/corr 定数据地基是否修复

## 下一步（TODO）
1. 用户：跑 prepare（--mains-ids 1 --kettle-meter-id 10 --out ukdale_prepared_v2.npz，不覆盖旧 npz）→ 回传输出
2. 用户：跑 diagnose_split.py --npz ukdale_prepared_v2.npz + 回传 data_spec.json 关键字段（mains_meter_ids_used/时间范围/n_output）→ 本侧判读 aggOffW/corr（证伪口：aggOffW≈0 则推翻 meter1=mains 假设）
3. 判读红旗：agg_off_mean≈0 且 corr≈1 → 确认 aggregate 泄漏 → 修数据制备（prepare_ukdale.py --list-meters 核对 mains 表号 → 重新生成 npz → 人工抽查 aggregate 一天曲线）→ 全部 KPI 重启（先 baseline 再走搜索，Test 协议重置一次并记录）；若数据无误（agg_off_mean 数百 W）→ 回到漂移结论：方向 B（记录教训收尾）或 C（改切分）
4. 收尾仪式：session 纪要追加、STATUS 更新、commit/push（视红旗结论而定）
5. （可选，后续）torch 2.14 的 enable_nested_tensor UserWarning 噪音清理（不影响结果）

## 决策记录 / 踩坑
- 2026-09-08：`ROLE.md.md` 与台账文件名（`ROLE.md`）不一致，且最近 commit 意图即「上传ROLE.md」→ 执行 `git mv ROLE.md.md ROLE.md`，无损、可回退
- 2026-09-08：真实 UK-DALE 实验需要数据 + GPU/conda 环境，本 sandbox 两者皆无 → 涉及真实 KPI 的任务只能输出代码/配置/方案，结论须标注口径，不伪造数据
- 2026-09-08：BOOTSTRAP.md 指定 `STATUS.md` / `ROLE.md` / `session/NILM_AC_session_complete.md` / `REPORT_TEST.md` / `REPORT.md` 为事实来源与沉淀文件，所有结论只追加写入这些文件
- 2026-09-08（任务口径）：用户确认①尚未跑真实数据→方案覆盖全流程；②消费级 GPU（3060/4060 级）单卡、数小时预算→搜索规模按单 trial 10–60s 核算；③双文档交付；④验收业务口径优先：ON/OFF F1 + 能量误差为主，MAE 为辅（非纯 val_mae 单指标）
- 2026-09-08（盘点·代码缺口）：见 REPORT_TEST.md 专题（4 项，全部已修复）
- 2026-09-08（方案要点）：KPI 综合分 S=0.4·(MAE/2000)+0.4·(1−F1)+0.2·|EE|，先硬门槛（F1≥0.75、recall≥0.7、|EE|≤0.15，待 baseline 摸底校准）再按 S 排序；Test 集冻结（House1 后 15%），全程只在阶段 0 与最终锁定各碰一次
- 2026-09-08（配套改造·实现决策）：综合分收敛到 `src/objective.py`（纯函数无 torch 依赖）；fit 默认 mae 向后兼容；tune 默认不评估 test（eval_test=false）；`--trials` 覆盖供烟雾；prepare_ukdale.py 契约= NILMTK 风格布局 + `--list-meters` 探表号，布局不符报错+键树提示不硬解
- 2026-09-08（粗搜判读）：S 构成中 MAE 项仅 0.001 级、F1 主导 + EE 次之 → 细搜主战场是 F1 2–4 个百分点与 EE 收敛；trial18 MAE/F1 全优却输 trial13（EE +0.055 vs +0.012）→ 综合分防住了「MAE 刷分、总量跑偏」；w256 家族 F1 齐 0.868 落后 → 排除长窗，w160 封顶；trial13 vs 18 同家族 S 差 0.0022 → 细搜判定噪声基线 ≈0.002–0.003，改善需 mean±std 判定
- 2026-09-08（细搜 v1.1 判读）：锚 trial13 best_epoch=4（~9 epoch 早停）→ 收敛极快+轻度过拟合；dropout 0.1→0.2 是最大单因子嫌疑（trial18 证据：MAE 3.80/F1/P=R 0.931/best_epoch 14）；锚 P=0.900<R=0.931 假阳偏多（与 EE +1.2% 同向）→ do 方向正确；S 拆解算例：trial18 F1 赢 0.0063 但 EE 恶化 +0.043 倒输 0.0022 → 三列同看；nhead8 独苗、ff=2×d64 非标准 → 入批次验证；lr 5e-4 两败 → 只测 2e-4；锚家族单 trial 46–65s → 5 seeds/变体预算可行（40 runs ≈ 35–50 min）
- 2026-09-08（细搜批次 1 判读）：V0 锚 5-seed 噪声 0.0645±0.0479 → 粗搜 top-1 单跑 0.0372 是优胜者偏差上端；v2(do0) 各分量全面优（MAE 3.60/F1 0.913/R 0.924/EE −0.029）+ 更稳（σ0.013）→ 新基准；v1(do0.2) 未复现 trial18 → dropout 非增益源；w96/w160 稳定但 recall 0.85 天花板 → 弃窗口方向；lr2e4/nhead4 在 do0.1 基准有方向性改善 → 批次 2 于 do0 基准单测（C1/C2）
- 2026-09-08（批次 2 判读）：c2(nhead4) S 0.0381±0.0102(n=9) 最优且分布最紧，MAE/F1/P 三项全场最优，P/R 拉平 0.920/0.920（修掉 P<R 假阳老问题），EE −0.0025 近零；vs v2 差 0.0053 < 2×合并SEM 0.0105 未达严格显著 → 但为第二次同构证据（更稳+分量全优），判定性锁定 c2，如实标注非统计显著；c1 与 v2 无差 → lr 保持 3e-4；搜索收敛（w/do/lr/ff 方向探明无增益），不开新批次
- 2026-09-08（工程）：summarize_fine.py 曾只扫 v*_s* 致 c* 目录静默漏报（已修复为任意 <变体>_s<种子> + 打印扫描计数）；train.py 新增 --test 显式开关（仅在最终 Test 一步用，落实 Test 冻结纪律）
- 2026-09-08（30/7 复核判读）：final_c2(epochs30/pat7, n=3) S 0.0476±0.009 劣于 c2(25/5, n=9) 0.0381±0.010，劣化几乎全来自 EE（−0.054 vs −0.0025，≈3×合并 SEM 显著转负）；best_ep 9.9→12.3 因 pat7 多等 7 轮选到平台期更晚点；30/7 无稳健性收益反而引入系统性能量低估 → 复核未通过，维持 25/5 口径锁定 c2（与全部选型证据同口径）【后撤回：跨种子批比较混杂】
- 2026-09-08（种子批次效应 → 最终锁定 v2）：fc2_25(7000系, 0.0560/EE−0.075) 与 final_c2(0.0476/−0.054) 同种子对照 → 30/7 vs 25/5 无差，撤回前条「30/7 更差」；c2 vs v2 的 0.0381 系 6000 系运气，池化打平；fv2_25(7000系, 0.0407/EE+0.005) 配对 → 7000 系变差系 nhead4 特异性 → 锁 v2(nhead8)（跨种子族一致 + EE 池化近零）；教训：多 seed 批次效应 + 配对比较 + 池化估计；单批（6000 系）会出假阳性选型
- 2026-09-08（踩坑·torch 依赖）：PyPI torch 2.14.0+cu130 Linux wheel 不在 wheel 内带 CUDA 运行库，需按 `nvidia-*` 包补齐；cu13 系 pip 包已改名（`nvidia-cuda-runtime-cu13` 等旧名报「请用不带后缀新名」）；cudnn/nccl/cusparselt/nvshmem 仍用 `-cu13` 后缀且版本由 torch METADATA 钉死；cufft/cusparse/cusolver/curand 用不带后缀新名（soname .12）；`nvidia-nccl`（新名）sdist 损坏 → 装 `nvidia-nccl-cu13==2.30.7`；小坑：nvidia-cuda-profiler-api 不含 libcupti，需 `nvidia-cuda-cupti`。安装时用 `--only-binary :all:` 避免 sdist 回退
- 2026-09-08（踩坑·工程）：①`reports/smoke/*` 是 git 跟踪的历史产物（commit 7824bb4），本地验证先备份、跑完恢复，勿覆盖；②仓库历史误提交 `__pycache__/*.pyc` → 本次清理出库并加 `.gitignore`；③prepare 脚本 `meter_groups` 曾对 h5py Group 对象二次索引报 TypeError → 已修（单测捕获）；④合成 h5 测试的 mains 需包含 kettle 事件才物理自洽
- 2026-09-09（踩坑·pandas 时区）：真实 NILMTK ukdale.h5 经 `pd.read_hdf` 读出的 index 是 tz-aware DatetimeIndex（Europe/London），`np.issubdtype(tz_dtype, np.integer)` 直接抛 `TypeError: Cannot interpret …` —— numpy 不认 pandas 扩展 dtype；修法= DatetimeIndex 先行处理 + try/except 包裹。另：pandas 3 默认时间单位是 us 而非 ns，`asi8/astype(int64)` 数值差 1000 倍，采样间隔必须用 Timedelta 口径求，勿写死 /1e9
- 2026-09-09（决策·不猜表号）：转述 metadata 出现与 h5 矛盾的 meter 0 且 meter2 身份不明 → 坚持要全文，拿到 ground truth 才发 prepare。兑现价值：全文证明 meter2=锅炉，纠正 1,2 双总表假设
- 2026-09-09（决策·单总表）：mains 只用 meter1（2 为锅炉回路，加进去 double count）；新 npz 另存 v2 不覆盖旧文件（旧文件关联历史 KPI/Test 记录）；diagnose 设证伪口
- 2026-09-09（踩坑·工具）：同一回合内并行发给同一文件的多个 edit_file 只会活一个（互相覆盖）→ 同文件多处改动必须串行或单次原子写入（bash/python 整段改），且 commit 前必须 grep 验活

## 关键文件路径
- 协议：`BOOTSTRAP.md`（v2.1）、`ROLE.md`（角色库，默认角色=资深电力算法专家）
- 续接：`STATUS.md`（本文件）、`session/NILM_AC_session_complete.md`（会话纪要，session 收尾追加）
- 报告：`REPORT_TEST.md`（专题，追加式）、`REPORT.md`（稳定结论）、`TUNING_GUIDE.md`（调参教学）
- 代码：`src/`（data / model / metrics / objective / trainer / experiment）、`scripts/`（train / evaluate / tune / prepare_ukdale / inspect_h5 / run_smoke）、`configs/`（baseline.yaml / tuning.yaml）
- 运行：`run_baseline.ps1` / `run_tuning.ps1` / `run_real.ps1`（Windows + Conda）
- 产物：`reports/smoke/`（smoke 基线，git 跟踪勿覆盖）、`reports/`（实验输出目录）
- 测试：`tests/`（test_model / test_objective / test_prepare_ukdale）；依赖：`requirements.txt`
.txt`
