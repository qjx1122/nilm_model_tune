# STATUS.md

## 当前角色
- 角色：资深电力算法专家（定义见 ROLE.md）
- 生效范围：本 session 全部任务（用户另行指定角色时覆盖）

## 当前目标
- 【已完成·收官】任务 3「调参执行」：F4（v2_do00×w96）锁定 + Test 验收四线全过（实录 20：S_test 0.0541 / F1 0.8941 / R 0.9268 / EE +5.2%；对照旧纪元 S 2.3 倍改善）。任务起点阻塞（Test EE −23%）正式关闭。
- 本任务角色：实验/调参教练（已随任务收官回归默认角色「资深电力算法专家」）

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
- [x] 2026-09-09 prepare 二跑 n=2058 三因确诊并修复（REPORT_TEST.md 执行实录 10）：①双表缺口密布（meter10 缺 21% 跨度、meter1 缺≈48 万格、最长双净段仅 3.4h）→ 弃「最长段」改全量拼接留痕（schema v3）；②选段索引空间混用 bug；③sum skipna 假零（→min_count=1，回归测试拦截）；resample 显式 origin=epoch；pytest 14 passed
- [x] 2026-09-09 prepare 三跑 n=8,849,796 判读：**红旗解除**（aggOffW 354.8/327/400.3、corr 0.42-0.53、evt/day 5-6，实录 11）+ fillna(value,limit=N) 全轴限额 bug 确诊（240 万格只填 49 格→146 万碎段）→ v4 整段桥接（schema v4、双缺口回归测试、pytest 14 passed）；沙箱重克隆事故恢复（fetch+逐文件哈希对账，零丢失）
- [x] 2026-09-09 prepare 四跑 n=10,377,651 判读（实录 12）：结构健康（456 缝/最大段 49.1 天/保留 91.7% 跨度/+153 万桥接格回归/身份指标三验稳定）但 evt/day 翻倍 12.55/12.42/16.24、平均事件 93s→35s（ON 样本数与绝对 kWh 不变→v4 补0 切碎煮沸确诊）→ v5 kettle 短缺口 ffill（schema v5、事件内微缺口回归测试、pytest 14 passed）
- [x] 2026-09-09 prepare 五跑 v5 复验**全过→数据锁定**（实录 13）：evt/day 4.73/4.49/5.31、平均事件 ~106s、绝对事件数与 v3 交叉验证一致（2386≈2393 等）；kWh 较 v3 +14% 系煮沸中掉线格回归真值；aggOffW/corr 第四次稳定；n 较 v4 −1=头部缺口诚实剔除；Test 预算重置 2 次触碰，旧调参结论降级为待复核假设
- [x] 2026-09-09 baseline 摸底 ×3 seeds on v5 判读（实录 14）：Test EE −0.090/−0.094/−0.060（旧纪元最终模型 −0.234 未过验收→新 baseline 全门槛过，数据修复红利）；纪律事故：eval_test 缺省 True 致未带 --test 仍碰 Test→触碰#1 记账+缺省翻 False；val ON 样本仅≈36 个→seed 方差警示；evaluate.py 升级注入 best_epoch_val（history.json 已有完整 val 指标）
- [x] 2026-09-09 evaluate.py val KPI 回传判读（实录 15）：val F1 0.918/P 1.0/R 0.848 **三种子完全同值**（33 ON 中 28/33 离散化，非稳定性质示）；val S 0.0477±0.0060（F1 项恒定 0.0328、MAE 项可忽略→S 方差≈全来自 |val EE|）；val EE −7.1%±3.0% 与 test EE −8.1%±1.5% 方向一致无爆炸漂移；双探针方案+configs/baseline_100k.yaml 入库（pyyaml 单变量校验过）
- [x] 2026-09-09 双探针判读（实录 16）：A（v2_do00 平移）S 三种子配对全胜（ΔS −0.007/−0.030/−0.008）、val EE −7.1%→+1.0%（归零）、F1≥基线，代价 val MAE 变差（7.60 vs 3.66，composite 选型已知性质）；B（100k）S 打平（0.049±0.013 vs 0.048±0.006）、2.7× 代价→弃；configs/tuning_v5.yaml 入库（搜索空间以 A 邻域为中心+val 30000+train 30k+gates 不变）
- [x] 2026-09-09 重搜 32 trials 判读（实录 17）：32/32 过门槛（守门员失区分度属预期）；top-1 trial20 S=0.0400（w96 d64 h4 do0 bs128 lr5e-4）；w96 系 top-10 占 7（旧锁 w128 的新数据反例）；F1 分布恢复 0.860-0.903（val30000 生效）；EE 近零复现；单 seed 噪声→细搜多 seed；configs/fine_v5 五配置入库（校验过）；沙箱第二次重置事故恢复（零丢失）
- [x] 2026-09-09 细搜批次 1 判读（实录 18）：F0 0.0522±0.0045 夺冠（σ 最小+EE +0.0003±0.0062 死零+跨纪元）；FB 0.0604 垫底（调参>不调参坐实）；trial20 单 seed 0.0400→复核 0.0567（赢家诅咒）；F0 MAE 8.93±5.35 披露（单种子盆地，S 近盲不淘汰）；F4=v2_do00×w96 入库（唯一差异 window，校验过）；Test 预注册（seed 7000/--test 恰一次/验收口径）
- [x] 2026-09-09 细搜批次 2 判读→**锁定 F4**（实录 19）：0.0472±0.0042 预注册判定树命中；MAE 5.80±0.77（F0 盆地治愈）、F1 0.8943 六变体最高、P 0.9102 最高；EE +1.7% 换 F1/MAE/σ 三赢；最终配置=w96+d64nhead8L2ff128do0bs64lr3e-4wd1e-4 25/5（两纪元洞察杂交）；Test 预算 #2/2 审计完成
- [x] 2026-09-09 **Test 终局验收通过，任务 3 收官**（实录 20）：seed 7000 预注册一跑，S_test=0.0541（≤0.0622）、F1 0.8941、R 0.9268、|EE| 5.2%，无分布漂移；REPORT.md v1.0 创建、TUNING_GUIDE v2.0 重写（战史）、README 生产推荐更新、Test 预算 2/2 审计闭合
- [x] 2026-09-09 文档维护（用户指令）：REPORT_TEST.md 全部 25 个执行实录节补齐用户执行命令（20 处插入，标注「2026-09-09 补充留痕」；路径按用户机器实录：旧纪元 D:\datasets\ukdale_prepared.npz、h5 与 v5 纪元 D:\Work\testPython\datasets\；CLI 参数逐脚本核对 diagnose --npz / inspect --path / parse --h5-path）；复验 24 含命令块 + 1 指针节可解析
- [x] 2026-09-09 流程复盘（用户指令）：执行实录遗漏命令的 ROLE.md 条款层归因（实录 21 沉淀）——五层叠加：收尾条款只沉淀结果结论/职责边界无人负责命令归档/验收标准缺档案可复现维度/极简偏好裁剪/台账无字段+STATUS 滚动覆盖；改进建议待用户裁定（ROLE.md 收尾条款+验收标准补丁）
- [x] 2026-09-09 prepare 首跑 n=345 确诊秒级相位差并修复（REPORT_TEST.md 执行实录 9）：meter1=:15 vs meter10=:18 精确 join 拼不上；改统一 6s 网格 resample 对齐；schema_version→2；--mains-ids 默认→1；偏移 3s 回归测试；pytest 13 passed
- [x] 2026-09-09 metadata 全文回传→定表号（REPORT_TEST.md 执行实录 8）：mains=meter1 单表、kettle=meter10；meter2=锅炉回路（纠正 1,2 假设）；meter54=1s mains 备选

## 进行中
- （用户侧）git pull 后跑 tune.py 重搜（32 trials，30-60min GPU），回传门槛统计+Top-5
- （本侧）无阻塞；判读 aggOffW/corr 定数据地基是否修复

## 下一步（TODO）
1. （可选后续，非必需）REPORT.md / TUNING_GUIDE.md 已固化两纪元全部结论；如需继续：prepare/diagnose 泛化到其他 house 或电器、F4 邻域继续深挖（注意 Test 预算已耗尽，新探索需新立项+新预算规则）、或按 REPORT.md §1 口径投入生产验证
2. 无未决阻塞；本分支（arena/01a07f1d-nilm-model-tune）保持推送最新，可随时合并/续接（同 seeds 42/2024/7）→ 定 tune.py 重搜方案：A 胜→搜索以 do0/nhead8/bs64/lr3e-4 邻域为中心；B 胜（100k 显著优）→重搜用大 train 预算；均不胜→以 baseline 为锚全空间粗搜；val 预算一律扩 30000。Test 预算剩 1 次（最终锁定用）
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
- 2026-09-09（决策·全量拼接）：UK-DALE 双表缺口密布，最长双净段仅小时级 → 弃「只取最长连续段」，改剔除缺口行后全量拼接、接缝/段数/最大段留痕（schema v3）；残缺事件为已接受代价
- 2026-09-09（踩坑·sum 假零）：DataFrame.sum(axis=1) 默认 skipna 会把全 NaN 缺口静默写成 0W——多表合并必须 min_count=1（回归测试桥接断言拦截，未流入真实数据）
- 2026-09-09（踩坑·pandas fillna）：fillna(value, limit=N) 的 limit 是全轴总限额（method 未指定时按整轴计），不是每段限额；ffill(limit=N) 是每段头部 N 格——缺口桥接必须 run-length 整段语义（v4 _bridge_short_gaps）
- 2026-09-09（决策·kettle 桥接值=前值）：短缺口 ffill 而非补 0——壶 ~99% 时间关断（前值=0，「关断即 0」语义自动保持），煮沸中掉线保持 ~2300W 不断流；补 0 会把一次煮沸切成 ~3 片 35s 碎片（evt/day 12.55 vs 真实 5.56，ON 样本总数不变为铁证）
- 2026-09-09（决策·数据锁定 v5）：ukdale_prepared_v2.npz（schema v5）为唯一口径；旧 npz（mains 1+2 锅炉双计时代）全部 KPI/Test/调参记录作废归档；旧调参结论（c2/v2/nhead8 等）降级为待复核假设
- 2026-09-09（决策·Test 预算重置）：新数据纪元 Test 触碰预算=2 次（旧 Test 基于作废数据不计数；与原纪律同构）；baseline 摸底不碰 Test
- 2026-09-09（记账·Test 触碰#1）：baseline 摸底 3 seeds 实际碰了 Test（eval_test 缺省 True，指令语义误记所致）——按旧纪元「阶段0b=触碰1」同构记账；剩余 1 次给最终锁定模型；缺省已翻 False（冻结成为代码默认）
- 2026-09-09（决策·val 噪声对策）：val 6000 样本仅 ≈36 个 ON（≈2 次事件）→ F1/EE 噪声大；调参选型一律多种子均值 + composite S，单 seed 单 epoch val 指标不作依据
- 2026-09-09（决策·双探针先行）：重搜前先跑两个廉价对照——A=旧优胜 v2_do00 整体平移（测旧洞察可迁移性）+ B=baseline_100k（测数据量杠杆，单变量 max_train 30000→100k）；配对 seeds 42/2024/7 控制种子方差
- 2026-09-09（发现·S 判别力集中）：6000 样本 val 下 S 的 F1 项恒定（0.0328）、MAE 项可忽略 → S 方差≈全来自 |val EE|（33 ON 离散化）→ 重搜必须扩 val 至 30000（ON≈180）
- 2026-09-09（决策·重搜方案）：搜索空间以 v2_do00 邻域为中心（探针 A 旧洞察可迁移）；train 保持 30k（探针 B 无显著收益）；val 扩 30000（判别力）；细搜阶段 v2_do00 必入作锚；S 的 MAE 项≈可忽略（0.4×MAE/2000）→ composite 选型会牺牲点误差换 EE/F1，属已知性质如实记录
- 2026-09-09（发现·w96 崛起）：重搜 top-10 中 w96 占 7（top-4 全 w96），旧纪元锁的 w128 在 v5 数据上被压过——桥接语义改变窗口边界所致（假设），细搜 F1-F3 全为 w96 与 F0(w128) 配对验证
- 2026-09-09（决策·细搜含双锚）：F0=v2_do00（旧优胜，实测 30000-val 口径）+ FB=baseline 架构（协议归一 25/5+composite）——没有锚的排名无法区分「搜索发现」与「单 seed 运气」
- 2026-09-09（判读·批次1三结论）：①旧冠军跨纪元夺冠（EE 死零+σ 最小，领先属判断非铁证：top-4 差距在噪声内）；②FB 垫底=调参价值坐实；③搜索 top-1 赢家诅咒（0.0400→0.0567）——多 seed 复核纪律必要
- 2026-09-09（预注册·Test 最终一跑）：锁定配置 × seed 7000（新鲜族）× --test 恰好一次；验收 S_test≤val均值+0.015 且 F1≥0.75/R≥0.70/|EE|≤0.15；预算就此耗尽
- 2026-09-09（锁定·F4）：w96+d64nhead8L2ff128do0bs64lr3e-4wd1e-4（25/5 composite val30000）——预注册判定树命中+五分量同时第一+σ 最小；EE +1.7% 为明确权衡（门槛内 9 倍富余）；两纪元洞察杂交（旧冠军协议×新搜索窗口发现）
- 2026-09-09（验收·Test 通过）：S_test=0.0541 四线全过、无漂移（Δ+0.0069<0.015）；任务 3 原始阻塞（实录 5 时代 EE −23%）关闭；数据纪元 v5 + 配置 F4 成为推荐稳定版本（REPORT.md §1）
- 2026-09-09（角色切换·收官）：任务 3 角色「实验/调参教练」随验收通过卸任，回归默认「资深电力算法专家」（ROLE.md）；依据 BOOTSTRAP v2.1 收尾仪式落盘
- 2026-09-09（决策·就地补充命令）：用户明确指令在执行实录内补命令 → 覆盖台账「只追加」惯例一次；每处插入标注补充日期保持审计性；命令来源=当轮指令与实录自带命令块回溯，CLI 参数按当前脚本核对
- 2026-09-09（发现·数据杠杆边界）：30k→100k train 在 6000-val 分辨率下无可测收益（配对 1 胜 2 负）；模型当前更受容量/正则而非数据量约束的假设待细搜后复核
- 2026-09-09（踩坑·沙箱重克隆）：平台可整箱重克隆沙箱（本地 commit 链消失、/tmp 清空）；远端分支是唯一可靠真值——回合初 git log + git ls-remote 对账，恢复=fetch+逐文件哈希比对+reset
- 2026-09-09（决策·网格对齐）：多表秒级相位差是 UK-DALE 常态，对齐必须先 resample 到统一网格再 join，精确时间戳 join 不可用；data_spec schema_version 升 2 标记口径变化
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
