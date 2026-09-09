# NILM_AC 会话纪要

> 依据 BOOTSTRAP.md：每 session 追加一条，不新建文件。首建于 2026-09-08。

## [2026-09-08] 会话纪要
- 目标：
  1. 用户专题「调参方案设计」：Transformer-NILM（UK-DALE House1 Kettle）全流程调参方案（双文档交付）
  2. 用户任务「调参执行配套改造」：方案中代码缺口 #1–#4 的实现与验证
- 本会话角色：默认「资深电力算法专家」；任务 1 声明「实验/调参教练」；任务 2 声明「工程实现工程师」
- 完成项：
  - 开局仪式：ROLE.md.md → ROLE.md 正名、STATUS.md 建档、环境核查（UK-DALE 数据/GPU 均无 → 口径说明）
  - 交付 REPORT_TEST.md 专题 v1 + 初始化 TUNING_GUIDE.md v1.0（4 项口径经 ask_user 确认）
  - 配套改造 #1–#4 全部完成：src/objective.py、trainer/experiment 判据参数化、tune.py 重写（val 综合分排名 + Test 冻结 + 留档 + best_config.yaml）、prepare_ukdale.py（h5→npz + data_spec.json）、tuning.yaml 对齐、README 修正、run_tuning.ps1 --trials、.gitignore + 清理误入库 pyc
  - 验证：pytest 10 passed；run_smoke 默认路径 PASS（CPU MAE 59.8）；tune 复合口径烟雾 PASS（门槛过滤/排序/产物正确）；sandbox 内 /tmp/tvenv 完成 torch 2.14.0+cu130 CPU 环境搭建
  - 提交 8 个 commit 已推送（arena/01a07f1d-nilm-model-tune）
- 关键决策：
  - Test 冻结（搜索中默认不评估 test，只碰 2 次）；排序只用 val 综合分 S=0.4·(MAE/2000)+0.4·(1−F1)+0.2·|EE|，业务硬门槛先行
  - fit() 默认 select_metric=mae 保持向后兼容；综合分收敛 src/objective.py 共享纯函数
  - prepare_ukdale.py 只承诺 NILMTK 风格布局 + --list-meters，不猜官方原始布局
- 未决问题：真实 ukdale.h5 是否已下载未确认；真实 KPI/推荐配置待用户机器执行后回填（步骤见 REPORT_TEST.md「方案执行更新·待办」）
- 相关文件/分支：arena/01a07f1d-nilm-model-tune（commit 6324dc1…收尾 commit）；REPORT_TEST.md / TUNING_GUIDE.md / STATUS.md / ROLE.md / session/NILM_AC_session_complete.md（本文件）

## [2026-09-09] 会话纪要
- 目标：定位用户重跑 f10c795 版 prepare_ukdale.py --list-meters 报出的真实错误并修复
- 本会话角色：工程实现工程师
- 完成项：
  - 真因定位：用户回传 `Cannot interpret 'datetime64[ns, Europe/London]'` → 沙箱复现：真实 NILMTK h5 经 pd.read_hdf 读出的 index 是 tz-aware DatetimeIndex，`np.issubdtype(tz_dtype, np.integer)` 直接抛 TypeError（numpy 不认 pandas 扩展 dtype）
  - 修复 scripts/prepare_ukdale.py：_normalize_ts_index 增加 DatetimeIndex 先行分支（naive→localize UTC / tz-aware→tz_convert UTC）+ np.issubdtype 包 try/except；med_dt 改 Timedelta 口径（避开 pandas3 us-vs-ns 单位陷阱 /1e9 写死）
  - 新增回归测试 test_prepare_nilmtk_tz_aware_index（Europe/London tz-aware 合成 h5，断言无「读取失败」且 median_sample_gap_sec==6.0）；pytest tests/（除 test_model）12 passed
  - STATUS.md 更新（当前目标/已完成/进行中/下一步/踩坑）；本纪要追加；commit+push
- 关键决策：
  - 采样间隔一律用 Timedelta.total_seconds() 口径，不再依赖 asi8/astype(int64)+写死除数
- 未决问题：待用户 git pull 后重跑 --list-meters 回传输出（期望 54 表正常列出）；之后走 prepare 生成新 npz + diagnose_split 复验
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；scripts/prepare_ukdale.py / tests/test_prepare_ukdale.py / STATUS.md

## [2026-09-09] 会话纪要
- 目标：判读用户回传的 --list-meters 全文，定 prepare 制备参数
- 本会话角色：实验/调参教练
- 完成项：
  - 确认 tz 修复生效：54/54 表读出；关键表（1/2/3 apparent 全程、10 active、54 为 1s 表 56.7M、无 meter0）落盘 REPORT_TEST.md 执行实录 7
  - 重要修正：6s mains 无 active 列，「mains active 合并」不可行 → aggregate 默认 apparent 路线（留痕），meter54 列备选
  - 卡点判定：metadata 映射只有转述且含矛盾（"meter 0"、meter2 身份不明）→ 不猜表号，请用户重跑 parse 贴全文
  - STATUS.md 更新；本纪要追加；commit+push
- 关键决策：制备参数必须以全文 metadata 映射为准；apparent aggregate 的能量口径偏差在 data_spec 留痕
- 未决问题：待用户回传 parse 全文 → 定 --mains-ids/--kettle-meter-id → 发 prepare 命令
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；REPORT_TEST.md / STATUS.md

## [2026-09-09] 会话纪要
- 目标：判读 metadata 全文，定 prepare 制备参数
- 本会话角色：实验/调参教练
- 完成项：
  - metadata 全文（53 条）判读落盘 REPORT_TEST.md 执行实录 8：编号无偏移（meter10=壶双吻合）；mains=meter1 单表；meter2/3/8/25=回路 CT 非 mains；meter0 无 h5 组；meter54=1s mains 备选
  - 决策：--mains-ids 1 --kettle-meter-id 10，输出 ukdale_prepared_v2.npz（不覆盖旧文件）；diagnose 设证伪口
  - STATUS.md 更新；本纪要追加；commit+push
- 关键决策：纠正旧假设 1,2 双总表（2 为锅炉）；不猜表号原则兑现价值
- 未决问题：待用户回传 prepare 输出 + diagnose 输出 + data_spec 关键字段
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；REPORT_TEST.md / STATUS.md

## [2026-09-09] 会话纪要
- 目标：判读 prepare 首跑三份输出（n=345 事故），修复后重发制备指令
- 本会话角色：工程实现工程师（对齐 bug 修复）+ 实验/调参教练（判读）
- 完成项：
  - 确诊秒级相位差：kettle 自身无 NaN、对齐后 886 万 NaN + 起始秒 :15 vs :18；meter1=mains 未被证伪（aggW 基线数百 W）
  - 修复：_to_6s_grid resample 对齐、schema_version→2、--mains-ids 默认→1、docstring/README 同步、偏移 3s 回归测试；pytest 13 passed
  - REPORT_TEST.md 执行实录 9、STATUS.md、纪要同步；commit+push
- 关键决策：网格对齐为强制口径（schema v2）；旧 n=345 的 v2 npz 作废覆盖
- 未决问题：待用户重跑 prepare + diagnose 回传（预期 n≈800-900 万）
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；scripts/prepare_ukdale.py / tests/test_prepare_ukdale.py / README.md

## [2026-09-09] 会话纪要（二跑判读与三因修复）
- 目标：判读 prepare 二跑输出（n=2058），修复后重发制备指令
- 本会话角色：工程实现工程师（prepare 修复）+ 实验/调参教练（判读）
- 完成项：
  - 三因确诊：①双表缺口密布（meter10 缺 240 万格/21% 跨度、meter1 缺≈48 万格、最长双净段 3.4h）；②选段索引空间混用 bug；③sum skipna 假零（沙箱复现+回归测试拦截）
  - 修复：min_count=1、段统计统一索引空间、全量拼接留痕（schema v3：n_segments/n_concat_breaks/largest_segment_samples/union_grid_samples/dropped_gap_samples）、resample origin="epoch"、段内间隔抽查（Timedelta 口径）
  - 新增缺口拼接回归测试（桥接语义+事件剪断+段统计）；pytest 14 passed
  - REPORT_TEST.md 实录 10、STATUS.md、纪要同步；commit+push
- 关键决策：全量拼接替代最长段策略（残缺事件为已接受代价）；前两版 v2 npz 作废
- 未决问题：待用户重跑 prepare + diagnose（预期 n≈850-890 万、拼接数千处）
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；scripts/prepare_ukdale.py / tests/test_prepare_ukdale.py

## [2026-09-09] 会话纪要（三跑判读红旗解除 + v4 整段桥接 + 沙箱重克隆恢复）
- 目标：判读 prepare 三跑输出（n=8,849,796）；修复 fillna 语义 bug
- 本会话角色：实验/调参教练（判读）+ 工程实现工程师（v4 修复）
- 完成项：
  - 红旗解除判定：aggOffW 354.8/327.0/400.3、corr 0.42/0.49/0.53、evt/day 5.56/5.24/6.18——meter1=mains / meter10=kettle / NILM 前提闭环（实录 11）
  - fillna(value,limit=N) 全轴限额 bug 确诊（沙箱 pandas 3.0.5 验证：240 万格只填 49 格、146 万碎段）→ v4 _bridge_short_gaps 整段桥接；schema v4；双缺口回归测试；pytest 14 passed
  - 沙箱重克隆事故：reflog 只剩 clone+checkout、HEAD 漂到 7824bb4、/tmp/dvenv 消失；远端分支完好（3206715）→ fetch+逐文件哈希对账一致+reset 恢复，零丢失
  - REPORT_TEST 实录 11、STATUS、纪要同步；commit+push
- 关键决策：缺口桥接=整段语义（v4）；远端为真值的回合初对账 SOP
- 未决问题：用户重跑 v4 prepare+diagnose（预期 n≈950-1080 万、接缝骤降、身份指标不变）；test 段壶用量 +53% 漂移待数据锁定后作为调参核心议题
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；scripts/prepare_ukdale.py / tests/test_prepare_ukdale.py / REPORT_TEST.md / STATUS.md

## [2026-09-09] 会话纪要（四跑判读：结构健康+v5 ffill 修复）
- 目标：判读 prepare 四跑输出（n=10,377,651），裁定数据是否锁定
- 本会话角色：实验/调参教练（判读+算术确诊）+ 工程实现工程师（v5 修复）
- 完成项：
  - 结构判定健康：456 缝/最大段 49.1 天/保留 91.7% 跨度/153 万桥接格回归；aggOffW/corr 第三次稳定（351.9/324.2/398.2；0.39/0.46/0.50）
  - evt/day 翻倍确诊切碎：ON 样本总数 37,169≈37,048、绝对 kWh 151.9≈151.5 不变，事件 ×2.64、平均 93s→35s → v4 补0 在煮沸中掉线处断流（掉线率 13.5%×15 样本/煮沸→期望 2 断流 吻合）；kWh/day 下降系分母膨胀
  - v5：kettle 短缺口改 ffill 前值（schema v5、kettle_cells_bridged 留痕、事件内微缺口回归测试 target[500:540].min()>1500）；pytest 14 passed
  - REPORT_TEST 实录 12、STATUS、纪要同步；commit+push
- 关键决策：kettle 桥接值=前值（ffill）；v5 与 v4 差异仅填值语义（格数/接缝均不变）
- 未决问题：用户重跑 v5 prepare+diagnose（预期 evt/day 回 5-6）→ 通过即数据锁定 → 回主线 KPI/Test
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；scripts/prepare_ukdale.py / tests/test_prepare_ukdale.py / REPORT_TEST.md / STATUS.md
