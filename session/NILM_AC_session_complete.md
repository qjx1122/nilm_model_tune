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

## [2026-09-09] 会话纪要（五跑 v5 复验全过→数据锁定，主线重启）
- 目标：判读 v5 复验输出，裁定数据锁定与主线重启
- 本会话角色：实验/调参教练（判读+锁定裁定）
- 完成项：
  - v5 全过：evt/day 4.73/4.49/5.31（v4 12.55）、平均事件 ~106s、绝对事件数与 v3 交叉验证一致、桥接格数/接缝 456/最大段 49.1 天与 v4 完全一致、kWh +14%=煮沸掉线格回归真值、身份指标第四次稳定
  - n 较 v4 −1：头部缺口 ffill 无前值→诚实剔除（起始=壶表首个真实读数 22:28:18）
  - 数据锁定：v5 npz+data_spec 唯一口径；旧 npz 及 KPI/Test/调参记录作废归档；Test 预算重置 2 次
  - 主线重启方案：baseline.yaml ×3 seeds（42/2024/7）摸底（命令已入 STATUS）；REPORT_TEST 实录 13、STATUS、纪要同步；commit+push
- 关键决策：旧调参结论降级为待复核假设（不继承）；REPORT.md 拟在最终锁定+Test 通过后把实录 5-13 浓缩为数据制备章节
- 未决问题：用户回传三份 baseline KPI → 判读后定重搜 vs 平移复核策略
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；configs/baseline.yaml / REPORT_TEST.md / STATUS.md

## [2026-09-09] 会话纪要（baseline 摸底判读：EE 里程碑 + Test 触碰记账）
- 目标：判读 v5 数据 baseline ×3 seeds；补 val KPI 通道
- 本会话角色：实验/调参教练（判读+纪律记账）
- 完成项：
  - 判读：Test EE −0.090/−0.094/−0.060（旧纪元最终模型 −0.234 未过验收）→ baseline 全门槛过（|EE|/F1/R），数据修复红利超旧纪元全部调参之和；MAE 6.88-8.80 优于旧 final 9.84
  - 纪律事故处理：eval_test 缺省 True → 未带 --test 仍碰 Test 3 次；记账=新纪元触碰#1（与旧纪元阶段0b同构），剩 1 次；缺省翻 False + train.py help 同步（责任在我方指令语义误记，已在实录 14 致歉记录）
  - val KPI 通道：history.json 逐 epoch 已含完整 val_*；evaluate.py 升级注入 best_epoch_val
  - 噪声定量：val ON 样本 ≈36 个 → seed 方差/best_epoch 5-22 波动的根因；选型纪律=多种子+S
  - REPORT_TEST 实录 14、STATUS、纪要同步；commit+push
- 关键决策：Test 触碰记账结构（摸底#1+最终#2）；eval_test 默认冻结
- 未决问题：用户回传 3 份 evaluate.py 输出（val KPI）→ 定平移复核 vs 重搜
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；src/experiment.py / scripts/evaluate.py / scripts/train.py / REPORT_TEST.md / STATUS.md

## [2026-09-09] 会话纪要（val KPI 判读 + 双探针方案）
- 目标：判读 baseline ×3 的 val KPI；定平移复核 vs 重搜策略
- 本会话角色：实验/调参教练（判读+实验设计）
- 完成项：
  - 判读：F1/P/R 三种子完全同值（28/33 离散化，非稳定性；零假警报）；S=0.0477±0.0060，分解后 F1 项恒定、MAE 项可忽略 → S 方差≈全来自 |val EE| → 6000 val 下 S 判别力集中在最噪指标；val EE −7.1%±3.0% 与 test EE −8.1%±1.5% 一致（旧纪元爆炸漂移消失）
  - 实验设计：双探针——A=旧优胜 v2_do00（w128 d64 nhead8 do0 bs64 lr3e-4 25/5）平移复核 ×3；B=baseline_100k（唯一差异 max_train 100k）×3；配对 seeds 42/2024/7
  - configs/baseline_100k.yaml 入库（pyyaml 平铺校验：唯一差异 data.max_samples_train）
  - REPORT_TEST 实录 15、STATUS、纪要同步；commit+push
- 关键决策：重搜前双探针先行；重搜 val 预算扩 30000
- 未决问题：双探针回传后定 tune.py 搜索空间中心与 train 预算
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；configs/baseline_100k.yaml / REPORT_TEST.md / STATUS.md

## [2026-09-09] 会话纪要（双探针判读 + 重搜方案定稿）
- 目标：判读双探针六份输出；定 tune.py 重搜方案
- 本会话角色：实验/调参教练（判读+搜索设计）
- 完成项：
  - 探针 A 判读：S 三种子配对全胜（0.0329±0.0060 vs 0.0477±0.0060）、val EE −7.1%→+1.0%（归零）、F1≥基线；代价 val MAE 变差（7.60±4.4 vs 3.66±0.30，s2024 13.85/R²0.687 待细搜澄清；S 的 MAE 项≈可忽略系已知性质）
  - 探针 B 判读：S 打平（0.049±0.013 vs 0.048±0.006）、EE 改善不一致、2.7× 代价 → 弃 100k（数据杠杆边界记录）
  - configs/tuning_v5.yaml 入库：搜索空间以 v2_do00 邻域为中心裁剪（w96/128/192、d64/128、h4/8、L1/2、ff128/256、do0/0.1、bs64/128、lr2/3/5e-4、wd1e-5/1e-4、25/5）+ val 30000 + train 30k + gates/复合分不变
  - REPORT_TEST 实录 16、STATUS、纪要同步；commit+push
- 关键决策：重搜三要点（A 邻域中心 / 30k train / val 30000）；细搜必含 v2_do00 锚
- 未决问题：tune.py 32 trials 回传后判读 → 细搜设计
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；configs/tuning_v5.yaml / REPORT_TEST.md / STATUS.md

## [2026-09-09] 会话纪要（重搜判读 + 细搜批次1设计 + 第二次沙箱重置恢复）
- 目标：判读 tune.py 32 trials；设计细搜
- 本会话角色：实验/调参教练（判读+设计）
- 完成项：
  - 判读：32/32 门槛（守门员失区分度）；top-1 trial20 S=0.0400（w96 d64 h4 do0 bs128 lr5e-4）；w96 top-10 占 7；F1 分布 0.860-0.903（val30000 判别力恢复）；EE 近零复现；单 seed 噪声→细搜多 seed
  - 细搜批次 1：configs/fine_v5/{f0_v2do00,f1_t20,f2_t29,f3_t11,fb_basearch}.yaml 入库（pyyaml 逐项+协议不变量校验通过）；全部 val30000+25/5+composite；×seeds 42/2024/7
  - 沙箱第二次重置（HEAD→7824bb4、/tmp 清空）：fetch+逐文件哈希对账（全一致）+reset 528a01c，零丢失
  - REPORT_TEST 实录 17、STATUS、纪要同步；commit+push
- 关键决策：细搜双锚（F0 旧优胜 + FB 基线架构）——排名必须有锚才能区分发现与运气
- 未决问题：细搜 15 runs 回传判读 → 锁定或批次 2
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；configs/fine_v5/ / REPORT_TEST.md / STATUS.md

## [2026-09-09] 会话纪要（细搜批次1判读 + 批次2 F4 + Test 预注册）
- 目标：判读细搜批次 1（5 配置 ×3 seeds）；定锁定路径
- 本会话角色：实验/调参教练（判读+预注册）
- 完成项：
  - 批次 1 判读：F0（v2_do00）0.0522±0.0045 夺冠（σ 最小、EE +0.0003 死零、跨纪元）；FB 垫底 0.0604（调参>不调参坐实）；trial20 赢家诅咒（0.0400→0.0567±0.0077）；F0 MAE 8.93±5.35 披露（单种子盆地，S 近盲）；top-4 差距在噪声内（统计诚实记录）
  - 批次 2：F4=v2_do00×w96 入库（与 F0 唯一差异 window_size，pyyaml 校验过）；判定树预注册（F4<0.0522→锁 F4 否则锁 F0）
  - Test 预注册：seed 7000 --test 恰一次；验收 S_test≤val+0.015/F1≥0.75/R≥0.70/|EE|≤0.15
  - REPORT_TEST 实录 18、STATUS、纪要同步；commit+push
- 关键决策：批次 2 只补一个最高信息量变体（冠军协议×w96 交叉点）；Test 协议预注册防事后挑选
- 未决问题：F4 回传 → 锁定 → Test 一跑 → 验收/收官
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；configs/fine_v5/f4_v2do00_w96.yaml / REPORT_TEST.md / STATUS.md

## [2026-09-09] 会话纪要（批次2判读→锁定F4；Test最终一跑交付）
- 目标：判读 F4（v2_do00×w96）；按预注册判定树锁定；交付 Test 协议
- 本会话角色：实验/调参教练（判读+锁定裁定）
- 完成项：
  - F4 判读：0.0472±0.0042 命中判定树（<0.0522）；score/MAE/RMSE/F1/precision 五项第一；MAE 盆地治愈（8.93±5.35→5.80±0.77）；EE +1.7%±1.8% 为三赢权衡；统计诚实（ΔS 未达严格显著，三重非单证据支撑）
  - 锁定 F4=w96+d64nhead8L2ff128do0bs64lr3e-4wd1e-4 25/5 composite（两纪元杂交冠军）
  - Test 预算审计：#1 摸底（实录14）+ #2 本次=耗尽；验收线 S_test≤0.0622/F1≥0.75/R≥0.70/|EE|≤0.15
  - Test 命令交付（seed 7000 --test reports\final_v5\f4_test）；REPORT_TEST 实录 19、STATUS、纪要同步；commit+push
- 关键决策：F4 锁定（预注册规则，无事后挑选）；S_test 由本侧按公式计算
- 未决问题：Test 回传 → 验收裁定 → 收官 or 漂移诊断
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；configs/fine_v5/f4_v2do00_w96.yaml / REPORT_TEST.md / STATUS.md

## [2026-09-09] 会话纪要（Test 终局验收通过，任务 3 收官）
- 目标：裁定 Test 终局一跑；执行收官仪式
- 本会话角色：实验/调参教练（判读+验收裁定；收官后回归默认角色）
- 完成项：
  - S_test=0.0541（0.4×6.4156/2000+0.4×(1−0.8941)+0.2×0.0523）四线全过（≤0.0622/F1 0.8941/R 0.9268/|EE| 5.2%），无分布漂移（Δ+0.0069<0.015）；对照旧纪元 S≈0.1244 → 2.3 倍改善；EE 从 −23.4% → +5.2%
  - 任务 3 原始阻塞（Test EE −23% 未过验收）正式关闭；Test 预算审计闭合（#1 摸底+#2 终局=2/2）
  - 收官产物：REPORT.md v1.0（创建：推荐版本/数据口径 v5/KPI 冻结/七条稳定结论/版本史）、TUNING_GUIDE.md v2.0（整篇重写：六条铁律/SOP/两纪元战史/五层 bug 表/踩坑清单）、README 生产推荐段、STATUS 收尾仪式（角色切换落盘）
  - commit+push
- 关键决策：验收通过=任务 3 完结；v5+F4 成为推荐稳定版本；Test 预算耗尽（后续探索需新立项+新预算）
- 未决问题：无阻塞；可选后续=其他 house/电器泛化、F4 邻域深挖（需新预算规则）、生产验证
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；REPORT.md / TUNING_GUIDE.md / REPORT_TEST.md（实录 20）/ README.md / STATUS.md

## [2026-09-09] 会话纪要（文档维护：执行实录补齐用户命令）
- 目标：按用户指令检查 REPORT_TEST.md 各执行实录是否含具体用户执行命令，缺则补
- 本会话角色：资深电力算法专家（默认角色，文档维护）
- 完成项：
  - 盘点 25 个执行实录节：4 节原本含命令块（1补充/3补充系列）、1 节指针（2补充）、20 节缺失
  - 20 处插入「用户执行命令（2026-09-09 补充留痕）」块：路径按两纪元实录（旧 D:\datasets\ukdale_prepared.npz；h5/v5 D:\Work\testPython\datasets\）；CLI 逐脚本核对（diagnose_split --npz、inspect_h5 --path、parse_nilmtk_metadata --h5-path --house）；实录 16 的 evaluate 循环用修正引号版并注明裸词踩坑
  - Python 复验：24 节含命令块 + 1 指针节可解析，全覆盖
  - STATUS/纪要同步；commit+push
- 关键决策：用户指令就地补充 → 覆盖「只追加」惯例一次，插入处标注日期保审计
- 未决问题：无
- 相关文件/分支：arena/01a07f1d-nilm-model-tune；REPORT_TEST.md / STATUS.md
