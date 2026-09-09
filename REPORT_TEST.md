# REPORT_TEST.md — 专题报告台账（NILM_AC）

> 依据 BOOTSTRAP.md v2.1：用户 / 实验 / 验证专题一律**只追加**到本文件，按专题分节，不新建文件。
> 首次创建于 2026-09-08（协议 v2.1 建立后第一个专题）。

---

## [2026-09-08] 专题：Transformer-NILM（UK-DALE House1 Kettle）调参方案设计 v1

- **类型**：用户专题（含后续执行配套的完整实验方案）
- **本任务角色**：实验/调参教练（交付物含面向无算法背景工程师的 TUNING_GUIDE.md，本文件为正式方案与决策记录）
- **目标与假设**：
  - 输入：UK-DALE House1、6 秒采样、aggregate（总负荷）→ 输出 kettle 功率的 Transformer Encoder Seq2Point（现有 `src/` 代码）。
  - 目标：在**规范口径**（Test 全程冻结、只按 Validation 决策）下，产出一个比 baseline 更好且可复现的推荐配置；业务侧优先把「烧水壶开/关辨识准（ON/OFF F1）+ 总耗电量算得准（Energy Error）」做好，MAE 为辅。
  - 假设：用户按 README 用 Windows + Conda + 单张消费级 GPU（3060/4060 级）执行；真实数据未跑过（已确认）。
- **方法 / 数据 / 参数**：

### 0. 必须遵守的三条铁律（否则结果不算数）
1. **Test 冻结**：Test = House1 最后 15% 时间区间，全程只看一次（阶段 0 基线 + 最终锁定后），任何搜索 / 早停 / 选型只用 Validation。**现有 `tune.py` 按 test MAE 排名，属数据泄漏，执行前必须先改造（见代码改造清单 #1）。**
2. **同口径对比**：任何两个配置的对比，必须同数据切分、同预处理、同种子；跨 seed 结论报 mean±std。
3. **留档可复现**：每个 trial 记录 seed / 完整配置 / git commit / 运行时长 / 最优 epoch / 该 epoch 的 val 全套指标；结果只追加进本专题节与 `reports/tuning/`。

### 1. KPI 与目标函数（业务口径）
- 硬门槛（先过滤，防"单项刷分"）：`val F1 ≥ 0.75` 且 `val recall ≥ 0.70` 且 `|val energy_error| ≤ 0.15`。门槛在阶段 0 用 baseline val 分布校准（若 baseline 达不到，门槛下调为"不差于 baseline"，先求流程跑通）。
- 综合分（越小越好，仅用 Validation、在最优 epoch 上算）：
  `S = 0.4·(MAE/2000) + 0.4·(1−F1) + 0.2·|energy_error|`
  （MAE 归一化除 2000W：kettle 量程约 0–3000W；权重理由：业务先问"开没开对、总量对不对"，再看曲线贴近度。）
- 排序：门槛过滤 → 按 S 升序取 top；禁止直接按 test 排序。
- 报告指标全套：MAE / RMSE / R² / SAE / F1 / precision / recall / energy_error（现有 `regression_metrics` 已齐）。

### 2. 阶段 0：数据制备（**仓库缺口，需先补**）
- 现状：所有真实数据入口只接受 npz `{aggregate, target}`（`load_simple_npz`）；仓库没有 ukdale.h5 → npz 的制备脚本（代码改造清单 #3）。
- 制备规格（写脚本时照此验收）：
  1. 来源：UK-DALE House1，6 秒采样（mains 与 kettle 同频段），时间对齐后取**交集区间**；单位统一为 W。
  2. `aggregate`：House1 总负荷（若多相/多表，按官方 mains 汇总口径合并）。
  3. `target`：kettle 功率列；缺失 / NaN 段处理策略必须写进脚本注释并留痕（建议：短于 30 分钟的缺口填 0 需谨慎——kettle 关闭即 0，填 0 合理；长段缺失直接截断，不跨缺口拼接）。
  4. 输出 `ukdale_prepared.npz`（float32，一维等长），另存 `data_spec.json`（来源文件、时间范围、样本数、缺口处理记录、git hash）——**这是以后所有 KPI 的口径依据**。
  5. 替代路线：NILMbench 已处理数据（`labels_and_index.npz` 是 11 点上下文结构，与现有窗式管道不兼容，需另写适配），不如直接走官方 h5 制备，数据可得时二选一并在 data_spec.json 注明。
- 验收：npz 可被 `python scripts/train.py --config configs/baseline.yaml --data-path <npz>` 正常消费，绘图抽查 aggregate 中能看到 kettle 的 ~2–3kW 台阶事件。

### 3. 阶段 0b：Baseline ×3 seeds（先摸底，后校准门槛）
- 命令（每 seed 一遍，out 目录分开）：
  `python scripts\train.py --config configs\baseline.yaml --data-path D:\datasets\ukdale_prepared.npz --out reports\baseline_s{42,2024,7}`
- 记录：val/test 全套指标（val 取自 `history.json` 最优 epoch，test 取自 `result.json`），报 mean±std；记 runtime/trial 实测 → **回填预算核算**（见阶段 1）。
- 决策分支：
  - val MAE 量级异常大（>300W）→ 先查数据制备 / 归一化 / 对齐，不进入搜索；
  - Train 好 Val 差 → baseline 已过拟合，阶段 1 搜索空间向 dropout/wd/小模型倾斜（见 TUNING_GUIDE §5 对照表）；
  - 正常 → 用 baseline val 指标校准阶段 1 门槛，进入搜索。

### 4. 阶段 1：粗搜（随机搜索，Validation 决策）
- 范围（与 `configs/tuning.yaml` 兼容；`nhead` 与 `d_model` 整除约束代码已自动过滤）：
  | 旋钮 | 取值 | 备注 |
  | --- | --- | --- |
  | window_size | 64 / 128 / 256 | 6s×256≈25min 上下文 |
  | d_model | 32 / 64 / 128 | |
  | nhead | 2 / 4 / 8 | 受 d_model 整除约束 |
  | num_layers | 1 / 2 / 4 | |
  | dim_feedforward | 128 / 256 / 512 | 建议 ≥2×d_model，偏好 4× |
  | dropout | 0.0 / 0.1 / 0.2 | 过拟合时加 0.3 |
  | lr | 1e-4 / 3e-4 / 5e-4 / 1e-3 | AdamW |
  | weight_decay | 0 / 1e-5 / 1e-4 | |
  | batch_size | 64 / 128 | |
  epochs 上限 25、patience 5（早停按 val MAE 保留——见改造清单 #2 说明）。
- trials 预算：**默认 32 个**。核算：30k 训练窗 × ~0.5–2s/epoch（3060/4060 级）→ 早停均值 ~15 epoch ≈ 10–40s/trial → 32 trials ≈ 10–25 min（含开销）；阶段 0b 实测后按「单 trial >60s 则 trials 减半、<10s 可加到 48」缩放。
- 运行：改造后 `python scripts\tune.py --config configs\tuning.yaml --data-path <npz> --out reports\tuning_p1`；产物 `tuning_summary.csv` 须含 val 全套 + S 列（改造清单 #1）。
- 决策分支：
  - top 配置 S 明显优于 baseline（且过门槛）→ 进入阶段 2 细搜；
  - 全部 trial 不过门槛但 S 有梯度（S 最优仍优于 baseline）→ 阶段 2 在 top-3 邻域搜，重点排查数据/口径问题；
  - 全部不过门槛且与 baseline 持平 → 先做诊断（对照 TUNING_GUIDE §5），不盲目加大搜索。

### 5. 阶段 2：细搜（top-3 邻域 ×3 seeds）
- 取阶段 1 按 S 排序的 top-3 配置，各自邻域小步长再搜（每簇 3–4 个变体）：
  - lr：×2 / ÷2（如 5e-4 → 1e-3、2.5e-4）；
  - dropout：±0.05（限制 ≥0）；
  - window_size：上下邻档；
  - d_model / num_layers：仅在阶段 1 中表现好的方向动半档。
- 每个入选配置（含 top-3 原配置）用 **3 个种子**各跑一遍（seed 42 / 2024 / 7），报 S 的 mean±std——过滤"单次运气好"。
- 预算：~9–12 runs × 3 seeds ≈ 15–35 min（按阶段 0b 实测缩放）。
- 决策分支：邻域变体无一提振 → 接受 top-3 中 mean±std 最优者；若最优 S 的标准差 > 均值的一半，视为不稳，回阶段 1 加 trials 再搜（宁可多跑，不留抖动结论）。

### 6. 阶段 3：锁定 + Test 恰好一次
- 锁定配置写入 `reports/best_config.yaml` + `REPORT_TEST.md` 本专题「执行实录」；
- 用锁定配置重训 3 seeds，**只在此时**读 test 指标，报 mean±std vs baseline（同口径 test）：
  - test 全面不差于 baseline 且 F1/EE 达标 → 该配置进 REPORT.md（候选推荐版）；
  - test 明显差于 val（分布漂移特征）→ 按 README §8「Train/Val 好 + Test 差」排查（时间漂移 / 预处理不一致），结果如实记录，不掩盖。
- 全程 test 触碰次数：2 次（阶段 0b 基线 + 本次），写入执行实录留痕。

### 7. 代码改造清单（执行前按 #1→#4 顺序完成，每个都小步提交）
| # | 文件 | 改什么 | 为什么 / 验收 |
| --- | --- | --- | --- |
| 1 | `scripts/tune.py` | 排名改为按 val 综合分 S；`tuning_summary.csv` 增加 val 全套指标、S、best_epoch、runtime、seed 列；`val_objective` 占位串删除；产物与 README 对齐（输出 `best_config.yaml` 而非仅 `best_trial.json`） | 消除 test 泄漏（最高优先）；README 一致性 | 
| 2 | `src/trainer.py` + `src/experiment.py` | `fit()` 的早停/checkpoint 判据参数化（`selection_metric: mae \| composite`，composite 用 S）；trial 排名与 checkpoint 判据必须同一口径 | 业务口径下"最优 epoch"应由 S 定义；不改造则保持 mae 并注明口径（可接受，二选一须落盘） |
| 3 | `scripts/prepare_ukdale.py`（新建） | h5 → npz 制备 + `data_spec.json`（按阶段 0 规格） | 真实数据入口缺口；README 的 h5 描述与代码 npz 入口不一致，一并修 README |
| 4 | `configs/tuning.yaml` | 对齐本方案搜索空间（含 trials 数量与窗口覆盖）；可加 `objective` 字段供 #1/#2 读取 | 配置即文档；降低误用 |

### 8. 验收标准（本方案交付层面）
- [ ] 方案含数据制备 → 基线 → 搜索 → 锁定全链路，每阶段有命令、预算、决策分支
- [ ] KPI 口径明确：Test 冻结 2 次触碰；排序只用 val 综合分 S；硬门槛先行
- [ ] 代码缺口 4 项已列清单与改法；未改造前禁止拿现有 tune.py 结果下结论
- [ ] TUNING_GUIDE.md 已建（人话版），本专题结果将在执行后回填「执行实录」
- **是否进入 REPORT.md（稳定结论）**：否——本专题为方案设计，尚未执行；执行完成且指标稳定后，把「推荐配置 + KPI 口径」更新进 REPORT.md
- **遗留问题**：
  1. ukdale.h5 数据源用户侧是否已下载（README 列出官方与 NILMbench 两条路线，未确认）；
  2. baseline.yaml 的 epochs=30/patience=7 与 tuning 的 25/5 不同，阶段 1 用 25/5、锁定复跑建议用 30/7 校验稳健性；
  3. ON/OFF 阈值固定 500W 是否适合 kettle 全时段（阈值敏感性分析留到锁定后可选做，300–700W 扫一遍）；
  4. max_samples 截断（30k/6k/6k）下的结论外推性——若预算允许，锁定后可用全量样本复核一次。

### 方案执行更新（2026-09-08）：代码改造 #1–#4 已完成并验证
- **类型**：工程实现（配套改造，本任务角色=工程实现工程师）
- **完成内容**（逐项对应第 7 节清单，均已 commit）：
  1. `src/objective.py`（新）：综合分 S 与业务门槛纯函数（无 torch 依赖，trainer/tune 共用）；`src/trainer.py`：`fit()` 新增 `select_metric: mae|composite` + objective 参数（默认 mae，旧行为不变）；`src/experiment.py`：支持 `data.eval_test`（默认 true，向后兼容）
  2. `scripts/tune.py`：排名改为 **val 综合分**（先门槛过滤再排序，未过门槛 trial 留档在 summary 末尾并注明原因）；`tuning_summary.csv` 增加 seed / git_commit / runtime_sec / best_epoch / val 全套 / val_composite / val_score 列；产物改为 `best_config.yaml`（README 对齐）；`search.report_test: false` 时各 trial 不评估 test（Test 冻结）；新增 `--trials` 覆盖
  3. `scripts/prepare_ukdale.py`（新）：NILMTK 风格 h5 → npz + `data_spec.json` 口径留痕；`--list-meters` 探表号；缺口策略（aggregate ffill ≤30min / target 填 0 ≤5min，可配）；长缺口取最长连续段不拼接；负功率 clip 计数留痕；布局不符报错+键树提示。README §2/§4/§6/§7 同步修正
  4. `configs/tuning.yaml`：trials 32、`objective: composite`（含门槛）、eval_test: false、搜索空间与方案表一致
- **验证结果（本 sandbox，全部真实运行）**：
  - `pytest tests/`：**10 passed**（test_objective 6 + test_prepare_ukdale 3 + test_model 1）
  - `run_smoke.py`（默认路径向后兼容）：PASS，CPU 下 test MAE 59.8 / R² 0.909，与用户 GPU 历史产物（61.1/0.906）同量级
  - tune 烟雾（composite、eval_test=false、CPU 变体配置）：4 trials 全部跑通；门槛 1/4 通过 → 正确按 val 分排序；test 列留空；`best_config.yaml` 含 select_metric/objective/eval_test 且可 yaml 加载复现
  - 环境：sandbox 内 /tmp/tvenv 装好 torch 2.14.0+cu130（CPU 宿主）+ 全依赖（nvidia pip 包补齐过程见 STATUS.md 决策记录）；用户机器仍按 README 用 conda 环境
  - `reports/smoke/*` 为 git 跟踪历史产物，验证前备份、验证后恢复，**未改动**
- **遗留/待办（用户机器）**：
  1. 真实数据制备：`python scripts\prepare_ukdale.py --h5-path <ukdale.h5> --list-meters` → 确认表号 → 生成 npz + data_spec.json
  2. 阶段 0b baseline ×3 seeds（`--out reports\baseline_s{42,2024,7}`），用实测 runtime 校准搜索预算与业务门槛
  3. 粗搜 `python scripts\tune.py --config configs\tuning.yaml --data-path <npz> --out reports\tuning_p1`（32 trials）
  4. 细搜 top-3 邻域 ×3 seeds → 锁定 `reports\best_config.yaml` → `train.py` + `evaluate.py` 碰 Test 一次
  5. 结果按第 7 节 SOP 回填本专题「执行实录」
- **未决问题**：同上「遗留问题」4 条（数据未下载未确认 / 30-epoch 复核 / 阈值敏感性 / 全量样本外推）

### 执行实录 1（2026-09-08）：粗搜 32 trials（用户机器）→ Top-5 与细搜设计
- **类型**：实验专题执行实录（真实数据，用户机器 Windows + GPU 回传；本任务角色=实验/调参教练）
- **粗搜结果**（composite 口径；门槛通过 31/32；val MAE 单位 W；EE=energy_error；S=0.4·(MAE/2000)+0.4·(1−F1)+0.2·|EE|）：

| rank | trial | S | val_mae | val_f1 | val_ee | w | d_model | L | lr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 13 | 0.0372 | 4.4 | 0.915 | +0.012 | 128 | 64 | 2 | 3e-4 |
| 2 | 18 | 0.0394 | 3.8 | 0.931 | +0.055 | 128 | 64 | 2 | 3e-4 |
| 3 | 16 | 0.0553 | 4.4 | 0.906 | −0.083 | 128 | 128 | 2 | 5e-4 |
| 4 | 30 | 0.0566 | 7.7 | 0.868 | +0.011 | 256 | 64 | 1 | 3e-4 |
| 5 | 14 | 0.0577 | 6.7 | 0.868 | −0.018 | 256 | 128 | 4 | 5e-4 |

- **数据解读（判读要点）**：
  1. S 构成：MAE 项仅 0.001 级（0.4·4.4/2000≈0.0009），(1−F1) 项 0.03–0.07 主导，EE 项次之（±0.01–0.017 级）→ **细搜实际是在打 F1 的 2–4 个百分点**与 EE 的 ±0.05 级收敛。
  2. 综合分有效性实证：trial 18 的 MAE(3.8)/F1(0.931) 全面优于 trial 13(4.4/0.915)，但因 EE +0.055 vs +0.012 而 S 反而更差——纯 MAE 排名会错选 #2，业务口径下 #1 更优（总量高估更小）。
  3. 种子噪声量级：trial 13 vs 18（同为 w128 d64 L2 lr3e-4 家族）S 差 ≈0.0022 → **细搜改善 <0.003 不算数，需按 mean±std 判定**（3 seeds 起步）。
  4. 邻域中心明确：w128+d64+L2+lr3e-4；w256 家族（两例）F1 均 0.868，落后约 0.05 → 长窗在本数据上不占优，w160 上限足够；d128、L4 未带来增益（小模型 + 单电器任务容量已够）。
  5. 遗留待查（需用户 csv 回传）：锚（trial 13）的 dropout / dim_feedforward / nhead / batch_size / weight_decay / val_precision / val_recall / train_mae —— precision 与 recall 的拆解决定微调方向（见细搜清单备注）；train vs val 差判断过拟合程度。
- **细搜设计（v1，批次 1 模板；锚 = trial 13 全参数，取自 reports\tuning_p1\best_config.yaml）**：
  - 原则：一次只动一个因子；每个变体 ×3 seeds（建议 1000/2000/3000，与粗搜 seed=42+i 体系无重叠）；比较用 mean±std；批次间不叠加多个未验证改动。
  - 批次 1（7 个变体 ×3 ≈ 21 runs）：
    | 变体 | 改动（相对锚） | 验证问题 |
    | --- | --- | --- |
    | V0 | 无（锚自身 ×3） | 方差基线（本批最重要的对照） |
    | V1 | lr=2e-4 | lr 是否还能往下榨（3e-4 已优于 5e-4） |
    | V2 | lr=4e-4 | 反向确认 3e-4 不是谷底邻域内的偶然 |
    | V3 | window=96 | 短窗端是否更好（w128 中心是否平台） |
    | V4 | window=160 | 长窗端收敛性（w256 已排除） |
    | V5 | dropout=锚±0.1（上探） | 依 train/val 差定方向：差大→0.2；几乎无差→0.0 |
    | V6 | dim_feedforward=锚×2（到 512 封顶） | ff 容量是否吃紧（若锚已 512 则跳过） |
  - 批次 2（视批次 1 结果，≤4 变体 ×3）：只在「mean 优于 V0 且差值 > 种子噪声（本批估计 ≈0.002–0.003）」的方向加密（如 lr 2e-4→1.5e-4、window 96→112 或 160→176、d_model 96 且 nhead 整除约束）；若 precision 明显低于 recall（假阳多）→ 加一档 dropout 或 ON 阈值敏感性检查（300–700W 只做诊断不动阈值）；若 recall 明显低于 precision（漏报 ON）→ 优先 window 160 / lr 微降。
  - 中止/锁定规则：批次 2 后仍无变体显著优于锚 → 锁 V0；用 epochs=30 / patience=7 复跑 3 seeds 定最终配置（稳健性复核），然后 `train.py` + `evaluate.py` **碰 Test 恰好一次**，全程 Test 触碰计数=2（阶段 0b 基线 + 本次）。
- **预算核算规则**：单 trial 耗时 ×（21 + ≤12）≈ 粗搜 32 trials 的 1–1.5 倍；若用户实测单 trial 较长可先只跑 V0/V1/V3/V5（12 runs）快速定位方向。
- **遗留问题**：锚的 dropout/ff/bs/wd/nhead 与 precision/recall 拆解未回传（决定 V5/V6 精确取值）；是否需要把细搜做成「指定候选 × N seeds 自动汇总 mean±std」的脚本（当前 tune.py 是随机搜索，无候选枚举入口）。
- 是否进入 REPORT.md：否（细搜与锁定结果出来后再判）

### 执行实录 1 补充（2026-09-08）：完整 top-5 参数行回传 → 细搜批次 v1.1
- **完整 top-5 行**（用户机器回传，git_commit ef45c0e，composite 口径）：

| trial | seed | S | val_mae | rmse | r2 | val_f1 | P | R | EE | w | d | nhead | L | ff | do | bs | lr | wd | best_ep | run_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | 55 | 0.0372 | 4.36 | 58.3 | 0.867 | 0.9153 | 0.900 | 0.9310 | +0.0123 | 128 | 64 | 8 | 2 | 128 | 0.1 | 64 | 3e-4 | 1e-4 | 4 | 46 |
| 18 | 60 | 0.0394 | 3.80 | 54.8 | 0.882 | 0.9310 | 0.931 | 0.9310 | +0.0554 | 128 | 64 | 2 | 2 | 128 | 0.2 | 128 | 3e-4 | 1e-4 | 14 | 65 |
| 16 | 58 | 0.0553 | 4.36 | 68.1 | 0.818 | 0.9057 | 1.000 | 0.8276 | −0.0833 | 128 | 128 | 2 | 2 | 256 | 0.2 | 128 | 5e-4 | 1e-4 | 17 | 83 |
| 30 | 72 | 0.0566 | 7.68 | 81.3 | 0.757 | 0.8679 | 1.000 | 0.7667 | +0.0113 | 256 | 64 | 2 | 1 | 128 | 0.0 | 128 | 3e-4 | 1e-5 | 12 | 49 |
| 14 | 56 | 0.0577 | 6.67 | 83.1 | 0.746 | 0.8679 | 1.000 | 0.7667 | −0.0177 | 256 | 128 | 4 | 4 | 512 | 0.1 | 128 | 5e-4 | 0 | 8 | 171 |

- **判读（决定 v1.1 变体）**：
  1. 锚 best_epoch=4（~9 epoch 早停），收敛极快、随后轻微过拟合；trial 18（do0.2）best_epoch=14 且 MAE 3.80 / F1 / P=R 0.931 全优 → **dropout 0.1→0.2 是最强单因子证据**。
  2. 锚 P 0.900 < R 0.931 → 假阳偏多（预测"爱开"，与 EE +1.2% 一致）；trial 18 do0.2 后 P=R=0.931 → 正则方向正确。
  3. S 拆解算例（教学实证）：trial13 = 0.0009(MAE) + 0.0339(1−F1) + 0.0025(EE)；trial18 = 0.0008 + 0.0276 + 0.0111 → F1 赢 0.0063 被 EE 恶化（+1.2%→+5.5%）吃掉还倒输 0.0022。**细搜比价必须 S 三列一起看**。
  4. nhead=8 仅锚独苗（top 其余 2/4）→ 单因子验证；ff=128（=2×d64，非 4×标准）→ 试 256。
  5. lr：5e-4 两例皆输 → 只测下行 2e-4（4e-4 信息量低不测）；wd 无信号不测。
  6. 锚家族单 trial ≈46–65s → 预算宽裕，建议 **5 seeds/变体**。
- **细搜批次 v1.1**（一次只动一个因子；每配置 ×5 seeds = 1000/2000/3000/4000/5000，粗搜 seed 体系为 42+i 无重叠）：
  - V0 锚（trial 13 全参数：w128 d64 nhead8 L2 ff128 do0.1 bs64 lr3e-4 wd1e-4）——方差基线，最重要
  - V1 dropout 0.2（主嫌疑，trial 18 证据）
  - V2 dropout 0.0（反向对照，确认正则方向）
  - V3 lr 2e-4（只测下行）
  - V4 window 96（短窗端）
  - V5 window 160（长窗端，w256 不碰）
  - V6 nhead 4（独苗验证）
  - V7 dim_feedforward 256（容量检查）
  - 预算：8×5=40 runs × ~50–70s ≈ **35–50 min**；若想压时间可 3 seeds（24 runs ≈ 20–30 min），判定阈值相应放宽到 >0.005
- **判定规则**：V vs V0 比 mean±std：改善 > max(0.004, 2×SEM)（5 seeds 下约 0.003–0.004）才算有效；EE 恶化 >0.03 且 F1 改善 <0.01 视为无效交换；V0 std>0.005 则先加 seeds 再下结论。
- **批次 2 预案**：把批次 1 的赢家方向合成（如 do0.2+lr2e-4 单次组合验证可加性）；若 do0.2 提升 F1 但 EE 显著恶化 → 单独试 wd 3e-4 / bs64 不动；nhead 若 4 或 8 有差异则加密 6（整除约束）。
- **手动运行参考**（train.py 逐配置逐 seed；fine 配置继承 best_config.yaml 的 eval_test:false，不会碰 Test）：
  ```powershell
  foreach ($v in v0_anchor,v1_do02,v2_do00,v3_lr2e4,v4_w96,v5_w160,v6_nhead4,v7_ff256) {
    foreach ($s in 1000,2000,3000,4000,5000) {
      python scripts\train.py --config "configs\fine\$v.yaml" --data-path D:\datasets\ukdale_prepared.npz --seed $s --out "reports\fine\${v}_s$s" } }
  ```
  汇总：`python scripts\summarize_fine.py --runs-dir reports\fine`（输出按变体 mean±std 表）。
  - ⚠️ seed 陷阱：`best_config.yaml` 内 `seed: 55` 是固定值，**同一 yaml 直接跑 5 次结果会一模一样**——必须用 `train.py --seed`（2026-09-08 已加）逐次覆盖。
  - ⚠️ V1 注意：trial 18 是 nhead2/bs128/do0.2，**不等于** V1（nhead8/bs64/do0.2）。V1 故意保持锚的 nhead8/bs64 只动 dropout，隔离因子；若 V1 表现不如 trial 18，说明增益来自 bs128 或 nhead2，批次 2 再单测。
  - vX.yaml 生成规则：复制 best_config.yaml 后**只改一行**（表见下），其余字段一律不动：
    | 文件 | 键 | 新值 |
    | --- | --- | --- |
    | v0_anchor | 不改 | — |
    | v1_do02 | model.dropout | 0.2 |
    | v2_do00 | model.dropout | 0.0 |
    | v3_lr2e4 | training.lr | 0.0002 |
    | v4_w96 | data.window_size | 96 |
    | v5_w160 | data.window_size | 160 |
    | v6_nhead4 | model.nhead | 4 |
    | v7_ff256 | model.dim_feedforward | 256 |
  自查：`fc /n configs\fine\v0_anchor.yaml configs\fine\v1_do02.yaml` 应只差目标行。
- **锁定规则不变**：无变体显著优于 V0 → 锁锚；epochs=30/patience=7 ×3 seeds 复核 → `evaluate.py` 碰 Test 恰好一次（Test 触碰计数 2）。
- 是否进入 REPORT.md：否（待细搜/锁定）

### 执行实录 2（2026-09-08）：细搜批次 1 完成 → 判读与批次 2 设计
- **批次 1 结果**（用户机器回传，summarize_fine.py 输出，每变体 n=5，seed 1000–5000）：

| 变体 | S mean±std | MAE | RMSE | F1 | P | R | EE | best_ep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_do00 | **0.0463±0.0132** | 3.60±0.72 | 55.5 | **0.9126±0.0317** | 0.904 | 0.9241 | −0.0288 | 9.8 |
| v3_lr2e4 | 0.0517±0.0168 | 3.93±0.79 | 58.3 | 0.8866 | 0.865 | 0.9103 | +0.0168 | 9.0 |
| v6_nhead4 | 0.0542±0.0218 | 4.06±1.56 | 62.3 | 0.8784 | 0.850 | 0.9103 | +0.0071 | 9.0 |
| v4_w96 | 0.0552±0.0034 | 5.80±0.52 | 80.7 | 0.8844 | 0.920 | 0.8529 | +0.0085 | 10.6 |
| v7_ff256 | 0.0580±0.0350 | 3.83±1.64 | 60.3 | 0.8705 | 0.832 | 0.9172 | +0.0035 | 5.4 |
| v1_do02 | 0.0640±0.0424 | 5.51±3.80 | 65.6 | 0.8778 | 0.846 | 0.9172 | −0.0701 | 9.4 |
| v0_anchor | 0.0645±0.0479 | 4.82±3.31 | 63.3 | 0.8943 | 0.882 | 0.9103 | −0.1065 | 7.8 |
| v5_w160 | 0.0668±0.0108 | 4.88±0.70 | 79.0 | 0.8473 | 0.846 | 0.8500 | +0.0112 | 9.0 |

- **判读**：
  1. **锚（V0）种子噪声极大（0.0645±0.0479）**；粗搜单跑 0.0372 属幸运上端（优胜者偏差，32 选 1 必然偏乐观）→ 锚真实水平 ≈0.05–0.07；任何单跑数字不得作为配置水平表述。
  2. **dropout 方向反转**：v2(do0) 均值/分量全面最优（MAE 3.60、F1 0.913、R 0.924、EE −0.029 均胜锚；σ 0.013 vs 0.048 更稳）；v1(do0.2) 未复现粗搜 trial18 → trial18 增益源自 nhead2/bs128 或运气，非 dropout 本身。小模型+wd1e-4+val 早停下 dropout 非必需。
  3. **窗口族（w96/w160）稳定但天花板低**（recall≈0.85 漏报 ON）→ 不继续攻窗口。
  4. 严格显著性：v2 vs V0 差值 0.018 < 2×合并SEM≈0.044（V0 太吵）→ 未达显著；选 v2 为新基准依据"分量一致占优+更稳"，属判断非铁证 → 批次 2 验证。
  5. lr2e4（v3 0.0517 vs v0 0.0645）、nhead4（v6 0.0542）在 do0.1 基准上均呈方向性改善 → 值得在 do0 新基准上各单测一次。
- **批次 2 设计（基准切换为 v2_do00 = 锚但 dropout: 0.0）**，继续一次一因子：
  - C1 = v2 + lr 0.0002（lr 下行在无 dropout 基准上是否仍有增益）
  - C2 = v2 + nhead 4（独苗验证在新基准上是否成立）
  - v2 加跑 10 seeds 凑 n=15（压方差）
  - 每配置 ×10 fresh seeds（6000–6009）：v2 n=15、C1/C2 n=10，共 30 runs ≈ 30–40 min
  ```powershell
  # C1 = 复制 v2_do00.yaml 改 training.lr → 0.0002；C2 = 复制改 model.nhead → 4
  foreach ($cfg in "v2_do00","c1_lr2e4","c2_nhead4") {
    foreach ($s in 6000..6009) {
      python scripts\train.py --config "configs\fine\$cfg.yaml" --data-path D:\datasets\ukdale_prepared.npz --seed $s --out "reports\fine\${cfg}_s$s" } }
  python scripts\summarize_fine.py --runs-dir reports\fine
  ```
  - 判定：C 赢 = 均值改善 > 2×合并 SEM（≈0.010–0.012）且无「EE 恶化 >0.02 且 F1 改善 <0.01」的无效交换；否则锁 v2。
  - 收官：锁定配置用 epochs=30/patience=7 ×3 seeds 复核（do0 需防 30-epoch 过拟合，若 val 掉则试 do0.05 中档 ×5 seeds）→ `evaluate.py` 碰 Test 恰好一次（计数 2）。
- 是否进入 REPORT.md：否（批次 2/锁定未完成）

### 执行实录 2 补充（2026-09-08）：批次 2 部分结果（v2 n=15）与汇总脚本修复
- **批次 2 用户回传（v2 加跑 10 seeds 完成，n=15）**：
  - v2_do00 (n=15)：S **0.0434±0.0156**（SEM≈0.0040）| MAE 3.74±1.45 | F1 0.9115±0.034 | P 0.901 | R 0.9241 | **EE −0.0086±0.045**（近零偏差）| best_ep 9.6
  - 结论更新：n=15 下均值收敛于 ≈0.043（批次 1 的 n=5 估 0.0463 略偏乐观），EE 偏差几乎归零；v2 稳定性良好（σ0.0156）。
- **工具 bug（重要）**：`summarize_fine.py` 原只扫 `v*_s*` 目录，批次 2 的 `c1_/c2_` 变体被静默漏掉 → 已改为识别任意 `<变体>_s<种子>`（正则），并打印扫描到的 run 数；修复版自测通过（c1/c2/v2/非 run 目录混合识别正确）。
- **待回传**：c1_lr2e4 / c2_nhead4 的 10 seeds 结果（若尚未运行则补跑，命令见上一条；目录名保持 `c1_lr2e4_s6000` 等即可被新版脚本识别）。
- 判定规则不变：C 赢 = 均值改善 > 2×合并 SEM 且无「EE 恶化 >0.02 且 F1 改善 <0.01」无效交换；否则锁 v2。
- 是否进入 REPORT.md：否（待 c1/c2 与锁定）

### 执行实录 3（2026-09-08）：批次 2 完成 → 锁定候选 c2（nhead4），进入收官
- **批次 2 完整结果**（用户机器回传，c2 n=9 / c1 n=10 / v2 n=15）：

| 配置 | n | S mean±std | MAE | F1 | P | R | EE | best_ep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **c2_nhead4** | 9 | **0.0381±0.0102** | **3.56±1.06** | **0.9196±0.021** | **0.9201** | 0.9195 | **−0.0025** | 9.9 |
| c1_lr2e4 | 10 | 0.0420±0.0171 | 3.82±1.73 | 0.9109 | 0.8993 | 0.9241 | +0.0015 | 9.5 |
| v2_do00 | 15 | 0.0434±0.0156 | 3.74±1.45 | 0.9115 | 0.9008 | 0.9241 | −0.0086 | 9.6 |

- **判读**：
  1. **c2（nhead4）为批次 2 最优且分布最紧**：S 0.0381±0.0102（σ/SEM 全场最小）；分量全优或持平——MAE 3.56 / F1 0.9196 / P 0.9201 三项全场最优，EE −0.0025 近零；P/R 拉平（0.920/0.920），修正了锚时代"假阳偏多（P<R）"的老问题。
  2. **显著性（诚实口径）**：c2 vs v2 差值 0.0053 < 2×合并 SEM（≈0.0105），严格未达显著；但这是**第二次出现同构证据**（批次 1 v2 vs 锚：分量一致占优+更稳 → 换基准；批次 2 c2 vs v2 重演）→ 判定性选择 c2 为锁定候选，不宣称"统计显著"。
  3. **c1（lr2e4）与 v2 无差**（0.0420 vs 0.0434，噪声内）→ lr 维持 3e-4，不换。
  4. 收敛路线回顾：nhead8→4 是锚→v2→c2 三连跳中唯一"更稳且更准"的头型变化；w/ff/do/lr 方向均已探明无增益。搜索收敛，不再开新批次（边际收益 < 判定噪声）。
- **锁定候选**：c2 = `w128 d64 nhead4 L2 ff128 dropout0 bs64 lr3e-4 wd1e-4`（= v2_do00 + nhead 8→4）。
- **收官 SOP（请按序执行并回传）**：
  1. 生成 `configs\final_c2.yaml`：复制 `configs\fine\c2_nhead4.yaml`，改 `training.epochs` 25→**30**、`training.patience` 5→**7**（其余不动；eval_test 保持 false）。
  2. val 稳健性复核 ×3 seeds（7000–7002）：
     ```powershell
     foreach ($s in 7000,7001,7002) {
       python scripts\train.py --config configs\final_c2.yaml --data-path D:\datasets\ukdale_prepared.npz --seed $s --out "reports\final\final_c2_s$s" }
     python scripts\summarize_fine.py --runs-dir reports\final
     ```
     预期 val_score ≈0.038±0.01；若 >0.05 或与 25-epoch 复核差 >0.015 → 停下回报（30-epoch do0 有过拟合迹象时改试 do0.05）。
  3. **Test 恰好一次**（训练确定性保证：同 seed 7000 重训即同模型，只多算 test）：
     ```powershell
     python scripts\train.py --config configs\final_c2.yaml --data-path D:\datasets\ukdale_prepared.npz --seed 7000 --out reports\final\final_c2_test --test
     python scripts\evaluate.py --run-dir reports\final\final_c2_test
     ```
     `--test` 为本次新增显式开关（覆盖 eval_test），仅此一步使用。
  4. 回传：步骤 2 的汇总表 + 步骤 3 的 test 指标。验收口径（注：阶段 0b baseline 记录未回传，无法做 baseline 对比）：
     - test S 与 val 复核均值同量级（差异 <0.01 视为无分布漂移）；
     - 业务门槛在 test 上复验：F1/recall ≥0.75/0.70、|EE| ≤0.15（预期远超）；
     - 全程 Test 触碰计数 = 2（阶段 0b 基线 + 本次；若阶段 0b 从未执行则为 1）。
- **遗留问题**：baseline test 缺失（未回传）；锁定后可选做 ON 阈值敏感性（300–700W 诊断）、全量样本外推复核。
- 是否进入 REPORT.md：收官数据回传且验收通过后判（推荐配置 + KPI 口径拟进入）。

### 执行实录 3 补充（2026-09-08）：30-epoch 复核结果 → 决策：维持 25/5 口径锁定 c2
- **final_c2（epochs30/pat7，seed 7000–7002，n=3）回传**：S **0.0476±0.0090** | MAE 3.21±0.31 | F1 0.9095±0.023 | P 0.9011 | R 0.9195 | **EE −0.0540±0.0196** | best_ep 12.3
- **对照（锁定候选 c2，epochs25/pat5，seed 6000–6008，n=9）**：S 0.0381±0.0102 | MAE 3.56 | F1 0.9196 | P 0.9201 | R 0.9195 | EE −0.0025 | best_ep 9.9
- **判读**：
  1. 30/7 复核 S 劣化 +0.0095（0.0476 vs 0.0381），**劣化几乎全部来自 EE**（−0.054 vs −0.0025，差 0.0515 ≈ 3× 合并 SEM≈0.017 → 统计上显著转负，系统性低估总电量 ~5%）；F1 差 0.010 在噪声内（合并 SEM≈0.015）。
  2. best_ep 9.9→12.3：更宽的 patience(7) 让早停多等 7 轮，选到 val 平台期更晚的点——该点 EE 偏负。30/7 口径没有带来稳健性收益，反而引入系统性能量低估。
  3. **决策：复核未通过 → 不采纳 epochs30/pat7；锁定配置维持 25/5 口径的 c2**（w128 d64 nhead4 L2 ff128 do0 bs64 lr3e-4 wd1e-4，epochs25/pat5）——它与全部选型证据（粗搜→批次1→批次2）同口径，n=9 均值 0.0381、EE 近零。
- **修正后收官步骤**：
  1. fresh-seed 复核 25/5 口径 ×3（7000–7002，防"6000 系种子侥幸"，~8 min）：
     ```powershell
     foreach ($s in 7000,7001,7002) {
       python scripts\train.py --config configs\fine\c2_nhead4.yaml --data-path D:\datasets\ukdale_prepared.npz --seed $s --out "reports\final\fc2_25_s$s" }
     python scripts\summarize_fine.py --runs-dir reports\final
     ```
     预期 S ≈0.038–0.042；若 ≈0.047（与 30/7 同），说明 7000 系种子整体 EE 偏负 → 停下回报（改用 6009 seed 做 Test）。
  2. **Test 恰好一次**（25/5 口径，seed 7000）：
     ```powershell
     python scripts\train.py --config configs\fine\c2_nhead4.yaml --data-path D:\datasets\ukdale_prepared.npz --seed 7000 --out reports\final\c2_test --test
     python scripts\evaluate.py --run-dir reports\final\c2_test
     ```
  3. 回传：步骤 1 汇总表 + 步骤 2 test 指标。验收：test S 与 fc2_25 均值同量级（差 <0.01 无分布漂移）；test 业务门槛复验（F1/recall ≥0.75/0.70、|EE| ≤0.15）。全程 Test 触碰计数 = 1（阶段 0b 基线未执行）。
- 是否进入 REPORT.md：Test 回传验收通过后判。

### 执行实录 3 补充 2（2026-09-08）：fresh-seed 复核揭示「种子批次效应」→ 补 v2×7000 配对对照
- **fc2_25（c2 25/5，seeds 7000–7002，n=3）回传**：S **0.0560±0.0125** | MAE 3.48±0.45 | F1 0.8992 | P 0.8805 | R 0.9195 | **EE −0.0748±0.0195** | best_ep 7.3
- **关键对照（同 seeds 7000–7002，仅 epochs/patience 不同）**：final_c2(30/7) S 0.0476 vs fc2_25(25/5) S 0.0560；EE −0.054 vs −0.075 → **同种子下 30/7 反而更优** → 上一条「30/7 引入 EE 偏差、复核未通过」的结论系**跨种子批比较（6000 vs 7000）的混杂**，正式撤回。epochs/patience 25/5 vs 30/7 在 matched seeds 上无稳健差异（S 差 0.008，n=3 噪声内）。
- **判读（种子批次效应为主因）**：
  1. 7000 系 seeds 对 c2 系统性产出 S≈0.048–0.056、EE≈−5.4%~−7.5%（两档 epochs 一致），6000 系 c2 为 0.0381 / EE −0.0025 → n=3 均值偏移 ≈3×SEM，属该 seed 族收敛到「低幅估计盆地」的系统性坏运气，非配置问题。
  2. **c2 vs v2 的"胜出"部分源于 6000 系种子**：c2 全种子池化（n=12，25/5）S≈0.0426、EE≈−0.021；v2（n=15）S 0.0434、EE −0.0086 → **两者统计打平**（差 ≈0.001）；c2 的 0.0381 是 6000 系幸运端。
  3. **证据缺口**：v2 没有 7000 系 runs → 无法区分「7000 系种子对所有 do0 配置都不利」还是「nhead4 特定不利」。
- **决定：补 v2_do00 × seeds 7000–7002（3 runs ≈ 4–6 min）完成配对**：
  ```powershell
  foreach ($s in 7000,7001,7002) {
    python scripts\train.py --config configs\fine\v2_do00.yaml --data-path D:\datasets\ukdale_prepared.npz --seed $s --out "reports\final\fv2_25_s$s" }
  python scripts\summarize_fine.py --runs-dir reports\final
  ```
- **决策树**：
  - fv2_25 ≈ 0.038–0.042 且 EE≈0 → 7000 系对 nhead8 无碍 → 锁 **v2(nhead8)**（跨种子族稳健；c2 的 7000 系特异变差归因 nhead4）
  - fv2_25 ≈ 0.048–0.056 且 EE≈−0.05~−0.075（与 c2 同）→ 种子族效应全局 → v2 与 c2 打平，仍锁 **v2**（池化 n=18 估计更稳、EE 池化更近零），并如实记录「do0 家族的 EE 存在种子族敏感性，预期性能按池化值 ≈0.045±0.01 而非 0.038」
  - （两种分支最终都指向 v2，但依据与预期口径不同——分支 2 必须用池化悲观口径做 Test 预期）
- Test 预注册：锁定后以 seed 7000 跑恰好一次（Test 触碰计数仍 = 1）。
- 是否进入 REPORT.md：fv2 对照与 Test 回传后判。

### 执行实录 3 补充 3（2026-09-08）：fv2 配对对照完成 → 决策树分支 1 → 最终锁定 v2（nhead8）
- **fv2_25（v2_do00 × seeds 7000–7002，n=3）回传**：S **0.0407±0.0125** | MAE 3.05±0.44 | F1 0.9209±0.024 | P 0.9118 | R 0.9310 | **EE +0.0050±0.053** | best_ep 8.7
- **同 seeds 7000–7002 三方对照（nhead 为唯一系统差异）**：

| 配置 | nhead | epochs/pat | S | EE |
| --- | --- | --- | --- | --- |
| **fv2_25** | **8** | 25/5 | **0.0407±0.013** | **+0.005** |
| final_c2 | 4 | 30/7 | 0.0476±0.009 | −0.054 |
| fc2_25 | 4 | 25/5 | 0.0560±0.013 | −0.075 |

- **判读（分支 1 命中）**：7000 系种子对 v2(nhead8) 无碍（0.0407 落在预期 0.038–0.042 带内、EE 近零），而 nhead4(c2) 同种子系统性变差 → **7000 系变差是 nhead4 特异性**（该 seed 族上 nhead4 收敛进低幅低估盆地），非种子族全局效应；30/7 vs 25/5 无稳健差异（前条已撤回）。
- **最终锁定：v2_do00 = w128 d64 nhead8 L2 ff128 dropout0 bs64 lr3e-4 wd1e-4（epochs25/pat5）**。池化证据（全种子族）：
  - v2：n=18（seeds 1000–5000 ×5、6000–6009 ×10、7000–7002 ×3）→ S ≈0.043±0.015、EE ≈−0.006（6000 系 0.0434 / 7000 系 0.0407，跨族一致）
  - c2：n=12 → S ≈0.0426，但 7000 系不稳定（EE −0.075）→ 弃
  - 两配置统计打平，v2 凭「跨种子族一致 + EE 池化更近零 + 与批次 1 锚同族(nhead8)」胜出。
- **Test 预注册执行（恰好一次，25/5 口径，seed 7000）**：
  ```powershell
  python scripts\train.py --config configs\fine\v2_do00.yaml --data-path D:\datasets\ukdale_prepared.npz --seed 7000 --out reports\final\v2_test --test
  python scripts\evaluate.py --run-dir reports\final\v2_test
  ```
  - Test 预期口径（用 7000 系 + 池化，不用乐观 0.038）：**S ≈ 0.04–0.055**（test 不参与早停、略高于 val 属正常）；验收：S 差 <0.015 无分布漂移；业务门槛 F1/recall ≥0.75/0.70、|EE| ≤0.15（预期远超）。
  - 全程 Test 触碰计数 = 1。
- 是否进入 REPORT.md：Test 回传验收通过后判（推荐稳定配置拟 = v2，KPI 口径拟固化）。

### 执行实录 4（2026-09-08）：Test 回传 → 判定：v2 未通过验收（时间分布漂移），决策岔口
- **Test（v2_do00 25/5，seed 7000，eval_test 恰好一次，best_epoch 8，n_test=6000，cuda 69.7s）回传**：
  - MAE 9.84 | RMSE 131.7 | R² 0.666 | SAE 0.234 | EE **−0.234** | P 0.915 | R 0.729 | F1 0.811
  - S_test = 0.4·(9.84/2000)+0.4·(1−0.811)+0.2·0.234 = 0.002+0.076+0.047 = **0.124**
- **对照**：val（同 seed 7000 系，fv2_25）S 0.0407、MAE 3.05、RMSE 52.8、F1 0.921、R 0.931、EE +0.005。
- **判定**：
  1. ΔS ≈0.084 ≫ 0.015 漂移阈值 → **未通过验收**；业务门槛 test 复验：F1 0.811 ✅ / recall 0.729 ✅（擦线）/ **|EE| 0.234 ❌（低估 23.4%）**。
  2. 模式 = 「Train/Val 好 + Test 差」已知失败模式（README §8：时间分布变化）：test 段漏报 ON 事件（recall −0.20，precision 不变 → 漏报非误报），漏掉能量直接造成 EE −23%；非过拟合、非随机噪声。
  3. **校准不可行**：val EE≈0（无系统偏差）→ −23% 仅存在于 test → 非恒定乘性偏差，val 拟合的缩放无法修正 test；唯一出路是让训练分布覆盖尾段形态（改切分/扩数据）。
  4. **不做任何基于 test 的选型**；Test 触碰计数 = 1（阶段 0b 基线未执行）。
- **待决策 3 方向**（用户）：
  - A：数据诊断先行 → 用 `scripts/diagnose_split.py`（纯数据、不跑模型）量化 train/val/test 三段的壶事件形态/能耗差异，判定「test 段异常」还是「真实漂移」→ 再决定改切分或接受
  - B：接受记录为实验结论（时间漂移限制时间切分下的泛化），v2 暂列「候选非推荐」；失败模式与教训进 REPORT.md/TUNING_GUIDE
  - C：调整切分（如 val 覆盖尾部、或随机切分）+ 预注册新验收协议 → 重跑锁定评估（GPU 时间 ~15 min，Test 协议重置一次并记录）
- 是否进入 REPORT.md：暂缓（漂移问题有结论前，v2 不列为推荐稳定版）。

### 执行实录 5（2026-09-08）：分段诊断（方向 A）→ 双重发现：test 段真实漂移 + aggregate 数据红旗
- **诊断输出**（用户机器，on_threshold 500W，n=10,344,744 = 718.4 天；kWh 单位已修正为真值）：

| segment | 天 | evt/day | on_frac | 壶 ON 均/中位 W | kWh/天 | agg 均值 W | agg p95 W | agg 关断期均值 W | corr(agg,target) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 502.9 | 4.76 | 0.0058 | 2296/2344 | **0.344** | 14.3 | 1.0 | 待重跑 | 待重跑 |
| val | 107.8 | 4.46 | 0.0060 | 2318/2343 | **0.359** | 15.0 | 1.0 | 待重跑 | 待重跑 |
| test | 107.8 | **5.31** | **0.0088** | 2319/2332 | **0.515** | 21.4 | 1.0 | 待重跑 | 待重跑 |

  （注：用户首版输出无 agg_off_mean/corr 列——该两列为脚本升级后新增，需用户重跑补齐；旧版 kWh 列曾 ×1000 偏差，此处已换算为真值：train 343.5→0.344、val 359.0→0.359、test 514.8→0.515 kWh/天。）
- **发现 1（test 段真实漂移，Test 失败的根因成立）**：test 窗（末段 108 天）壶用量强度显著高于 train/val——evt/day 5.31 vs 4.46–4.76（+12~19%）、on_frac 0.0088 vs 0.0058–0.0060（**+47~52%**）、kWh/天 0.515 vs 0.344–0.359（**+43~50%**）、事件时长推算更长（on_frac/evt 比 val 高约 25%）；ON 功率本身不变（2.3kW 级，2332–2344W）。→ 训练/验证期「壶用得少而短」，test 期「用得多而长」：模型按训练分布外推 → 多出的/更长的事件漏报（recall 0.93→0.73，precision 不变），漏掉能量 → EE −23%。**诊断成立：真实时间漂移，非 test 段异常、非过拟合。**
- **发现 2（数据红旗，NILM 前提存疑）**：agg 均值 14.3–21.4W ≈ 同期壶均值（target 平均功率同值），**agg p95 仅 1.0W**（真实家庭总负荷 p95 应为数百 W）→ aggregate 序列 95% 时间≈0、仅在壶事件时抬升，**强烈提示 npz 的 aggregate 几乎只含 kettle 通道，不是全屋总负荷**（输入≈输出，任务退化为"带噪自回归"，val MAE 3–4W 的"好成绩"是泄漏产物，不代表 NILM 泛化）。待用户重跑升级版脚本确认（看 agg_off_mean 与 corr 两列）。
- **推断（待验证）**：制备时 `--mains-ids 1,2` 命中的 meter 可能不是 UK-DALE House1 的 mains 相表，或 npz 由旧路线生成（aggregate 取自错误通道）；需核对 `ukdale_prepared.data_spec.json` 的 mains_meter_ids_used。
- **影响**：若红旗坐实，先前全部 KPI 仅对"壶通道重建"有效，对 NILM（总负荷→壶）不构成证据；调参方法论/脚本仍有效，但需**先修数据制备**（重跑 prepare_ukdale.py，--list-meters 确认 mains 表号，人工抽查 aggregate 一天曲线）再重新走验收。
- **待用户执行**：① `git pull` 后重跑 `diagnose_split.py`（新增 aggOffW/corr 列）回传；② 打开 `ukdale_prepared.data_spec.json` 回传 mains_meter_ids_used / 时间范围 / gap 记录；③ 可选抽查 aggregate 一天曲线确认基线负荷。
- 是否进入 REPORT.md：否（数据红旗查清前全部 KPI 挂起）。

### 执行实录 6（2026-09-08）：h5 布局探查 → NILMTK 格式确认，数据层问题定位
- **探查结果（新版 inspect_h5，building1）**：
  - 结构：NILMTK 转换格式（building1–5，每户 elec/meter1–54）；meter 组下是 pandas HDFStore 表（`_i_table/table`，`pd.read_hdf` 可读），不是直接数据集 → prepare_ukdale.py 的契约 A 分支不适配，需加 nilm 分支。
  - 列语义：meter1/2/3 列 = `('power','apparent')`（**视在功率**，n≈1000 万）；meter4+ 列 = `('power','active')`（**有功功率**，n≈250–930 万）。NILM 分解必须用 active；apparent 含无功分量，**不可**当 aggregate/壶功率。
  - 时间戳：表 index 为 int64（head 值 599/582/600W 像 6s 采样），单位待从 metadata 确认。
  - 身份：meter 组 attrs 含 pickle metadata（building1 顶层 attrs 可见 `appliances` 结构），**kettle/mains 表号需解 pickle metadata 确定，不能靠表号猜**。
- **推断（数据红旗根因）**：旧 npz 的 aggregate 几乎只含 kettle（agg p95≈1W），强烈怀疑是旧抽取流程把「壶通道」或某 apparent 通道误当作 aggregate；apparent vs active 混用也可能是功率语义错乱的来源之一。待 metadata 解析 + 重跑 list-meters 确认后重做 npz。
- **待办**：① 写 metadata 解析（pickle attrs → 表号→电器映射）；② prepare_ukdale.py 加 NILMTK(pandas table) 读取分支 + apparent/active 选择；③ 重生成 npz（mains active 正确合并 + kettle active）→ diagnose_split 复验 aggOffW 数百 W 且 corr 不≈1；④ 数据确认后 KPI 协议重置一次并记录。
- 是否进入 REPORT.md：否（数据修复前）。
- 是否进入 REPORT.md：否（方案与改造本身不是实验结论；待真实 KPI 出现后另行判定）

### 执行实录 7（2026-09-09）：--list-meters 54/54 通过，mains 功率语义确认
- **事实（用户回传全文）**：tz 修复（commit 1e7538c）后重跑，54 张表全部读出，无「读取失败」。关键行：
  - meter1/2/3：apparent，n≈10.07–10.24M，起止 2012-11-09→2015-01-05（全程，6s 口径约 90% 覆盖）
  - meter8/25：apparent，n≈10.22M/9.19M（身份待 metadata 定）
  - meter10：active，n=8.94M，2012-11-09→2015-01-05（kettle 候选，待 metadata 确认）
  - meter54：active，**n=56,687,460**，2013-03-17→2015-01-05（≈1/s 采样，疑似 mains 1 秒数据）
  - 无 meter0 组（表号 1–54，与 metadata 转述中出现的"meter 0"矛盾，见下）
- **重要修正（vs 实录 6 待办③）**：6s 口径下 meter1/2/3 **只有 apparent 列、没有 active** → 「mains active 合并」不可行。aggregate 二选一：(a) 6s apparent 总表直接用（文献常见做法，视在≥有功，能量口径需留痕）；(b) meter54 1s-active 降采样到 6s（备选，需先确认身份+加代码）。默认先走 (a)。
- **未决（卡点）**：表号→电器 ground truth 缺失。上一版 parse_nilmtk_metadata 输出只有转述（meter10→kettle、meter2→boiler、meter5→washer dryer、meter6→dish washer、无显式 mains），且转述含与 h5 结构矛盾的"meter 0"编号、meter2 身份（mains vs boiler）直接决定 --mains-ids。**不能靠猜定制备参数** → 请用户重跑 parse 脚本并贴**全文**，再定 prepare 命令。
- 是否进入 REPORT.md：否（数据修复中）。

### 执行实录 8（2026-09-09）：metadata 全文回传 → 定表号：mains=meter1（单表），kettle=meter10
- **事实（用户回传 parse 全文，共 53 条）**：meter10→kettle/food processor/toasted sandwich maker；meter2→boiler；meter3→solar thermal pumping station；meter8→light×2；meter25→light(16)；meter5→washer dryer；meter6→dish washer；meter0→immersion heater/water pump/security alarm/fan/drill/laptop（**h5 中无 meter0 组**）；**无 mains/无 meter1/无 meter54 条目**。
- **判定（编号对齐）**：metadata 编号 == h5 表号，无 off-by-one。证据：旧 npz target 就是教科书级壶脉冲（4–5 次/天、ON 2.3kW 级）且旧管线 kettle 表号为 10，与"metadata 10→kettle"双吻合。
- **判定（mains 身份）**：meter1 = mains（site meter，不在 appliance metadata 中属正常 NILMTK 行为；apparent、全程 10.24M、晚间 head≈600W 吻合）。meter2/3/8/25 系硬接线回路 CT 表（只测 apparent：锅炉/太阳能泵/灯回路），**不是 mains**——纠正了之前"--mains-ids 1,2"的假设（2 是锅炉回路，加进去会 double count）。metadata 0 无 h5 组 = 这些电器无独立子表数据，只存在于 mains 残差中。meter54（1s active，56.7M）同样不在 metadata 中 → 1 秒 mains，列为备选 aggregate（与 kettle 交叠约 660 天，充足）。
- **决策**：prepare 用 `--mains-ids 1 --kettle-meter-id 10`（**单总表**）；输出新文件 `ukdale_prepared_v2.npz`，**不覆盖**旧 npz（旧文件关联历史 KPI/Test 记录，保留备查）。
- **验证计划（证伪口）**：diagnose 新 npz，期望 aggOffW 数百 W、corr≪1、agg p95 数百~数千 W；若 aggOffW≈0 则 meter1=mains 假设被证伪，回滚重议，不硬上训练。
- 是否进入 REPORT.md：否（数据修复中）。
