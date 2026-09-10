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
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**：
  ```powershell
  python scripts\tune.py --config configs\tuning.yaml --data-path D:\datasets\ukdale_prepared.npz --out reports\tuning_p1
  （32 trials 默认；tuning_summary.csv 与 best_config.yaml 由该命令产出）
  ```

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
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（即实录 1 补充「手动运行参考」块，8 变体 ×5 seeds）：
  ```powershell
  foreach ($v in v0_anchor,v1_do02,v2_do00,v3_lr2e4,v4_w96,v5_w160,v6_nhead4,v7_ff256) {
    foreach ($s in 1000,2000,3000,4000,5000) {
      python scripts\train.py --config "configs\fine\$v.yaml" --data-path D:\datasets\ukdale_prepared.npz --seed $s --out "reports\fine\${v}_s$s" } }
  python scripts\summarize_fine.py --runs-dir reports\fine
  ```

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
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（即实录 2 批次 2 设计块：v2 补跑凑 n=15 + c1/c2 ×10）：
  ```powershell
  foreach ($cfg in "v2_do00","c1_lr2e4","c2_nhead4") {
    foreach ($s in 6000..6009) {
      python scripts\train.py --config "configs\fine\$cfg.yaml" --data-path D:\datasets\ukdale_prepared.npz --seed $s --out "reports\fine\${cfg}_s$s" } }
  python scripts\summarize_fine.py --runs-dir reports\fine
  ```

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
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（实录 3 补充 3 预注册命令的执行）：
  ```powershell
  python scripts\train.py --config configs\fine\v2_do00.yaml --data-path D:\datasets\ukdale_prepared.npz --seed 7000 --out reports\final\v2_test --test
  python scripts\evaluate.py --run-dir reports\final\v2_test
  ```

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
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（首版脚本；aggOffW/corr 两列为随后升级新增，升级后同命令重跑）：
  ```powershell
  python scripts\diagnose_split.py --npz D:\datasets\ukdale_prepared.npz
  ```

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
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**：
  ```powershell
  python scripts\inspect_h5.py --path D:\Work\testPython\datasets\ukdale.h5
  ```

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
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（tz 修复 commit 1e7538c 后重跑）：
  ```powershell
  python scripts\prepare_ukdale.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --list-meters
  ```

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
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**：
  ```powershell
  python scripts\parse_nilmtk_metadata.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --house 1
  ```

- **事实（用户回传 parse 全文，共 53 条）**：meter10→kettle/food processor/toasted sandwich maker；meter2→boiler；meter3→solar thermal pumping station；meter8→light×2；meter25→light(16)；meter5→washer dryer；meter6→dish washer；meter0→immersion heater/water pump/security alarm/fan/drill/laptop（**h5 中无 meter0 组**）；**无 mains/无 meter1/无 meter54 条目**。
- **判定（编号对齐）**：metadata 编号 == h5 表号，无 off-by-one。证据：旧 npz target 就是教科书级壶脉冲（4–5 次/天、ON 2.3kW 级）且旧管线 kettle 表号为 10，与"metadata 10→kettle"双吻合。
- **判定（mains 身份）**：meter1 = mains（site meter，不在 appliance metadata 中属正常 NILMTK 行为；apparent、全程 10.24M、晚间 head≈600W 吻合）。meter2/3/8/25 系硬接线回路 CT 表（只测 apparent：锅炉/太阳能泵/灯回路），**不是 mains**——纠正了之前"--mains-ids 1,2"的假设（2 是锅炉回路，加进去会 double count）。metadata 0 无 h5 组 = 这些电器无独立子表数据，只存在于 mains 残差中。meter54（1s active，56.7M）同样不在 metadata 中 → 1 秒 mains，列为备选 aggregate（与 kettle 交叠约 660 天，充足）。
- **决策**：prepare 用 `--mains-ids 1 --kettle-meter-id 10`（**单总表**）；输出新文件 `ukdale_prepared_v2.npz`，**不覆盖**旧 npz（旧文件关联历史 KPI/Test 记录，保留备查）。
- **验证计划（证伪口）**：diagnose 新 npz，期望 aggOffW 数百 W、corr≪1、agg p95 数百~数千 W；若 aggOffW≈0 则 meter1=mains 假设被证伪，回滚重议，不硬上训练。
- 是否进入 REPORT.md：否（数据修复中）。

### 执行实录 9（2026-09-09）：prepare 首跑 n=345 → 秒级相位差确诊，改统一 6s 网格对齐
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（v2 首次制备 + 诊断）：
  ```powershell
  python scripts\prepare_ukdale.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --mains-ids 1 --kettle-meter-id 10 --out D:\Work\testPython\datasets\ukdale_prepared_v2.npz
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_prepared_v2.npz
  ```

- **事实（用户回传三份）**：prepare 输出 n=345（仅开头半小时 22:28:15→22:58:26），kettle NaN 0→策略后 8,862,517；diagnose n=345（0.0 天，无事件）；data_spec：mains_used=[1]/apparent、kettle=10/active、median_gap 6.0。
- **根因（事实+推断）**：kettle 序列本身无 NaN（before=0），886 万 NaN 全部来自外连接对齐 → 两表时间戳大面积错位。直接证据：list-meters 起始秒 meter1=:15 vs meter10=:18。推断：各表采样相位差秒级，精确 join 只拼上开头时钟偶然对齐的半小时。meter1=mains 假设未被证伪：345 行的 aggW≈350–525W（真实总负荷基线，非 0）。
- **修复（commit 本回合）**：prepare 对齐前各表先 `_to_6s_grid()` resample（bin 内均值，epoch 原点；已在网格数据为恒等变换）；`_combine_mains` 与 kettle 同处理；data_spec schema_version 1→2 + `resample_policy` 留痕；`--mains-ids` 默认 1,2→1（House1 地面真相）+ docstring/README 示例同步。新增偏移 3s 回归测试（旧逻辑下 shape 会膨胀错位，新逻辑 n 不丢），pytest（除 test_model）13 passed。
- **待用户**：git pull 后重跑 prepare（同命令，覆盖 v2 文件）+ diagnose，回传。预期 n≈800–900 万、跨度 2012→2015。
- 是否进入 REPORT.md：否（数据修复中）。

### 执行实录 10（2026-09-09）：prepare 二跑 n=2058 → 三因确诊（缺口地形+选段索引 bug+sum 假零），改全量拼接留痕（schema v3）
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（git pull 取 v3 修复后同命令重跑）：
  ```powershell
  python scripts\prepare_ukdale.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --mains-ids 1 --kettle-meter-id 10 --out D:\Work\testPython\datasets\ukdale_prepared_v2.npz
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_prepared_v2.npz
  ```

- **事实（用户回传两份）**：resample 生效——对齐行 mains 11,323,076 / kettle 11,323,078 网格点；但 n=2058（仅头部 3.4h，22:28:12→01:54:00），kettle NaN 策略前 2,403,512 → 策略后 2,403,463；diagnose：3.4h 夜间数据，无壶事件，aggW 基线 134–1298W 正常、corr≈0（无事件时属预期）。
- **根因①（数据地形，事实）**：两表内部缺口密布——meter10 缺 240 万格（≈167 天当量，占跨度 21%），meter1 缺≈48 万格（10.84M 样本 vs 11.32M 格）。双表同时无缺口的最长段仅 ~3.4h 量级 →「只取最长连续段」策略在该数据上不可行。
- **根因②（选段索引 bug，本侧责任）**：`df[mask]` 过滤后仍用过滤前的位置编号算段长 → 末段长度被低估「剔除行数−1」（本例 ≈240 万），选段结果不可信；n=2058 实为头部段。修复：段统计统一在过滤前索引空间计算（diff on bool mask）。
- **根因③（sum 假零，潜伏雷，本侧责任）**：`_combine_mains` 的 `DataFrame.sum(axis=1)` 默认 skipna → 单表全 NaN 缺口格被静默写成 **0W 假零**（非 NaN），ffill 完全失效，≈48 万格假数据将混入 aggregate 毒害训练。由新增缺口回归测试的桥接断言拦截发现（沙箱复现：agg len=3000 nan=0、缺口处全 0.0）。修复：`sum(axis=1, min_count=1)`（全缺保持 NaN，交统一缺口策略）。
- **修复清单（本 commit）**：min_count=1；段统计索引空间统一；策略改「剔除缺口行后全量按序拼接，接缝留痕」（schema_version 3：新增 n_segments / n_concat_breaks / largest_segment_samples / union_grid_samples / dropped_gap_samples）；resample 显式 `origin="epoch"`（跨 pandas 版本网格恒等，与留痕串一致）；采样间隔抽查改段内口径（Timedelta 单位安全——顺带踩坑记录：pandas 3 的 `asi8` 随 index 单位返回 us，ns 假设翻车）。
- **拼接代价（已接受）**：跨缺口的烧水事件会被剪成残缺事件（缺口切断概率≈10%/缺口，事件 4–5 次/天）；接缝处 aggregate 电平跳变。npz 本不带时间戳（v1 同为拼接流语义），代价以留痕换诚实。
- **判读**：meter1=mains 依然成立（aggW 基线正常）。前两版 v2 npz（n=345 / n=2058）作废。
- **待用户**：git pull 后同命令重跑 prepare + diagnose。预期：n≈850–890 万、拼接数千处、最大连续段小时~天级、diagnose days≈590–620。
- 是否进入 REPORT.md：否（数据修复中）。

### 执行实录 11（2026-09-09）：prepare 三跑 n=8,849,796 → 红旗解除 + fillna 全轴限额 bug 确诊（v4 整段桥接）
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（git pull 取 v4 修复后同命令重跑）：
  ```powershell
  python scripts\prepare_ukdale.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --mains-ids 1 --kettle-meter-id 10 --out D:\Work\testPython\datasets\ukdale_prepared_v2.npz
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_prepared_v2.npz
  ```

- **事实（用户回传两份）**：n=8,849,796（614.6 天，2012-11-09→2015-01-05）；diagnose：train/val/test=430.2/92.19/92.19 天，壶事件 5.56/5.24/6.18 次/天，壶功率 meanW≈2300W，aggOffW=354.8/327.0/400.3，corr=0.42/0.49/0.53。
- **判定：aggregate 红旗正式解除**（实录 5 起挂起）：①aggOffW 数百 W=真实家庭基线；②corr≈0.5=aggregate 含壶+其他负荷；③evt/day 5-6 + 壶功率 2300W=meter10 身份复验通过。meter1=mains、meter10=kettle、NILM 前提成立，数据身份链闭环。
- **但留痕暴露新 bug（本侧责任）**：「kettle NaN 2,403,512 → 策略后 2,403,463」——240 万缺口格只被填了 **49 格**；「跨缺口拼接 1,457,219 处、最大连续段 2057」——平均段长 ~6 格（36 秒）。
- **根因（沙箱 pandas 3.0.5 验证）**：`fillna(value, limit=N)` 的 limit 是**全轴总限额**（pandas 文档：method 未指定时按整轴计）而非"每段最多 N 格"——壶表 ~120 万个微掉线（1-2 格/次）全部未桥接，数据碎成 146 万段；另 `ffill(limit=N)` 为每段头部 N 格语义，与"短缺口 ffill"语义亦不符。验证：三处 2 格缺口只填了全序列头 2 格。
- **修复（v4，schema_version 4）**：新增 `_bridge_short_gaps`（run-length 整段桥接）：≤阈值缺口**整段**补值（agg 用 ffill 前值 / kettle 补 0），更长缺口**整段**剔除；gap_policy 留痕 policy 字符串+桥接格数；测试改双缺口用例（短缺口桥接保留+长缺口整段剔除+接缝统计），pytest 14 passed。
- **事故记录（本回合开头）**：沙箱被平台重克隆（reflog 仅剩 clone+checkout，本地链一度消失、/tmp 清空）；远端分支完好（3206715）→ fetch+逐文件哈希对账（全部一致）+reset 恢复，零数据丢失。教训：远端分支是唯一可靠真值，回合初必须 `git log`+`git ls-remote` 对账。
- **判读补充**：test 段壶用量 kWh/day 0.541 vs train 0.353（+53%）、on_frac 0.006→0.0093——时间漂移证据仍在（模型评估议题，非数据问题），数据锁定后回 TUNING_GUIDE 战史处理。
- **待用户**：git pull 重跑 prepare + diagnose。预期：n≈950-1080 万（v3 误删的 ~160 万桥接格回归）、接缝大幅下降（若仍 >10 万处可 --kettle-gap-min 30 提高壶桥阈值）、最大连续段有望天级；aggOffW/evt/day 身份指标不变。
- 是否进入 REPORT.md：否（数据修复收尾中；红旗解除结论待 v4 复验后并入）。

### 执行实录 12（2026-09-09）：prepare 四跑 n=10,377,651 → 结构健康/红旗三验稳定，但 evt/day 翻倍确诊 v4 补0切碎煮沸（v5 ffill）
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（git pull 取 v5 修复后同命令重跑）：
  ```powershell
  python scripts\prepare_ukdale.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --mains-ids 1 --kettle-meter-id 10 --out D:\Work\testPython\datasets\ukdale_prepared_v2.npz
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_prepared_v2.npz
  ```

- **事实（用户回传两份）**：n=10,377,651（720.7 天）；桥接 agg 467,337 格 / kettle 1,532,978 格；长缺口整段剔除 945,428 格；**跨缺口拼接仅 456 处，最大连续段 707,703 样本 ≈ 49.1 天**；负值 clip 0/0。diagnose：train/val/test=504.47/108.1/108.1 天，aggOffW=351.9/324.2/398.2，corr=0.3945/0.4622/0.5013（身份指标第三次稳定）；**但 on_evt/day=12.55/12.42/16.24（v3 为 5.56/5.24/6.18）**。
- **结构判定：健康**。v3 误删的 153 万桥接格回归（8.85M→10.38M）；接缝 456 处/720.7 天、最大段 49.1 天；保留跨度 720.7/787=91.7%。
- **切碎确诊（算术铁证）**：train ON 样本总数 v3≈v4（37,169 vs 37,048）、绝对 kWh 151.9≈151.5——煮沸能量与时长不变，但事件数 ×2.64、平均事件 93s→35s → v4 的「kettle 补 0」在煮沸中掉线处断言关断（实为无线表传输丢失，壶仍在烧），一次煮沸被切成 ~3 片。掉线率 1,532,978/11,323,078=13.5%，煮沸 ~15 样本 → 期望断流 2 次/煮沸 → ×2.6 吻合。另：v4 的 kWh/day 下降（0.353→0.30）系分母（天数）膨胀，非能量损失。
- **修复（v5，schema_version 5）**：kettle 短缺口桥接值改为 **ffill 前值**（关断时前值=0，「关断即 0」语义自动保持；煮沸中保持 ~2300W 不断流）；gap_policy 留痕 kettle_cells_bridged；新增**事件内微缺口**回归测试（target[500:540].min()>1500，煮沸不得被切零）；pytest 14 passed。
- **v5 与 v4 差异仅「桥接填什么值」**：格数/剔除/拼接/接缝全部不变（n、456 缝、49.1 天最大段均应复现）。
- **待用户**：git pull 重跑 prepare + diagnose（第 4 次）。预期：n=10,377,651 不变、桥接/接缝/最大段不变；evt/day 回 5-6、平均事件 90s+、on_frac 微升、kWh/day 微升 ~+0.04（煮沸中桥接格 0→2300W）、aggOffW/corr 不变。**如符合 → 数据锁定，回主线重跑 KPI/Test（test 段壶用量 +53% 漂移为主战役）。**
- 是否进入 REPORT.md：否（数据修复收尾；红旗解除+v5 锁定结论待复验后一并并入）。

### 执行实录 13（2026-09-09）：prepare 五跑 v5 复验全过 → **数据锁定**，主线重启（KPI 重新摸底）
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（v5 复验，同命令第 5 跑）：
  ```powershell
  python scripts\prepare_ukdale.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --mains-ids 1 --kettle-meter-id 10 --out D:\Work\testPython\datasets\ukdale_prepared_v2.npz
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_prepared_v2.npz
  ```

- **事实（用户回传两份）**：n=10,377,650（v4 −1：头部缺口 ffill 无前值→诚实剔除，起始 22:28:18=壶表首个真实读数）；桥接 agg 467,337 / kettle 1,532,978 格、接缝 456 处、最大段 707,703（49.1 天）——与 v4 **完全一致**（差异仅桥接填值，符合设计）；diagnose：evt/day=4.73/4.49/5.31（v4: 12.55/12.42/16.24）、平均事件 ~106s（v4: 35s）、kWh=173.2/38.7/55.7、kWh/day=0.343/0.358/0.516、aggOffW=350.3/322.6/395.9、corr=0.419/0.4868/0.5249。
- **判定：v5 全过，数据锁定。** ①绝对事件数 2386/485/574 与 v3 的 2393/483/570 几乎一致（drop 微缺口 vs ffill 两种独立处理交叉验证同一物理事实）；②evt/day 低于 v3 系分母天数更全（504.47 vs 430.2 天），非事件变少；③kWh 较 v3 +14%：煮沸中掉线格按 2300W 计（v3 整行丢弃→系统性低估，v5 更接近真值——掉线是传输丢失，不是断电）；④aggOffW/corr 第四次稳定，身份链闭环。
- **数据锁定声明**：`ukdale_prepared_v2.npz`（schema v5）+ 同名 data_spec.json 为唯一数据口径；旧 `ukdale_prepared.npz`（mains 1+2 锅炉双计时代）及其上全部 KPI/Test/调参记录**作废归档**（保留文件备查，不再作为依据）。
- **主线重启纪律**：①旧调参结论（c2/v2、nhead8、dropout 细搜等）降级为「待复核假设」，不继承为新纪元事实；②KPI 重新摸底：baseline.yaml ×3 seeds（42/2024/7）on v5 数据（不含 --test，Test 不触碰）；③**Test 预算重置：新数据纪元 2 次触碰**（旧 Test 结果基于作废数据，不计数；与原纪律同构）。
- **待用户**：三条 baseline 命令（见 STATUS 下一步），回传三份 train 输出（含 val KPI：MAE/F1/energy_error/S 综合分）。
- 是否进入 REPORT.md：否（主线重启后，KPI 摸底 + 最终锁定 + Test 通过之时，数据修复全程（实录 5-13）浓缩为「数据制备」章节一并进入）。

### 执行实录 14（2026-09-09）：v5 数据 baseline 摸底 ×3 seeds → EE −23%→−6~−9% 里程碑；Test 触碰记账+缺省翻转；val KPI 补读通道
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**：
  ```powershell
  python scripts\train.py --config configs\baseline.yaml --data-path D:\Work\testPython\datasets\ukdale_prepared_v2.npz --seed 42 --out reports\base_v5_s42
  python scripts\train.py --config configs\baseline.yaml --data-path D:\Work\testPython\datasets\ukdale_prepared_v2.npz --seed 2024 --out reports\base_v5_s2024
  python scripts\train.py --config configs\baseline.yaml --data-path D:\Work\testPython\datasets\ukdale_prepared_v2.npz --seed 7 --out reports\base_v5_s7
  ```

- **事实（用户回传三份）**：baseline.yaml（d64/h4/l2/ff128/do0.1）on v5，seeds 42/2024/7：best_epoch 5/22/20；Test MAE 8.80/6.88/7.97、RMSE 112.7/96.3/113.8、R² 0.639/0.736/0.631、**EE −0.0903/−0.0935/−0.0598**、P 0.88/0.87/0.83、R 0.75/0.825/0.75、F1 0.811/0.846/0.789；n=30000/6000/6000（linspace 均匀子采样，确定性无随机；种子只影响初始化/训练顺序）。
- **纪律事故与记账（本侧责任）**：三份输出均含 test 指标——命令未带 `--test` 仍碰了 Test。根因：`experiment.py` 的 `eval_test` **缺省 True**（`--test` 是"强制开"而非"开关"，我方上轮「不带 --test 即冻结」系语义误记）。记账：**新纪元 Test 触碰 #1 = 本次 baseline 摸底（3 seeds）**，与旧纪元「阶段 0b 基线=触碰 1」同构；剩余预算 1 次（最终锁定模型）。修复：缺省翻转 False（Test 冻结成为代码默认纪律，触碰必须显式 `--test` 或 yaml `eval_test: true`）；train.py help 措辞同步。
- **判读（里程碑）**：旧纪元**最终精调模型** Test EE −0.234（|EE| 0.234 ≫ 0.15 未过验收，实录 5 时代）；新纪元**未调参 baseline** 即 EE −0.060~−0.093，业务门槛 |EE|≤0.15 / F1≥0.75 / R≥0.70 **全过**。修 aggregate 锅炉双计 + 煮沸切碎带来的能量偏差改善，超过旧纪元全部调参努力之和——「先修数据再谈模型」路线的最终验证。MAE 6.88-8.80 亦优于旧 final 的 9.84。
- **噪声警示**：seed 方差显著（best_epoch 5/22/20；val best MAE 3.25/3.90/3.25-3.90 区间；val 曲线剧烈抖动 3.8↔6.3）。定量原因：val 6000 样本中 ON 样本仅 ≈6000×0.006≈**36 个**（约 2 次事件当量）→ F1/EE 在 val 上噪声极大。调参纪律：选型用多种子均值 + composite S，单 seed 单 epoch 的 val 指标不可作为依据。
- **val KPI 补读通道**：result.json 只含 test 指标，但 `fit()` 逐 epoch 已把完整 `val_*`（mae/f1/energy_error/precision/recall…）落盘在 history.json → evaluate.py 升级为自动注入 `best_epoch_val` 字段（不重训、不碰 Test）。
- **待用户**：git pull 后 `python scripts\evaluate.py --run-dir reports\base_v5_s{42,2024,7}` ×3，回传（重点 best_epoch_val 的 mae/f1/energy_error/precision/recall）。
- 是否进入 REPORT.md：否（待 val KPI 齐后定平移 vs 重搜策略）。

### 执行实录 15（2026-09-09）：baseline val KPI 齐表 → F1 离散化/S 判别力集中/EE 一致性判读；双探针方案（平移复核+数据量杠杆）
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（升级版 evaluate.py 注入 best_epoch_val；不重训不碰 Test）：
  ```powershell
  python scripts\evaluate.py --run-dir reports\base_v5_s42
  python scripts\evaluate.py --run-dir reports\base_v5_s2024
  python scripts\evaluate.py --run-dir reports\base_v5_s7
  ```

- **事实（用户回传三份 evaluate.py，best_epoch_val）**：seeds 42/2024/7 的 val MAE=3.827/3.901/3.245、R²=0.876/0.855/0.878、EE=−0.0421/−0.1118/−0.0583、**P/R/F1 三种子完全同值：P=1.000、R=0.8485、F1=0.9180**。
- **判读 1（F1 同值=离散化，非稳定性）**：val 6000 样本中 ON 恰 33 个（0.0060×6000，与 diagnose on_frac 吻合）；R=28/33、P=1.0（零假警报）→ 三个不同种子的模型在各自 best epoch 检出同样 28 个、漏同样 5 个。F1 在此预算下落在粗离散格上、几乎种子盲——**不得解读为"极稳定"，亦无分辨力区分配置**。漏掉的 5 个 ON 疑似系统性难点（接缝上下文/桥接残缺事件/低幅边缘），待后续定位。
- **判读 2（S 判别力集中在最噪指标上）**：手算 composite S（0.4×MAE/2000+0.4×(1−F1)+0.2×|EE|）=0.0420/0.0559/0.0451 → **0.0477±0.0060**。分解：F1 项恒定 0.0328、MAE 项 ≈0.0007 可忽略 → **S 的全部方差来自 |val EE|**（0.0084~0.0224）。当前 val 预算下"按 S 选型"≈"按 |EE| 选型"，而 EE 恰是 33 ON 样本支撑的最噪指标 → **重搜配置必须扩大 val（拟 max_samples_val 30000，ON≈180）**。
- **判读 3（val→test 一致性，漂移温和化）**：val EE −4.2/−11.2/−5.8% vs test EE −9.0/−9.4/−6.0%——方向量级吻合，无旧纪元 val −0.6%→test −23.4% 的爆炸性脱节（坏数据时代产物）。test 段壶用量 +50% 的分布漂移仍真实，但模型响应温和。新纪元基线锚（多种子口径）：**val S 0.0477±0.0060、val F1 0.918、val EE −7.1%±3.0%、test EE −8.1%±1.5%**。
- **策略决策（双探针先行，再定重搜）**：
  - **探针 A（平移复核）**：旧纪元最终优胜 v2_do00（w128 d64 **nhead8** L2 ff128 **do0** **bs64 lr3e-4 25/5**，与新 baseline 差 5 因子）on v5 ×3 seeds（42/2024/7 与 baseline 配对）→ 测"旧洞察可迁移性"，并直接与基线比较。
  - **探针 B（数据量杠杆）**：新增 `configs/baseline_100k.yaml`（与 baseline.yaml **唯一差异** max_samples_train 30000→100000，pyyaml 平铺校验通过）×3 seeds → 测"加大训练子采样"边际收益（train 池 7.26M 中心，当前仅用 0.4%），为重搜定 train 预算。
  - 两探针后定 tune.py 重搜方案（val 扩 30000、搜索空间按探针结果裁剪）。预计 GPU 合计 ~25-45 min。新跑均为 eval_test 缺省 False（不碰 Test，剩余预算 1 次）。
- **待用户**：6 条 train + 6 条 evaluate（命令见 STATUS 下一步），回传 best_epoch_val（test 字段应为 null）。
- 是否进入 REPORT.md：否（待双探针回传定重搜方案后，连同数据纪元切换一并规划）。

### 执行实录 16（2026-09-09）：双探针回传判读 → A（旧优胜平移）配对全胜、B（100k）打平；重搜方案定稿（tuning_v5.yaml）
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（探针 A 用旧纪元配置 configs\fine\v2_do00.yaml；evaluate 循环为修正引号版——裸词列表会报 ParserError）：
  ```powershell
  python scripts\train.py --config configs\fine\v2_do00.yaml --data-path D:\Work\testPython\datasets\ukdale_prepared_v2.npz --seed 42 --out reports\trans_v2do00_s42
  python scripts\train.py --config configs\fine\v2_do00.yaml --data-path D:\Work\testPython\datasets\ukdale_prepared_v2.npz --seed 2024 --out reports\trans_v2do00_s2024
  python scripts\train.py --config configs\fine\v2_do00.yaml --data-path D:\Work\testPython\datasets\ukdale_prepared_v2.npz --seed 7 --out reports\trans_v2do00_s7
  python scripts\train.py --config configs\baseline_100k.yaml --data-path D:\Work\testPython\datasets\ukdale_prepared_v2.npz --seed 42 --out reports\base100k_s42
  python scripts\train.py --config configs\baseline_100k.yaml --data-path D:\Work\testPython\datasets\ukdale_prepared_v2.npz --seed 2024 --out reports\base100k_s2024
  python scripts\train.py --config configs\baseline_100k.yaml --data-path D:\Work\testPython\datasets\ukdale_prepared_v2.npz --seed 7 --out reports\base100k_s7
  foreach ($d in 'trans_v2do00_s42','trans_v2do00_s2024','trans_v2do00_s7','base100k_s42','base100k_s2024','base100k_s7') { python scripts\evaluate.py --run-dir "reports\$d" }
  ```

- **事实（用户回传六份，均为 test:null ✓）**：
  - 探针 A（v2_do00=w128 d64 h8 L2 ff128 do0 bs64 lr3e-4 25/5，**composite 选型**）：seeds 42/2024/7 的 best_ep 3/4/6；val S=0.0351/0.0261/0.0375、EE=+0.66%/+2.14%/+0.16%、F1=0.918/0.952/0.909、P=1.0/1.0/0.909、R=0.848/0.909/0.909、MAE=5.06/13.85/3.90、R²=0.869/0.687/0.878。
  - 探针 B（baseline_100k，MAE 选型）：best_ep 4/5/5；val S（手算）=0.0514/0.0606/0.0360、EE=−8.9%/−3.2%/−2.4%、F1=0.918/0.867/0.923、MAE=3.83/4.56/2.72；runtime 115-131s（baseline 42-101s，≈2.7×）。
- **判读 1（探针 A 胜，旧洞察可迁移）**：与 baseline 配对（同 seeds）：**S 三种子全胜**（ΔS=−0.0069/−0.0298/−0.0076，均值 0.0329±0.0060 vs 0.0477±0.0060）；**val EE 从 −7.1%±3.0% 收敛到 +1.0%±1.0%**（近零且转正）；F1 两平一升。旧纪元结论（do0/nhead8/bs64/lr3e-4/25/5 + composite 选型）在新数据上成立。**代价如实记录**：val MAE 全面变差（7.60±4.4 vs 3.66±0.30，s2024 达 13.85/R²0.687）——S 中 MAE 项权重 0.4×MAE/2000≈可忽略，composite 选型实质用点误差换 EE/F1；13.85W 均误业务上仍小（壶功率 0.6%），但 s2024 该跑的拟合质量需在细搜多 seed 阶段澄清。
- **判读 2（探针 B 平，数据杠杆暂无可测收益）**：配对 ΔS=+0.0094/+0.0047/−0.0091（1 胜 2 负，均值 0.0493±0.0125 vs 0.0477±0.0060 打平）；EE 2/3 种子改善但不一致（−4.8%±3.6% vs −7.1%±3.0%）；F1 均值微降（0.903）；代价 2.7×。30k→100k 在当前 6000-val 分辨率下无显著收益 → 搜索用 30k train（省算力），数据杠杆留待细搜后以 30000-val 复评。
- **重搜方案定稿（configs/tuning_v5.yaml 入库）**：①搜索空间以 v2_do00 邻域为中心裁剪（w∈{96,128,192}、d∈{64,128}、h∈{4,8}、L∈{1,2}、ff∈{128,256}、do∈{0,0.1}、bs∈{64,128}、lr∈{2e-4,3e-4,5e-4}、wd∈{1e-5,1e-4}、25/5；删旧空间低天花板/未用区域）；②**val 扩 30000**（ON≈180，恢复 S/F1 判别力）；③train 保持 30k；④gates/复合分口径不变（KPI 冻结）；⑤trial seed=42+i（tune.py 机制，单 seed 排名噪声由细搜多 seed 复核消化，**v2_do00 作锚候选必入细搜**）。
- **待用户**：`python scripts\tune.py --config configs\tuning_v5.yaml --data-path <v2.npz> --out reports\tuning_v5_p1`（32 trials，预计 30-60 min GPU），回传末尾门槛统计 + Top-5 表（及 tuning_summary.csv 前 10 行如方便）。
- 是否进入 REPORT.md：否（重搜+细搜+锁定+Test 后一并规划）。

### 执行实录 17（2026-09-09）：重搜 32 trials 判读（32/32 门槛、w96 崛起、val30000 判别力恢复）→ 细搜批次 1 设计（5 配置 ×3 seeds）
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（本节「待用户」处命令的完整实录路径版）：
  ```powershell
  python scripts\tune.py --config configs\tuning_v5.yaml --data-path D:\Work\testPython\datasets\ukdale_prepared_v2.npz --out reports\tuning_v5_p1
  ```

- **事实（用户回传）**：32/32 全过业务门槛（F1≥0.75/R≥0.70/|EE|≤0.15）；Top-5（单 seed 各异 43-74）：#1 trial20 S=0.0400/MAE 5.75/F1 0.903/EE+0.001（w96 d64 h4 L2 ff128 do0 bs128 lr5e-4 wd1e-4）、#2 trial29 S=0.0493（w96 d64 h4 **L1** ff256 do0 bs64 lr5e-4 wd1e-5）、#3 trial11 S=0.0523（w96 d64 h8 L2 ff256 do0.1 bs64 lr2e-4）、#4 trial12 S=0.0538（w96 d64 h4 L1 ff128 do0.1）、#5 trial24 S=0.0542（w96 **d128** h4 L2 ff256 do0 bs64 lr3e-4）；CSV top-10 里 w96 占 7、w128 占 2、w192 占 1；git_commit=528a01c 留痕链正常。
- **判读**：
  1. **门槛失去区分度**（32/32）→ v5 数据+val30000 下门槛只是守门员，选择靠 S——符合设计预期，非异常。
  2. **w96 崛起**（top-4 全 w96、top-10 占 7）：旧纪元锁 w128，新数据偏好 9.6 分钟上下文（v5 桥接语义下窗口边界变了，合理性待细搜确认）。
  3. **val30000 判别力恢复**：F1 跨 trial 有分布（top-10 内 0.860–0.903，6000-val 时代卡死 0.918 同值）；S 三分量均有贡献空间（top-1 分解：F1 项 0.0388 + MAE 项 0.0012 + EE 项 ≈0）。
  4. **EE 近零复现**：top-2 |EE|<0.004——探针 A 的 EE 归零特性在搜索中重现，非孤例。
  5. **单 seed 噪声未消**：trial seeds 各异（62/71/53…），top-1 可能含幸运成分（旧纪元锚 σ0.048 教训）→ 必须多 seed 复核后才可锁定。
  6. h4 占 top-5 四席（与旧纪元锁 h8 相反）；v2_do00 精确配置未被搜索抽中（最接近 trial31 w128 h8 ff256 排 #6）→ 锚候选须手动入细搜。
- **细搜批次 1 设计（configs/fine_v5/ 五配置入库，pyyaml 逐项校验+协议不变量校验通过）**：全部 val30000 + 25/5 + composite + eval_test:false（与重搜同口径）× seeds 42/2024/7（配对）：
  - **F0**=v2_do00（旧纪元最终优胜，锚候选，实录 16 承诺必入）
  - **F1**=trial20（搜索 top-1，w96 h4 do0 bs128）
  - **F2**=trial29（top-2，L1 单层最小容量代表）
  - **F3**=trial11（top-3，h8+do0.1 反方向代表）
  - **FB**=baseline 架构锚（协议归一版 25/5+composite；回答"搜索架构是否真优于不调参架构"）
  - 共 15 runs ≈ 20-35 min；汇总用 summarize_fine.py（best=val_score 最小行，与早停同口径）。
- **事故记录**：回合初沙箱第二次被平台重置（HEAD 漂回 7824bb4、/tmp 清空）；远端 528a01c 完好 → fetch+逐文件哈希对账（全一致）+reset 恢复，零丢失（实录 11 流程复用）。
- **待用户**：15 条 train + 1 条 summarize（命令见 STATUS），回传汇总表。
- 是否进入 REPORT.md：否（细搜判读+锁定+Test 后一并规划）。

### 执行实录 18（2026-09-09）：细搜批次 1 判读（F0 旧冠军夺冠、FB 垫底、赢家诅咒再证）→ 批次 2 单变体 F4（冠军×w96）+ Test 预注册
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（5 配置 ×3 seeds，全部不带 --test）：
  ```powershell
  foreach ($c in 'f0_v2do00','f1_t20','f2_t29','f3_t11','fb_basearch') {
    foreach ($s in 42,2024,7) {
      python scripts\train.py --config "configs\fine_v5\$c.yaml" --data-path D:\Work\testPython\datasets\ukdale_prepared_v2.npz --seed $s --out "reports\fine_v5\${c}_s$s" } }
  python scripts\summarize_fine.py --runs-dir reports\fine_v5
  ```

- **事实（用户回传 5 变体 ×3 seeds，val30000 同口径）**：score 排序 F0_v2do00 **0.0522±0.0045** < f3_t11 0.0546±0.0093 < f1_t20 0.0567±0.0077 < f2_t29 0.0582±0.0090 < fb_basearch 0.0604±0.0098。关键分量：F0 的 EE=+0.0003±0.0062（死零）且 σ_score 最小；f1_t20 recall 最高 0.9045 但 precision 最低 0.8641（换检出的代价）；FB 的 F1 0.8623 全场最低。
- **判读**：
  1. **F0（v2_do00）夺冠**：均值最低+σ 最小+EE 归零+跨纪元（旧纪元冠军在新数据新口径下仍第一）——旧洞察的迁移性再次确认。
  2. **FB 垫底**：调参价值坐实（搜索/旧优胜配置全面优于不调参架构，ΔS≈0.008、F1 差 0.014-0.021）。
  3. **赢家诅咒再证**：搜索 top-1（trial20 单 seed 0.0400）三种子复核 0.0567±0.0077（回落 +0.017）——32 选 1 的乐观偏差与旧纪元锚教训一致，多 seed 复核纪律必要。
  4. **统计诚实**：F0 vs F3 差 0.0024，SEM 各 ~0.003/0.005 → top-4 差距在噪声内；F0 领先依据=分量一致（唯一 EE 死零）+σ 最小+跨纪元稳健，属判断而非铁证（与旧纪元锁 v2 时的定性同构）。
  5. **MAE 披露**：F0 的 MAE 8.93±5.35（σ 大：一个种子 ~14，与探针 A s2024 模式跨 val 规模复现，系该种子收敛盆地问题）；S 对 MAE 近盲（0.4×MAE/2000）、业务不设 MAE 门槛——如实记录，不作淘汰依据。
  6. **悬而未决的交叉点**：重搜 w96 占 top-10 七席，但冠军协议（do0/h8/bs64/lr3e-4）从未在 w96 上测过 → 批次 2 单变体补刀。
- **批次 2 设计（configs/fine_v5/f4_v2do00_w96.yaml 入库，pyyaml 校验：与 F0 唯一差异 data.window_size 128→96）**：×seeds 42/2024/7，3 runs ≈5 min。**判定树（预注册）**：F4 均值 < 0.0522 → 锁 F4（w96 偏好迁移到冠军协议）；F4 ≥ 0.0522 → 锁 F0（w96 为搜索交互效应，不迁移）。
- **Test 预注册（锁定后执行，预算最后一次）**：`train.py --config <锁定配置> --seed 7000（新鲜种子族，未参与任何搜索/细搜）--out reports\final_v5\<名>_test --test` 恰好一次；验收口径（沿用旧纪元）：S_test ≤ val 均值+0.015（无分布漂移）且 F1≥0.75 / R≥0.70 / |EE|≤0.15。
- **待用户**：F4 ×3 + summarize 全目录（自动重聚合 18 runs）。
- 是否进入 REPORT.md：否（锁定+Test 后一并规划）。

### 执行实录 19（2026-09-09）：批次 2 判读 → 判定树命中，**锁定 F4 = v2_do00×w96**；Test 最终一跑（预注册）交付
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（F4 ×3 seeds + 全目录重聚合）：
  ```powershell
  foreach ($s in 42,2024,7) {
    python scripts\train.py --config configs\fine_v5\f4_v2do00_w96.yaml --data-path D:\Work\testPython\datasets\ukdale_prepared_v2.npz --seed $s --out "reports\fine_v5\f4_v2do00_w96_s$s" }
  python scripts\summarize_fine.py --runs-dir reports\fine_v5
  ```

- **事实（用户回传 F4 ×3 seeds，seeds 42/2024/7）**：score **0.0472±0.0042**（vs F0 0.0522±0.0045）；MAE 5.80±0.77（F0 8.93±5.35）；RMSE 77.4±4.5（F0 90.5±15.0）；F1 **0.8943±0.0169**（六变体最高）；P 0.9102（最高）；R 0.8794；EE +0.0170±0.0176；best_ep 8.0。
- **判定：预注册判定树命中（F4 < 0.0522 → 锁 F4），无事后挑选空间。** 且 F4 在 score/MAE/RMSE/F1/precision **五项同时第一**：
  1. **MAE 盆地问题消失**（8.93±5.35 → 5.80±0.77）：F0 那个陷差盆地的种子在 w96 下被治好——综合分与点精度这次同向，不再是"拿 MAE 换 EE"的权衡。
  2. **w96 偏好迁移到冠军协议**：重搜的窗口发现（实录 17）与旧纪元冠军协议（实录 16 探针 A）杂交成功 → 最终配置 = **w96 + d64 nhead8 L2 ff128 do0 bs64 lr3e-4 wd1e-4（25/5 composite）**，跨两个数据纪元的洞察合并。
  3. EE +1.7%±1.8%：放弃 F0 的死零（±0.0003）换取 F1 +0.018 / MAE −3.1 / σ 减半——|EE|≤0.15 门槛内富余 9 倍，权衡明确占优。
  4. 统计诚实：ΔS=0.0050 未达严格显著（合成 SEM≈0.007），支撑=分量一致性（5/6 指标占优）+σ 最小+预注册规则命中，三重非单证据。
- **Test 预算审计**：新纪元触碰 #1=baseline 摸底（实录 14 记账）；**#2=本次 F4 最终一跑，预算耗尽**（与预注册协议一致：seed 7000 新鲜族 / --test 恰一次）。
- **验收口径（预注册，沿用旧纪元）**：S_test（由 test mae/f1/EE 按复合分公式计算）≤ 0.0472+0.015=**0.0622**；F1≥0.75；R≥0.70；|EE|≤0.15。
- **待用户**：Test 一跑命令（见 STATUS）；回传 evaluate.py 全文，S_test 由本侧计算并裁定验收。
- 是否进入 REPORT.md：Test 裁定后，数据纪元切换+调参全程（实录 5-19）一并规划进入（含 TUNING_GUIDE 战史）。

### 执行实录 20（2026-09-09）：Test 终局回传 → **四线全过，验收通过，任务 3 收官**
- **用户执行命令（2026-09-09 补充留痕，按用户机器实录路径）**（预注册协议：seed 7000 新鲜族，--test 恰好一次）：
  ```powershell
  python scripts\train.py --config configs\fine_v5\f4_v2do00_w96.yaml --data-path D:\Work\testPython\datasets\ukdale_prepared_v2.npz --seed 7000 --out reports\final_v5\f4_test --test
  python scripts\evaluate.py --run-dir reports\final_v5\f4_test
  ```

- **事实（用户回传 Test 一跑，seed 7000，预注册协议，best_epoch 6）**：test MAE=6.4156、RMSE=90.33、R²=0.7788、**EE=+0.0523**、P=0.8636、**R=0.9268**、**F1=0.8941**；n=30000/30000/6000；eval_test:true（本纪元第 2 次显式触碰，预算就此耗尽）。
- **验收裁定（预注册口径，S_test 本侧按公式计算）**：
  | 线 | 数值 | 验收线 | 判定 |
  | --- | --- | --- | --- |
  | S_test | **0.0541** | ≤0.0622（val 0.0472+0.015） | ✅（富余 0.0081） |
  | F1 | 0.8941 | ≥0.75 | ✅ |
  | Recall | 0.9268 | ≥0.70 | ✅ |
  | \|EE\| | 5.2% | ≤15% | ✅（富余 3 倍） |
  **无分布漂移**（ΔS=+0.0069<0.015）。**验收通过。**
- **判读**：①test F1 0.8941 与 val 0.8943 几乎重合；②recall 0.9268>val 0.8794——test 段壶用量 +50% 的"漂移"反而利于检出（事件多→召回容易），precision 相应微降（0.8636）；③EE +5.2%（微高估）vs 旧纪元 −23.4%（严重低估）——方向都变了，量级缩 4.5 倍；④对照旧纪元最终 Test S≈0.1244 → 0.0541，**2.3 倍改善**。
- **任务 3 收官**：任务起点"Test 未通过（EE −23%）"的原始阻塞正式关闭。收官产物：`REPORT.md` v1.0 创建（稳定结论：推荐版本/数据口径/KPI/七条结论/版本史）、`TUNING_GUIDE.md` v2.0 整篇重写（六条铁律+SOP+两纪元完整战史+踩坑清单）、README 生产推荐段更新、STATUS 收尾仪式。Test 预算审计闭合：#1 摸底（实录 14 记账）+ #2 本次=2/2。
- 是否进入 REPORT.md：**是**（本次触发创建 v1.0——数据纪元切换+配置锁定+验收通过三条件同时满足）。

### 执行实录 21（2026-09-09）：流程复盘——执行实录为何系统性遗漏用户执行命令（ROLE.md 条款层归因）
- **类型**：用户专题（流程/文档质量复盘；分析全文见当轮对话，本节为沉淀版）
- **现象**：25 个实录节中 20 节缺用户命令（aa070d1 已补齐）。遗漏呈系统性：命令作为"下发的未来指令"时被写全（实录 1 补充/3 补充系列命令块），作为"已执行的历史"时被裁（实录 9-19 只写"命令见 STATUS"）。
- **ROLE.md 条款层归因（五层叠加）**：
  1. **收尾条款只沉淀"结果与结论"**：资深电力算法专家·收尾="结论沉淀到 REPORT_TEST.md"；实验/调参教练·收尾="实验结果与结论追加到 REPORT_TEST.md"——两处均无"命令/过程"字样；命令的去处是教练「执行」条款的三段式（操作→预期→异常），即**对话层**。角色体系把"过程（命令）"定位为对话、"产出（结果/判读）"定位为档案。
  2. **职责边界盲区**：教练"不应做：替用户跑实验（实验由用户自执行）"→ 用户侧操作被视为对方执行细节，我方沉淀责任只覆盖方案/判读/修复。命令的归档**无角色负责**。
  3. **验收标准缺口**：教练验收="用户按步骤可复现"（即时性验收：当时照做能跑），无"档案可复现"维度（事后翻文件能重跑）。
  4. **表达偏好放大**：默认角色"面向结论、先结论后依据"+"极简优先，能一句话说清不用一段话"→ 实录模板自然长成"事实→判读→决策"三段，命令块被极简偏好系统性裁掉。
  5. **结构放大器**：BOOTSTRAP 台账/专题模板无命令字段；命令实际落位 STATUS「下一步」——**滚动覆盖的活文件**，新命令每轮顶掉旧命令 → 命令持久化链条三层皆断（对话=临时、STATUS=滚动、REPORT_TEST=无字段）。
- **定性**：未违反任何明文禁令（所有结果数字真实、来源可查），但违反「验收标准·每个数字可追溯」的**精神**——数字可追溯到实录、"怎么跑出来的"（命令）不可追溯，追溯链只到一半。
- **改进建议（待用户裁定后落盘）**：①ROLE.md 两角色「工作方式·收尾」补"执行实录须含命令+输出+判读三件套（命令含实录路径）"；②角色「验收标准」补"档案可复现"维度；③（可选）BOOTSTRAP 专题模板加"用户执行命令"字段。
- 是否进入 REPORT.md：否（流程改进，非算法结论）。

### 执行实录 22（2026-09-09）：prepare/diagnose 泛化到其他 house/电器（任务 4）——通用名+别名兼容+阈值表
- **本任务角色**：工程实现工程师（代码泛化改造）
- **用户执行命令（沙箱验证，2026-09-09）**：
  ```powershell
  /tmp/dvenv/bin/python -m py_compile scripts/prepare_ukdale.py scripts/diagnose_split.py scripts/parse_nilmtk_metadata.py
  /tmp/dvenv/bin/python -m pytest tests/ -q --ignore=tests/test_model.py
  ```
- **输出（沙箱实录）**：COMPILE_OK；**16 passed**（14 个既有用例全过=v5 冻结口径零回归；新增 test_prepare_generic_appliance_flags + tests/test_diagnose_split.py）。
- **改动清单**：
  1. `prepare_ukdale.py`：`--appliance`（标签，默认 kettle）/`--appliance-meter-id`/`--appliance-gap-min` 通用名；`--kettle-meter-id`/`--kettle-gap-min` 保留为兼容别名（同时给且值不同→报错，不接受静默覆盖）；data_spec 新增 appliance/appliance_* 通用键，kettle 路径额外保留 legacy 键（v5 冻结口径与既有测试零改动通过）；schema_version 保持 5（policy 未变，仅标签）。
  2. `diagnose_split.py`：`--appliance` + 默认阈值表（kettle 500 / fridge·freezer 50 / dish_washer·washer_dryer·washing_machine 20 / microwave 200 / boiler 100W；未收录回退 500）；显式 `--on-threshold` 优先；判读提示注入电器名 + 常开型电器提示（fridge on_frac≈1 属正常）。
  3. `parse_nilmtk_metadata.py`：输出尾追加 prepare 命令模板（--appliance-meter-id 口径）。
  4. `README.md`：新增「泛化到其他 house / 电器」三步工作流章节。
- **用户侧验证命令（待用户执行，真实数据）**：
  ```powershell
  # House1 洗碗机（meter6，metadata 已知）全链：
  python scripts\prepare_ukdale.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --house 1 --mains-ids 1 --appliance-meter-id 6 --appliance dish_washer --out D:\Work\testPython\datasets\ukdale_dw.npz
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_dw.npz --appliance dish_washer
  # 其他 house 探查（以 House2 为例）：
  python scripts\prepare_ukdale.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --house 2 --list-meters
  python scripts\parse_nilmtk_metadata.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --house 2
  ```
- **判读要点（预注册预期）**：dw 的 diagnose——aggOffW 数百 W（真实基线）、corr<0.5、on_frac 远低于 kettle 属正常（洗碗机占空比低）；House2 mains 表号须以 list-meters+parse 实测为准（**不猜表号**纪律）。**每个 (house, appliance) 为独立数据纪元：Test 预算各 2 次，须重走身份验证→摸底→搜索→锁定→Test 全流程。**
- 是否进入 REPORT.md：否（工具泛化非实验结论；真实数据验证后再议 README/REPORT 收录）。

### 执行实录 23（2026-09-09）：泛化真实数据验证——House1 dish_washer 全链过 + House2 探查（双 mains/19 电器表）；dw 事件阈值预警
- **本任务角色**：工程实现工程师（验证判读）→ 转实验/调参教练（下一电器纪元设计）
- **用户执行命令（2026-09-09，真实数据）**：
  ```powershell
  python scripts\prepare_ukdale.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --house 1 --mains-ids 1 --appliance-meter-id 6 --appliance dish_washer --out D:\Work\testPython\datasets\ukdale_dw.npz
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_dw.npz --appliance dish_washer
  python scripts\prepare_ukdale.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --house 2 --list-meters
  python scripts\parse_nilmtk_metadata.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --house 2
  ```
- **输出关键数字**：①House1 dw：n=10,685,551（742.1 天，跨度保留 94.3%），mains 网格 11,323,076 / dw 网格 11,323,174（meter6 覆盖 ≈99.96% 近乎无缺口，远健壮于 kettle 表的 79%）；diagnose（20W 表值）：evt/day 2.53/2.17/2.70、on_frac 0.0222/0.0188/0.0235、ON 功率 mean 722/649/699、**med 120**、p95 2363、kWh/day 0.405/0.313/0.414、aggOffW 348.6/317.5/406.4、corr 0.413/0.400/0.353。②House2：20 表（meter7 在 metadata 有映射但 h5 无组，同 House1 meter0 现象）；**双 mains 结构**——meter1（apparent，6s，2013-02-17→10-10，235 天）+ meter20（**active，1s，12,166,699 样本**，亚秒时间戳，2013-04-16→10-10，177 天）；电器表两批分期安装（m8-11 起于 2013-04-16、m12-19 起于 2013-05-20）；映射：kettle=m8、rice cooker=m9、washing machine=m12、dish washer=m13、fridge=m14、microwave=m15、toaster=m16 等 19 项。
- **判读**：
  1. **House1 dw 身份链通过**：aggOffW 317-406（真实基线）+ corr 0.35-0.41（aggregate 含 dw+其他）——NILM 前提成立，泛化口径产出结构合法（时间范围与 kettle 口径一致到分钟级）。
  2. **dw 事件定义预警（若开 dw 纪元须先处理）**：0.405 kWh/day ÷ 2.53 evt/day ≈ **0.16 kWh/事件**，远低于洗碗机典型周期 1-1.5 kWh；ON 功率双峰（med 120W 泵相位 / p95 2363W 加热相位）→ 20W 阈值把一个洗涤周期的多相位**切分成多个"事件"**（水壶 v4 补0切碎的原生版）。做 dw 纪元前须做事件阈值敏感性（--on-threshold 100/200 对照）或事件合并（min-gap）——已记入预警，不影响本次验证结论。
  3. **缺口处理行未随贴**：用户贴文跳过了 prepare 的「缺口处理」行（桥接格数/接缝/最大段在 data_spec.json gap_policy 有档）；后续回传请带上该行（档案可复现纪律）。
  4. **House2 结构判读**：meter1 与 House1 meter1 同型（6s apparent mains）；meter20 与 House1 meter54 同型（1s active mains，12.17M 样本）——且 prepare 的 6s 网格 resample 对 1s 源**自动 bin-mean 降采样**（实录 9 修复的副产品能力），meter20 可直接 `--mains-ids 20` 使用（active 语义更纯，代价是跨度 177 天 < meter1 的 235 天）。kettle=meter8 纪元与 mains 交集 ≈2013-04-16→10-10（约 177 天，独立数据纪元）。
  5. **泛化验证结论**：prepare（通用名+别名）/diagnose（阈值表+标签）/parse（命令模板）在真实 House1 dw 与 House2 探查全链工作正常——任务 4 代码目标达成。
- **下一步（House2 kettle pilot，独立纪元 Test 预算 2 次）**：命令见 STATUS；预期 n≈2-2.5M、~170-180 天、aggOffW 数百 W、corr<0.5（证伪口同前：aggOffW≈0 则 mains 判错，备选 meter20）。
- 是否进入 REPORT.md：否（工具验证+探查判读，非实验结论）。

### 执行实录 24（2026-09-09）：House2 kettle pilot 判读——身份链过/数据可用，但 kettle 网格点异常对账（代码沙箱复核无罪，probe 交付定谳）
- **本任务角色**：实验/调参教练（判读）+ 工程实现工程师（对账复现）
- **用户执行命令（2026-09-09，真实数据）**：
  ```powershell
  python scripts\prepare_ukdale.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --house 2 --mains-ids 1 --appliance-meter-id 8 --appliance kettle --out D:\Work\testPython\datasets\ukdale_h2_kettle.npz
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_h2_kettle.npz --appliance kettle
  ```
- **输出关键数字**：n=2,145,698（149.0 天）；桥接 agg 97,137 / kettle 52,038；长缺口剔除 1,231,859；接缝 9 处；最大段 1,091,839（**75.8 天**）；diagnose（500W 表值）：evt/day 5.23/7.78/3.71、on_frac 0.0096/0.0133/0.0064、ON 功率 med 2947/2973/2969、p95 3028/3051/3052（**H2 壶为 3kW 档**，H1 为 2344W）、kWh/day 0.70/0.97/0.48、aggOffW 285.4/308.6/266.0、corr 0.6212/0.6665/0.555。
- **判读 1（身份链过）**：aggOffW 266-309（真实基线）+ evt 形态壶样（3.7-7.8 次/天、3kW）+ corr 0.56-0.67（高于 H1 的 0.35-0.53——H2 壶功率 2950W 对 ~300W 家庭均值占方差比更大，方向合理）→ meter1=mains / meter8=kettle 成立。
- **判读 2（结构健康）**：交叠区 176.3 天保留 149.0 天（84.5%）；9 接缝；最大段 75.8 天。
- **判读 3（网格点异常对账，本节核心）**：prepare 打印「kettle 网格点 3,377,557」**大于 meter8 跨度理论上限**（2013-04-16 21:18→10-10 05:15 ≈ 2,539,200 格），超出 838,399 格 ≈ 58.2 天（恰为 Feb17→Apr16 间隔）；且 n+剔除=3,377,557=外连接并集 > mains 网格（3,377,384）173 格（≈172 格在 mains 起点 2013-02-17 16:17:34 之前 +1 格在终点后 05:15:58=meter8 表列终点）→ 算术指向 **meter8 表内含 ~172 个 mains 覆盖前的 finite 杂散行（约 Feb 17 16:00 起）**。**但**与上轮 list-meters（meter8 起点 2013-04-16、n=2,094,523）互斥。
- **沙箱孪生复现（代码无罪证明）**：构造 mains 早于 kettle 7 天的迷你 House2——变体 A（无杂散）：kettle 网格=自身跨度 ✓ 正确；变体 B（表内 20 行 mains 前杂散）：kettle 网格≈mains 全跨度、**list-meters 起点变为杂散时间**、npz 输出与变体 A 完全一致（杂散行因 agg 缺失被长缺口剔除）→ 代码不可能从 Apr 起点数据产出 3.38M 网格；若表有杂散行，list-meters 必然显示 Feb 起点 → **用户两份输出对应不同的文件状态（或转写误差）**，须以当前文件实测定谳。
- **判读 4（npz 有效性）**：由变体 B 证明：即便存在杂散行（mains 覆盖之外），输出 npz 不受影响——本 npz 的 target 严格为 Apr 16→Oct 10 的壶数据，可用性不因异常悬置（但纪元锁定仍待 probe 定谳后宣布，先对账再锁定纪律）。
- **判读 5（H2 drift 反向签名）**：val 段 7.78 evt/day / 0.97 kWh/day 远重于 test 3.71 / 0.48——与 H1（test 重 +50%）方向相反；H2 纪元若训练，选型段偏重、考核段偏轻，EE 预期偏正方向。另：val 段仅 22.35 天（若训练，val 子样本 30000 时 ON≈400，F1 分辨率尚可）。
- **沙箱事故记录**：第四次平台重置（HEAD 漂回 7824bb4、/tmp 清空）；逐文件哈希对账全一致后 reset 恢复，零丢失；新增 scripts/probe_meter.py（零过滤单表探查：rows/min/max/NaT/重复/NaN/截断点前后明细，沙箱变体 B 测试通过）。
- **待用户**：probe meter8 定谳（+可选重跑 list-meters 对照当前文件状态）：
  ```powershell
  python scripts\probe_meter.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --house 2 --meter 8 --cutoff "2013-04-16 21:18:09"
  ```
- 是否进入 REPORT.md：否（pilot 判读+工具，纪元未锁）。

### 执行实录 25（2026-09-10）：House2 kettle 网格异常 probe 定谳——8 行 Feb17 杂散坐实/算术全闭环/「互斥」系转写误差；**H2 纪元锁定** + 摸底交付；probe crash 修复
- **本任务角色**：资深电力算法专家（对账判读）+ 工程实现工程师（probe v2 修复）
- **用户执行命令（2026-09-10，真实数据，实录 24 交付版）**：
  ```powershell
  python scripts\probe_meter.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --house 2 --meter 8 --cutoff "2013-04-16 21:18:09"
  ```
- **输出关键数字**：rows=2,094,523（**与上轮 list-meters 完全一致**）；index min=**2013-02-17 16:00:22+00:00** / max=2013-10-10 06:15:56+01:00；NaT=0 / 重复=0 / 单调=True；NaN=0 / inf=0 / finite=全 / 值域 0.0–3998.0；cutoff=2013-04-16 21:18:09+01:00 **之前 8 行**（前 5 行 16:00:22/25/31/37/43 全 0.0）、之后 2,094,515 行；随后 line 53 crash（ndarray 无 to_numpy）——**crash 发生在决定性证据打印之后，判读不受影响**。
- **判读 1（杂散坐实，定谳）**：meter8 表头夹带 **8 行 2013-02-17 16:00:22 起的 0W 行**（安装测试残留，与 m1 起点 16:17:34 同日——2/17 为 H2 记录系统安装日），主体数据 2,094,515 行自 Apr 16 21:18:09 起。修正实录 24 的「~172 行杂散」猜测：杂散实际 8 行；173 格并集超出=mains 自身起点晚 172 格（kettle resample 满跨度在 m1 首格之前的区间，几乎全为 NaN 格）+1 格终点后，**并非杂散行数**。
- **判读 2（算术全闭环，零自由参数）**：
  1. kettle 满跨度格数 = 6s bins(Feb 17 16:00:22 → Oct 10 06:15:56) = **3,377,557，与 prepare 打印一字不差**——「超跨度上限」系误把主体起点 Apr 16 当表起点；真实跨度 234.55 天 = 主体 176.37 天 + Feb17→Apr16 静默 58.17 天。
  2. 并集 3,377,557 = mains 3,377,384 + **173 = 头 172 + 尾 1**：m1 首格 16:17:30（raw min 16:17:34）、末格 06:15:48（由 mains 网格数反推，raw max ∈ [06:15:48, 06:15:54)）；kettle 首格 16:00:18 / 末格 06:15:54 → mains 格集 ⊆ kettle 格集，`DataFrame({agg, target})` 外连接并集=kettle 网格，结构性成立。
  3. 剔除 1,231,859 = 头部 172 + 静默区 837,606（58.17 天，mains 有/kettle 无）+ 交叠内双缺 394,081（27.37 天）——**一字不差**。
  4. 保留 2,145,698 = 149.0 天 = 交叠 176.37 天的 84.5% ✓（与 diagnose 全部对上）。
- **判读 3（「互斥」消解——记录侧转写误差，非文件状态变化）**：list-meters 的 start=read_power_series 原始 min（仅 dedup/sort/isfinite 过滤，**0W 行保留**）→ 在本文件上 meter8 必显 2013-02-17 16:00:22。两处 n=2,094,523 完全一致坐实**同一文件状态**；实录 24 所记「meter8 起点 Apr 16」与「05:15:58=meter8 表列终点」均有转写误差（后者与实测 06:15:56+01:00=05:15:56 UTC 差 2 秒，佐证记录侧来源）。**撤回实录 24「两份输出对应不同文件状态」假说**——文件单一稳定。
- **判读 4（npz 有效性终审）**：8 行杂散全落在 mains 覆盖之前的头部 172 格区域（m1 自 16:17:34 起）→ 制备时随长缺口整段剔除，npz 首样本 Apr 16 21:18:09、target 严格为主体壶数据——与实录 24 孪生复现（有无杂散 npz 输出一字不差）互证闭环。
- **判读 5（孪生 v2 端到端复现，真实时间戳）**：以 probe 实测时间戳精确构造迷你 H2（m8：8 行 Feb17 杂散+Apr 16 21:18:09 起主体；m1：16:17:34 起）跑 prepare → 打印「**mains 网格点 3377384 / kettle 网格点 3377557**」与用户 pilot **一字不差**；头部+静默剔除 837,778=172+837,606 ✓（交叠内剔除 394,081 为真实缺口所有，孪生稠密化故为 0）；时间范围首格 21:18:06（6s 对齐）。m1 侧边界取实录 24 记录与 mains 网格数反推区间，其 ground truth 是 pilot 打印本身；m1 精确边界待可选 probe 存档。机制链全链复现：杂散行→meter8 跨度起点 Feb17→resample 满跨度 3,377,557→并集=kettle 网格。
- **事故与修复（probe v2）**：line 53 `pre.to_numpy()`——`DatetimeIndex < Timestamp` 返回 **ndarray**（无 to_numpy）→ 改 `np.asarray`；上轮沙箱测试未覆盖「截断前行存在」分支（教训：交付脚本必须以用户同款 CLI 入口端到端跑全部分支，后加代码必须重新回归）。增强：截断前行 ≤20 行全量打印（补齐杂散行 6–8 明细）、新增截断后前 5 行、新增「6s 网格满跨度格数」行（floor 公式与 `resample('6s', origin='epoch')` 满跨度沙箱对拍一致）。孪生 v2 三项测试全过（m8+cutoff 无 crash/8 行全量/格数 3,377,557；m1 无 cutoff 格数 3,377,384；prepare 端到端）；pytest 16 passed 零回归。
- **判读 6（纪元锁定宣布）**：**House2 kettle 数据纪元锁定**——身份链过（实录 24：aggOffW 266-309/evt 3.7-7.8 次/天/3kW 壶/corr 0.56-0.67）+ 网格异常对账闭环（本实录）+ npz 双证有效（孪生×probe 机制）。npz=`D:\Work\testPython\datasets\ukdale_h2_kettle.npz`（mains=m1/kettle=m8）为该纪元唯一口径；**Test 预算 2 次 untouched**（摸底不带 --test，eval_test 缺省已 False）；drift 反向签名（val 7.78 evt/day 重 / test 3.71 轻，与 H1 相反）记录在案，EE 预期偏正方向。
- **待用户（摸底 baseline ×3 seeds，主命令；H2 纪元 Test 冻结）**：
  ```powershell
  python scripts\train.py --config configs\baseline.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed 42 --out reports\base_h2_s42
  python scripts\train.py --config configs\baseline.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed 2024 --out reports\base_h2_s2024
  python scripts\train.py --config configs\baseline.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed 7 --out reports\base_h2_s7
  ```
  可选存档复核（10 秒级，不阻塞摸底；预期 m8 起点必显 2013-02-17 16:00:22、m1 满跨度格数 3,377,384，不符再开对账）：
  ```powershell
  python scripts\probe_meter.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --house 2 --meter 8 --cutoff "2013-04-16 21:18:09"
  python scripts\probe_meter.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --house 2 --meter 1
  python scripts\prepare_ukdale.py --h5-path D:\Work\testPython\datasets\ukdale.h5 --house 2 --list-meters
  ```
- **回传要求**：摸底三份完整 stdout（含逐 epoch val 行与 best epoch 摘要）；摸底不碰 Test；可选复核输出一并存档。
- 是否进入 REPORT.md：否（对账+工具+纪元锁定；实验结论待摸底产生）。

### 执行实录 26（2026-09-10）：House2 kettle 摸底 baseline ×3 seeds stdout 判读——训练健康/Test 冻结首次实战生效（test:None ×3）；val F1/EE 待补读
- **本任务角色**：实验/调参教练（摸底判读）
- **用户执行命令（2026-09-10，实录 25 交付版）**：
  ```powershell
  python scripts\train.py --config configs\baseline.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed 42 --out reports\base_h2_s42
  python scripts\train.py --config configs\baseline.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed 2024 --out reports\base_h2_s2024
  python scripts\train.py --config configs\baseline.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed 7 --out reports\base_h2_s7
  ```
- **输出关键数字（三份完整 stdout）**：best_epoch 12/16/15；best val MAE **4.95/4.12/5.30**（均值 4.79±0.61）；best val R² 0.9594/0.9596/0.9380；早停于 19/23/22 epochs（=best+patience7 恰好，三种子算术全闭环）；runtime 71.1/81.7/78.5s（cuda）；n_train/val/test=30000/6000/6000；逐 epoch val MAE 抖动 4.1↔13.2（H1 同款 val 噪声签名）；**`'test': None` ×3**；可选 probe/list-meters 复核未跑（不阻塞，实录 25 已定谳）。
- **判读 1（Test 冻结首次实战验证）**：三份 result 均 `test: None`——实录 14 的 eval_test 缺省翻转修复在首个新纪元摸底中实战生效，**H2 Test 预算 2 次 untouched**（摸底全程零触碰；对照 H1 时代摸底曾因缺省 True 误碰 #1）。
- **判读 2（训练健康/日志完整性法证）**：三种子中程收敛（best 12/16/15，无 epoch1 崩溃、无 30 跑满）；val R² 0.94-0.96；train MAE 终值 5.1-6.3 正常收敛。stdout 各缺一行（s42 的 Epoch 009、s2024 的 Epoch 014）——trainer 每 epoch 无条件打印 + 早停算术 best+7=last 三种子全闭环 → **系粘贴/终端层丢失而非训练缺失**，且缺失行均非 best epoch，判读不受影响。
- **判读 3（vs H1 摸底对照）**：H1 v5 摸底 val best MAE ≈3.3/3.9（实录 14）vs H2 4.1-5.3——同量级略高（H2 壶功率 2950W vs 2344W、aggOffW 基线相近，方向合理）；runtime 同量级。
- **判读 4（val 分辨率估算）**：val 段 321,855 样本（22.35 天）× on_frac 0.0133 → 全 val ON≈4,281；6000 linspace 子采样 → **val ON≈80**（H1 ≈36 的 2.2 倍，方向利好），但搜索选型仍须扩 val 30000（ON≈400）+ 多种子均值（H1 纪律直接沿用）。
- **缺口（本实录边界）**：train.py stdout 只打 val MAE/R2/score——**val F1/EE/P/R 不可见，摸底判读未完成**；EE 方向验证（drift 反向签名预判偏正）是下一步核心目的。须走 evaluate.py 补读通道（实录 14/15 建立：读 result.json+history.json 的 best_epoch_val，不重训、不碰 Test）。
- **待用户（val KPI 补读 ×3）**：
  ```powershell
  python scripts\evaluate.py --run-dir reports\base_h2_s42
  python scripts\evaluate.py --run-dir reports\base_h2_s2024
  python scripts\evaluate.py --run-dir reports\base_h2_s7
  ```
- **回传要求**：三份 JSON 全文（重点 best_epoch_val 的 val_f1 / val_energy_error / val_precision / val_recall）。
- 是否进入 REPORT.md：否（摸底进行中，val KPI 未齐）。

### 执行实录 27（2026-09-10）：House2 kettle 摸底判读完成——val F1 0.9773/EE +2.91% 方向反转坐实/全门槛过；H2 粗搜配置 tuning_h2.yaml 交付
- **本任务角色**：实验/调参教练（摸底判读 + 搜索方案设计）
- **用户执行命令（2026-09-10，实录 26 交付版）**：
  ```powershell
  python scripts\evaluate.py --run-dir reports\base_h2_s42
  python scripts\evaluate.py --run-dir reports\base_h2_s2024
  python scripts\evaluate.py --run-dir reports\base_h2_s7
  ```
- **输出关键数字（best_epoch_val，val 6000）**：val F1 **0.9829/0.9773/0.9718**（均值 0.9773±0.0056）；val P 0.9885/0.9773/0.9663（0.9774±0.0111）；val R **0.9773×3（三种子完全一致）**；val EE **+2.83%/−0.90%/+6.78%**（+2.91%±3.84%）；val MAE 4.95/4.12/5.30（与 stdout 一致）；val RMSE 71.3/71.1/88.1；val R² 0.9594/0.9596/0.9380；test 字段无（evaluate 只读 result+history，不碰 Test）。
- **判读 1（val 事件结构数字法证）**：recall 0.977273=**86/88** 三种子一致 → **val 6000 窗口中 ON 窗口=88 个**（实录 26 估算 80 的 1.1 倍，linspace 均匀+on_frac 段均值可解释）；逐种子解码全闭环——TP=86、FN=2 **固定**（2 个硬窗口疑边缘/低幅，种子不敏感）、FP=1/2/3（precision 86/87、86/88、86/89，F1=172/175、172/176、172/177 全对上）→ **F1 种子差异=纯 FP 噪声**。@val 30000 → ON≈440，判别力充分。
- **判读 2（sae 口径补记）**：val_sae=|val_energy_error|（s2024 反号坐实：sae +0.00895/EE −0.00895）。
- **判读 3（EE 方向反转=drift 反向签名 val 侧坐实）**：H2 摸底 val EE **+2.91%±3.84%** vs H1 摸底 val EE **−7.1%±3.0%**（实录 15）——方向反转成立（H2 三种子 2 正 1 负、均值正；H1 全负）。Test 侧检验留锁定后（H1 经验 val→test 同向放大，H2 预期偏正方向；此为纪元验收时的关键观察量）。
- **判读 4（摸底判读完成，全门槛过）**：F1 0.9773≥0.75、R 0.9773≥0.70、|EE| max 6.78%≤0.15——富余巨大。vs H1 摸底 val（F1 0.918/R 0.848）全面占优；结构原因：①H2 壶 3kW 档对 ~300W 基线信噪比更高（corr 0.56-0.67 vs H1 0.35-0.53）；②val ON 88 vs H1 33（事件样本多一倍以上）。MAE 4.79 略高于 H1 的 3.3-3.9（壶功率更大，方向合理）。
- **判读 5（搜索方案）**：`configs/tuning_h2.yaml` = **tuning_v5.yaml 搜索空间原样平移**（pyyaml 校验：结构逐字段一致+双锚可达——H1 锁定 F4 点（w96/d64/h8/L2/ff128/do0/bs64/lr3e-4/wd1e-4）与 H2 摸底 baseline 架构点（w128/d64/h4/do0.1/bs128/lr5e-4/wd1e-4）均在空间内）；val 30000/train 30k/gates 不变/epochs 25/pat 5/trial seed=42+i。依据：①「H1 F4 邻域平移起步」既定方针（F4 邻域=v2_do00 邻域，空间本就以其为中心）；②w96 结论不跨纪元 → 窗口维 [96,128,192] 重验；③baseline 可达点=粗搜内天然锚（区分「搜索发现」与「单 seed 运气」）。**头室管理**：F1 已 0.977（天花板 <2.3pt），搜索价值重心=EE 收敛+多种子 σ 稳健性；**对照口径警示**：H1 摸底 val=6000 口径、本搜索=30000，数字不可直接比，锚须在搜索协议下重建。
- **沙箱事故记录**：第六次平台重置（同第五次新形态：HEAD 回 7824bb4+工作区幸存）；SOP 直接适用（显式 `git fetch origin arena/...`+mixed reset），逐文件对账零丢失；/tmp 再清空，pyyaml 以 --break-system-packages 装入系统 python3（--user 被 PEP 668 拦）。
- **待用户（粗搜 32 trials，~30-60min GPU）**：
  ```powershell
  python scripts\tune.py --config configs\tuning_h2.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --out reports\tuning_h2
  ```
- **回传要求**：门槛统计（过/总数）+ Top-5 完整参数行（含 val S/F1/MAE/EE 与全部超参），同 H1 粗搜回传格式（实录 17）。
- 是否进入 REPORT.md：否（搜索未跑，摸底结论待搜索+Test 后一并沉淀）。

### 执行实录 28（2026-09-10）：House2 kettle 粗搜 32 trials 判读（上）——32/32 过门槛/Top-5 极差 0.0017 排名噪声内/lr 下行+w192 苗头/口径警示兑现；csv 全貌待补
- **本任务角色**：实验/调参教练（粗搜判读）
- **用户执行命令（2026-09-10，实录 27 交付版）**：
  ```powershell
  python scripts\tune.py --config configs\tuning_h2.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --out reports\tuning_h2
  ```
- **输出关键数字**：门槛 **32/32** 全过（按 val composite 排序，test 未评估 ✓）；Top-5：①trial 11 S=0.0149/MAE 5.1/F1 0.968/EE +0.005（w96 d64 L2 lr2e-4）②trial 18 S=0.0156/4.3/0.972/+0.018（w192 d128 L2 lr2e-4）③trial 9 S=0.0157/8.1/0.965/−0.001（w192 d64 L1 lr3e-4）④trial 30 S=0.0159/6.6/0.965/−0.003（w192 d64 L2 lr2e-4）⑤trial 31 S=0.0166/5.1/0.963/−0.004（w128 d64 L2 lr3e-4）。
- **判读 1（S 构成算术复核，三 trial 全对上）**：S=0.4·MAE/2000+0.4·(1−F1)+0.2·|EE|——trial 11=0.0010+0.0128+0.0010=0.0148、trial 18=0.0009+0.0112+0.0036=0.0157、trial 9=0.0016+0.0140+0.0002=0.0158 ✓✓✓。S 方差≈全来自 F1 项（MAE 项 0.001 级、EE 项仅 trial 18 可观）——H2 的 S 排名本质=F1 排名微调。
- **判读 2（门槛失区分度属预期）**：F1 门槛 0.75 在 H2 富余过大（摸底 baseline 即 0.977）→ 32/32 同 H1 重搜（实录 17）——守门员在强纪元失效，选型全靠 S 排序+细搜复核。
- **判读 3（方向信号）**：**lr 下行**——2e-4 占 Top-5 三席（#1/#2/#4），5e-4（H1 粗搜冠军 lr）零席；**w192 苗头**——三席（#2/#3/#4）但 top-1 仍 w96：窗口结论第三纪元再翻转（旧纪元 w128→H1 v5 w96→H2 w192 苗头），窗口维未收敛，细搜须 w96×w192 交叉；**L1 苗头**（trial 9，H1 空间从未上榜）与 **d128 苗头**（trial 18 #2，MAE 全场最低 4.3）各一席；d64 仍主导（4/5）。
- **判读 4（口径警示兑现）**：摸底 val F1 0.9773@6000（ON=88）→ 粗搜 0.963-0.972@30000（ON≈440）——难窗口增多 F1 温和下降属预期非退化；EE 摸底 +2.91%±3.84%@6000 → 粗搜 Top-5 **−0.4%~+1.8% 近零**——大 val 离散化细化+配置双效应（细搜分离）；EE 风险重心移至 test 侧（drift 反向签名：test 段 3.71 evt/day 偏轻，预期 test EE 偏正，验收时关键观察量）。
- **判读 5（赢家诅咒预警）**：Top-5 极差仅 0.0017——排名在噪声内（H1 教训：粗搜 top-1 0.0400→复核 0.0567，实录 17/18）→ 粗搜 top-1 不可直接锁定，细搜必须多候选×多种子+双锚。
- **判读 6（两个缺口，待补料）**：①**参数缺口**——Top-5 行仅含 window/d_model/layers/lr 四维，nhead/ff/dropout/bs/wd 五维不可见（在 tuning_summary.csv）；②**锚缺口**——摸底 baseline 架构点（…bs128 lr5e-4）未进 Top-5（trial 31 w128 lr3e-4≠baseline 点），其在 32 trials 中的位置待 csv 全貌。
- **待用户（补料 ×2，秒级）**：
  ```powershell
  Get-Content reports\tuning_h2\tuning_summary.csv
  Get-Content reports\tuning_h2\best_config.yaml
  ```
- **回传要求**：csv 全文（33 行量级，含表头与全部 32 trials）+ best_config.yaml 全文（trial 11 完整参数）。
- 是否进入 REPORT.md：否（粗搜判读进行中，细搜未设计）。

### 执行实录 29（2026-09-10）：House2 kettle 粗搜判读·下（csv 全貌）——S 分解 32/32 闭环/P·R 口径定谳 seq2point 中心点/w96 双峰 vs w192 EE 稳健/lr 信号修正；细搜批次 1（fine_h2 ×7 配置）交付
- **本任务角色**：实验/调参教练（粗搜全貌判读 + 细搜设计）
- **用户执行命令（2026-09-10，实录 28 交付版）**：
  ```powershell
  Get-Content reports\tuning_h2\tuning_summary.csv
  Get-Content reports\tuning_h2\best_config.yaml
  ```
- **输出关键数字**：csv 32 行全量（粘贴处 4 处行合并已重构，全部 32 trial 恢复）；best_config=trial 11 完整参数（w96 d64 h8 L2 ff256 do0.1 bs64 lr2e-4 wd1e-4，与 csv 行逐字段一致）；全员 val_r2 0.916-0.942、gate=pass 32/32。
- **判读 1（转录重构+算术全闭环）**：S 分解（0.4·MAE/2000+0.4·(1−F1)+0.2·|EE|）**32/32 一致**（±0.0006 内）；sae=|EE| 32/32；seed=42+trial 32/32——csv 转录与代码口径全部对账闭合。
- **判读 2（P/R 口径定谳——seq2point 中心点逐点）**：读 src/metrics.py+src/data.py 证据：模型预测 target[center]（y=中心点单值），P/R/F1/EE 均为 30000 采样中心的逐点口径；val_centers 随 window_size 偏移边界（linspace 起点含 w//2）→ ON 中心数 N 随窗口微变：**N=400（w96/w128）/407（w192）**（recall 分母全解码：388/400、379/400、400/407、388/407…）。推论：**F1 量化步长 ≈1/400≈0.0025**——top-5 F1 差 0.963-0.972 仅 4 个量子；判定纪律：F1 差 <0.0025 视为并列。
- **判读 3（全貌修正实录 28 两判读）**：①**lr 信号弱化**——top-5 lr2e-4×3 但 top-10 三档均衡（2e-4×3/3e-4×3/5e-4×4），「5e-4 失效」不成立，仅「top-2 均 2e-4」；②**w96 双峰性**——rank1 在 w96，但底部 6 席占 5、|EE|>2% 的 5 例全部是 w96（max +8.9%）；**w192 无底部队且 12 trial 全部 |EE|≤1.8%**（EE 稳健维）；w128 top-10 占 4 席（ranks 5/6/8/10）中庸。窗口维结论：w96 高方差（赢可夺冠、输可爆 EE）、w192 稳健、w128 居中。
- **判读 4（其余维度信号）**：**ff256** top-10 占 7（top-1 为 d64+ff256=4×d，打破 H1 ff=2×d 惯例）；**h8** top-5 全席（top-10 6/10）；**bs64** top-10 7/10；**L1 双峰**（top-10 5 席但底部 6 席占 5——快而便宜但高风险）；do0 top-10 7 席但 top-2 均 do0.1；wd 均衡无信号。
- **判读 5（baseline 锚）**：摸底 baseline 精确架构点（…bs128 lr5e-4）**未被采样**（32/1152≈2.8% 覆盖，随机缺失属预期）；最近邻 trial 26（仅 lr 3e-4≠5e-4）rank 6——baseline 架构区域有竞争力，协议锚由细搜 fb_basearch 补齐。
- **判读 6（epochs 观察）**：best_epoch≥15 仅 trial 11（20/25，lr2e-4）与 trial 14（19/25）——cap 25 仅对 trial 11 临界；维持 25/5 协议（可比性+H1 30/7 劣化教训），30-epoch 敏感性留批次 2 备选。
- **判读 7（细搜批次 1 设计，configs/fine_h2/ ×7，均 25/5 composite val30000 协议=与粗搜可比）**：
  - **f0_t11**=粗搜 top-1 精确复核（赢家诅咒检验，对照 0.0149）；**f2_t18**=top-2 精确复核（MAE/F1 双最高档）；
  - **f1_t11w192 / f3_t18w96**：完成 {t11,t18}×{w96,w192} **2×2 因子**——窗口敏感性定谳（w96 高方差 vs w192 EE 稳健的正面对决）；
  - **f4_t18d64**：容量问题（top-5 唯一 d128 点，d128 是否必需）；
  - **f5_t9**：L1 苗头+EE 最优（−0.06%）复核；
  - **fb_basearch**：摸底 baseline 架构协议归一（双锚纪律的协议锚，H1 fb_basearch 同构）。
  - 种子 8000-8002（fresh 族；粗搜 42-74）；21 runs ≈40-70min。校验：f0/f2/f5 与粗搜 csv 行逐字段一致、f1/f3/f4 单变量交叉、协议字段 7 配置统一（pyyaml 全解析）。
- **预注册判定纪律（批次 1 判读时执行）**：①summarize_fine mean±σ val_score 排序；②赢家诅咒检验：f0 均值 vs 粗搜 0.0149（劣化 >+0.003 则粗搜排名整体降权）；③top-1 与次优差 <2×合并 SEM → 分量全优/σ 小者优先；④F1 差 <0.0025（1 量子）视为并列；⑤EE 的 σ 与方向计入权衡（w192 稳健性 vs w96 高方差）；⑥锁定后 Test 预注册：fresh seed 9000 × --test 恰一次，验收 S_test≤val 均值+0.015、F1≥0.75、R≥0.70、|EE|≤0.15，观察量=EE 方向（drift 反向签名预期偏正，>+5% 须如实披露）；H2 Test 预算 #1/2。
- **待用户（细搜批次 1：21 runs + 汇总）**：
  ```powershell
  foreach ($c in f0_t11,f1_t11w192,f2_t18,f3_t18w96,f4_t18d64,f5_t9,fb_basearch) {
    foreach ($s in 8000,8001,8002) {
      python scripts\train.py --config configs\fine_h2\$c.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed $s --out reports\fine_h2\${c}_s$s
    }
  }
  python scripts\summarize_fine.py --runs-dir reports\fine_h2
  ```
- **回传要求**：summarize_fine 输出全文（按变体 mean±std 表）；个别 run 异常时补该 run 的 evaluate.py JSON。
- 是否进入 REPORT.md：否（细搜未跑）。

### 执行实录 30（2026-09-10）：细搜批次 1 命令 PowerShell 解析错误定谳——foreach 裸词列表（台账既有坑重犯）；引号版修正交付
- **本任务角色**：工程实现工程师（命令修复）
- **用户执行命令（2026-09-10，实录 29 交付版原文）**：
  ```powershell
  foreach ($c in f0_t11,f1_t11w192,f2_t18,f3_t18w96,f4_t18d64,f5_t9,fb_basearch) {  foreach ($s in 8000,8001,8002) {    python scripts\train.py --config configs\fine_h2\$c.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed $s --out reports\fine_h2\${c}_s$s  }}
  ```
- **输出**：`ParserError: 参数列表中缺少参量（MissingArgument）`，错误位置行:1 字符:22、光标在首个逗号之后（f0_t11, 处）；错误回显中路径显示为 `\x5c` 转义、尾部附着 GUID（6bd10074-…）——多行粘贴被折叠为单行+粘贴标记噪音，非根因。
- **判读（根因=裸词列表）**：foreach 的集合子句是**表达式上下文**——裸词 `f0_t11` 被按命令调用解析，随后的逗号触发 MissingArgument（光标位置=首逗号，精确吻合）；数字列表 `8000,8001,8002` 合法（数值字面量），无需引号。**这是台账既有坑（「PowerShell 裸词列表加引号」，session 首日记录）的重犯——实录 29 交付侧责任**，向用户致歉；H1 时代同构循环命令（引号版）曾顺利跑通 20+ runs。
- **修正（最小变更=仅给 7 个配置名加单引号，其余字节不变）**：
  ```powershell
  foreach ($c in 'f0_t11','f1_t11w192','f2_t18','f3_t18w96','f4_t18d64','f5_t9','fb_basearch') {
    foreach ($s in 8000,8001,8002) {
      python scripts\train.py --config configs\fine_h2\$c.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed $s --out reports\fine_h2\${c}_s$s
    }
  }
  python scripts\summarize_fine.py --runs-dir reports\fine_h2
  ```
- **验证边界（如实）**：沙箱无 pwsh，本修正未在沙箱实测；依据=台账既有坑的既验修复模式（H1 实录 18/19 引号版循环 20+ runs 实战通过）。若修正版仍报错（如粘贴标记噪音导致 command-not-found），回传报错全文再定谳。
- **待用户**：跑修正版（21 runs + summarize，~40-70min GPU，Test 冻结）→ 回传 summarize_fine 全文。
- 是否进入 REPORT.md：否（工具修复）。

### 执行实录 31（2026-09-10）：House2 kettle 细搜批次 1 判读——赢家诅咒兑现（粗搜 top-1 垫底）/t11 协议出局/L1 兑现领跑/top-3 统计并列；批次 2 交付（加种子+2 变体+补缺 run）
- **本任务角色**：实验/调参教练（细搜判读 + 批次 2 设计）
- **用户执行命令（2026-09-10，实录 30 修正引号版）**：
  ```powershell
  foreach ($c in 'f0_t11','f1_t11w192','f2_t18','f3_t18w96','f4_t18d64','f5_t9','fb_basearch') {
    foreach ($s in 8000,8001,8002) {
      python scripts\train.py --config configs\fine_h2\$c.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed $s --out reports\fine_h2\${c}_s$s
    }
  }
  python scripts\summarize_fine.py --runs-dir reports\fine_h2
  ```
- **输出关键数字（summarize_fine，mean±std，n=3 除 fb n=2）**：f5_t9 **0.0175±0.0017**（F1 0.9656±0.0007 最高最稳/MAE 7.32±1.50/EE +1.14%±0.79%/ep 8.0）；f3_t18w96 **0.0178±0.0012**（σ 全场最小/EE −0.30%±0.23% 最准/F1 0.9607）；f2_t18 **0.0179±0.0017**（MAE 6.49±2.20 最低/EE −0.27%±0.82%）；f4_t18d64 0.0215±0.0056；f1_t11w192 0.0218±0.0050；fb_basearch 0.0227±0.0046（**n=2**）；f0_t11 **0.0302±0.0040 垫底**（EE +6.42%±1.79%/**best_ep 2.3**）。扫描 20/21 个 run。
- **判读 1（赢家诅咒兑现，预注册纪律②触发）**：f0（粗搜 top-1，单 seed 53 时 0.0149）细搜 3 fresh seeds **0.0302±0.0040，劣化 +0.0153 ≫ 阈值 +0.003**——粗搜 top-1 判死刑（H1 同构：0.0400→0.0567）；且 EE +6.42% 兑现 w96 大 EE 风险、best_ep 20→2.3（seed53 系异类：同配置 fresh 族 2-3 epoch 即最优后恶化）→ **粗搜排名整体降权，细搜结果为唯一选型依据**。
- **判读 2（2×2 因子定谳）**：t11 协议：w96 (f0) 0.0302 ≪ w192 (f1) 0.0218；t18 协议：w96 (f3) 0.0178 ≈ w192 (f2) 0.0179（差 0.0001）→ 窗口敏感性取决于协议；**t11 协议本身（d64+lr2e-4 组合）双双劣于 t18 协议 → t11 协议出局**。EE 稳定性亦随协议：t18 两窗口 EE 均 ≈−0.3%（稳），t11 两窗口 +1.5%/+6.4%（不稳）。
- **判读 3（容量方向性）**：f2 (d128) 0.0179 vs f4 (d64) 0.0215：Δ0.0036 < 2×合并 SEM 0.0067 未达显著，但方向与 MAE（6.49 vs 7.57）、EE σ（0.0082 vs 0.0338）一致 → d128 方向性优势（容量↑更稳）。
- **判读 4（L1 苗头兑现）**：f5（L1，粗搜 rank-3）**0.0175 领跑** + F1 0.9656±0.0007 全场最高且最紧 + best_ep 8.0（最健康收敛节奏）+ 42-96s 最便宜——粗搜排名（top-1 垫底、rank-3 领跑）与细搜排名基本不相关，再次坐实单 seed 排名不可选型。
- **判读 5（top-3 统计并列）**：f5/f3/f2 极差 0.0004 ≪ 2×合并 SEM（≈0.0026）→ 并列；分量权衡：f5 赢 F1（最高最稳），f3 赢 EE（−0.30%±0.23% 最准）+σ（0.0012 最小），f2 赢 MAE（6.49 最低但 σ 2.2 大，单种子盆地嫌疑）。**test EE 偏正预期**（drift 反向签名）下 f3/f2 的 val EE 近零余量更大——此权衡留批次 2 判读定夺。
- **判读 6（协议锚）**：fb 0.0227（n=2）——top-3 全部优于锚（调参价值二次坐实），但 f0 劣于锚（赢家诅咒的极端形态：粗搜 top-1 竟不如不调参）。
- **缺口（fb_basearch n=2）**：20/21 run——fb_basearch 缺 1 个种子，须查目录补跑（不影响 top-3 判读，影响锚的 σ 精度）。
- **批次 2 设计（13 runs ≈25-45min）**：①top-3 加种子 8003/8004（n=3→5，SEM 缩小）；②f6_f5d128（L1 领头羊 × d128 容量方向——两条独立方向信号的杂交检验）；③f7_f5w96（t9×w96——完成 {t9,t18}×{w96,w192} 窗口全因子）；④fb_basearch 补缺 run。配置已生成并校验（f6 仅 d 64→128、f7 仅 w 192→96，协议字段不变）。
- **预注册批次 2 判定树**：①若 f6 显著优于 min(f5,f3,f2)（>2×合并 SEM）→ 锁定候选=f6（必要时微批次确认）；②否则 top-3（n=5）均值排序，分量权衡锁定（倾向规则：EE 近零+σ 小优先于 F1 最高——test EE 偏正预期的防御性选择；MAE σ>2 的盆地不采信）；③锁定后 Test 预注册：fresh seed 9000 × --test 恰一次，验收 S_test≤val 均值+0.015、F1≥0.75、R≥0.70、|EE|≤0.15，EE 方向（预期偏正，>+5% 如实披露）；H2 Test 预算 #1/2。
- **待用户（先查缺哪个 run，再批次 2）**：
  ```powershell
  Get-ChildItem reports\fine_h2 | Select-Object Name
  # 若缺 fb_basearch_s800X（按实际补）：
  python scripts\train.py --config configs\fine_h2\fb_basearch.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed 8000 --out reports\fine_h2\fb_basearch_s8000
  # 批次 2（引号版）：
  foreach ($c in 'f5_t9','f3_t18w96','f2_t18') {
    foreach ($s in 8003,8004) {
      python scripts\train.py --config configs\fine_h2\$c.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed $s --out reports\fine_h2\${c}_s$s
    }
  }
  foreach ($c in 'f6_f5d128','f7_f5w96') {
    foreach ($s in 8000,8001,8002) {
      python scripts\train.py --config configs\fine_h2\$c.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed $s --out reports\fine_h2\${c}_s$s
    }
  }
  python scripts\summarize_fine.py --runs-dir reports\fine_h2
  ```
- **回传要求**：目录清单（确定缺哪个 run）+ 批次 2 后的 summarize_fine 全文（f5/f3/f2 将以 n=5 聚合）。
- 是否进入 REPORT.md：否（细搜进行中）。

### 执行实录 32（2026-09-10）：House2 kettle 细搜批次 2 判读 → **锁定 f5_t9**；Test 预注册执行交付（seed 9000，H2 预算 #1/2）
- **本任务角色**：实验/调参教练（批次 2 判读 + 锁定 + Test 预注册执行）
- **用户执行命令（2026-09-10，实录 31 交付版引号循环）**：
  ```powershell
  foreach ($c in 'f5_t9','f3_t18w96','f2_t18') { foreach ($s in 8003,8004) {
      python scripts\train.py --config configs\fine_h2\$c.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed $s --out reports\fine_h2\${c}_s$s } }
  foreach ($c in 'f6_f5d128','f7_f5w96') { foreach ($s in 8000,8001,8002) {
      python scripts\train.py --config configs\fine_h2\$c.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed $s --out reports\fine_h2\${c}_s$s } }
  python scripts\summarize_fine.py --runs-dir reports\fine_h2
  ```
- **输出关键数字（32 run 目录）**：top-3 n=5：f5 **0.0167±0.0016**（F1 0.9654±0.0005/EE +0.66%±0.87%/R 0.9533±0.0000/ep 7.2）；f2 0.0180±0.0020；f3 0.0181±0.0011（EE −0.40%±0.64%/MAE 6.42 最低）；f6_f5d128 0.0185±0.0017（EE −0.98%±0.05 全场最紧）；f7_f5w96 0.0226±0.0009；fb 仍 n=2；f0 0.0302 垫底不变。
- **判读 1（判定树①不触发）**：f6（0.0185）不优于 f5（0.0167），容量方向对 L1 线关闭；f6 的 EE −0.98%±0.05（σ 全场最小）记录为未来纪元线索（EE 敏感场景的 L1×d128 方向），本纪元 S 更差不锁。
- **判读 2（top-3 n=5 S 全不显著）**：f5 vs f3 Δ0.0014 < 2×SEM 0.0017；f5 vs f2 Δ0.0013 < 0.0023；f2 vs f3 Δ0.0001——S 层面三分量并列，进分量权衡。
- **判读 3（分量显著性定谳，锁定依据）**：f5 vs f3——**F1 +0.0049（2 量子）=6.2×SEM 显著**；**recall +0.0173（f5 σ=0.0000，5 种子确定性）决定性**；EE gap 0.0106=2.2×SEM 仅边缘性；MAE（f3 优 0.48）与 S σ 比（F=2.12<6.4）不显著。倾向规则（EE 近零+σ 小优先于 F1 最高）的**触发前提未成立**：该规则预设「F1 冠军 EE 坏」的取舍（如 f0 +6.4%），而 f5 的 EE +0.66%±0.87% 本身近零（与 f3 差异仅边缘性，两者均 ≪15% 门槛）→ f5 无 EE 牺牲，兼得 F1 最高+EE 近零 → **判定性锁定 f5_t9**。
- **判读 4（窗口因子收官）**：f7（t9×w96）0.0226 vs f5（t9×w192）0.0167：Δ0.0059 > 2×SEM 0.0018 显著——t9 协议强依赖 w192；t18 协议窗口不敏感（0.0178≈0.0179）→ **窗口敏感性是协议依赖的**（批 1 结论加固）；粗搜 rank-1 是 w96 的再次反转，细搜定谳 w192。
- **判读 5（搜索收敛，不开批次 3）**：批 2 两新变体均不优于 f5，top-3 排序跨批次稳定（f5 0.0175→0.0167 / f2 0.0179→0.0180 / f3 0.0178→0.0181），方向信号全部探明（窗口/容量/L 数/d128）无剩余增益空间 → 搜索收敛。
- **判读 6（小偏差如实记录）**：fb_basearch 补缺 run 未执行（仍 n=2；目录清单未随贴，仅扫描计数 32）——锚角色 n=2 已足（top-3 全体 0.0167-0.0181 < 锚 0.0227 结论不变），不回溯追补。
- **锁定宣告**：**House2 kettle 纪元最终配置 = f5_t9**（configs/fine_h2/f5_t9.yaml：w192 / d64 / h8 / **L1** / ff128 / do0.0 / bs128 / lr3e-4 / wd1e-4，25/5 composite val30000）；val S 0.0167±0.0016（n=5）、F1 0.9654±0.0005、EE +0.66%±0.87%、R 0.9533、MAE 6.90±1.26、best_ep 7.2。vs 协议锚 fb 0.0227：调参收益 0.0060（26%）。
- **Test 预注册执行（H2 触碰 #1/2）**：fresh seed 9000 × `--test` 恰一次；验收四线：**S_test≤0.0317**（=0.0167+0.015）、**F1≥0.75**、**R≥0.70**、**|EE|≤0.15**；观察量=EE 方向（drift 反向签名预期偏正，>+5% 如实披露）。Test 通过→REPORT.md H2 章节+收尾；不过→预算剩 1 次重新锁定（候选=f3）。
- **待用户（Test 一跑，~2-4min）**：
  ```powershell
  python scripts\train.py --config configs\fine_h2\f5_t9.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed 9000 --test --out reports\final_h2_f5_s9000
  ```
- **回传要求**：完整 stdout（逐 epoch 行+末尾 result JSON）。
- 是否进入 REPORT.md：待 Test 结果（通过后 H2 章节随收尾一并写入）。

### 执行实录 33（2026-09-10）：House2 kettle Test 终局验收——**四线全过，纪元收官**（S_test 0.0252/F1 0.9398/EE −0.02% 死零）；EE 方向预判未兑现（如实披露）
- **本任务角色**：实验/调参教练（Test 终局判读）→ 收官（回归默认角色）
- **用户执行命令（2026-09-10，实录 32 预注册交付版）**：
  ```powershell
  python scripts\train.py --config configs\fine_h2\f5_t9.yaml --data-path D:\Work\testPython\datasets\ukdale_h2_kettle.npz --seed 9000 --test --out reports\final_h2_f5_s9000
  ```
- **输出关键数字**：best_epoch 5（val score 0.0182，在 f5 家族 0.0167±0.0016 内 ✓）、早停于 10（=5+patience5 算术闭环 ✓）、runtime 48.4s；**test：MAE 5.18 / RMSE 80.52 / R² 0.8980 / EE −0.0199% / P 0.975 / R 0.90698 / F1 0.93976**；n_train/val/test=30000/30000/6000、eval_test: true（--test 显式触发，H2 触碰 #1/2）。
- **判读 1（四线验收，预注册口径）**：S_test = 0.4×5.18/2000+0.4×(1−0.93976)+0.2×0.0002 = **0.0252 ≤ 0.0317** ✅（富余 0.0065）；F1 **0.9398 ≥ 0.75** ✅；R **0.9070 ≥ 0.70** ✅；|EE| **0.02% ≤ 15%** ✅（富余 750 倍）；漂移检验 Δ=0.0252−0.0167=**+0.0085 < 0.015** ✅——**四线全过+无漂移，House2 kettle 纪元验收通过**。
- **判读 2（test 事件解码全闭环）**：R 0.90698=**39/43**、P 0.975=**39/40** → TP=39、FN=4、FP=1、test ON=43（6000 采样窗口 on_frac 0.72%，与 diagnose test 段 0.64% 吻合）；F1=78/83=0.93976 ✓ 三指标同构解码一字不差。
- **判读 3（EE 方向预判未兑现——如实披露）**：drift 反向签名（val 7.78 evt/day 重 vs test 3.71 轻）曾预判 test EE 偏正——实测 **−0.02% 死零，预判未兑现**。漂移的代价出现在 **recall**（val 0.9533→test 0.9070，多漏 4 个 ON 窗）而非能量（FN 的漏报能量与 FP 的多报能量相抵）→ 「val/test 事件率差→EE 方向」的推理链在 f5 上不成立（H1 的 val→test 同向放大也不普适）——**两纪元教训：EE 方向不可由 drift 签名预判，只能 Test 实测**。
- **判读 4（两纪元终局对照，进 REPORT.md §7）**：H2 (f5_t9) vs H1 (F4)——S_test 0.0252 vs 0.0541（好 2.1 倍）、F1 0.9398 vs 0.8941、EE −0.02% vs +5.2%、MAE 5.18 vs 6.42、R² 0.898 vs 0.779、R 0.9070 vs 0.9268（唯一 H1 略优项）；结构性原因：H2 壶 3kW 档对 ~300W 基线信噪比更高（corr 0.56-0.67 vs 0.35-0.53）。锁定配置跨纪元差异：L1 vs L2、w192 vs w96、bs128 vs bs64——**架构结论零重叠，跨纪元不迁移定谳**。
- **判读 5（epoch 7 val 毛刺）**：val MAE 21.96/score 0.0992 单点尖峰——已知 val 噪声签名（实录 26 同款），未影响 best_epoch 选择（早停窗口内自愈）。
- **Test 预算审计（H2 纪元闭合）**：预算 2 次，实际触碰 **1 次**（本次终局验收；摸底/粗搜/细搜全周期 test:None 冻结实战）——1 次未用封存，纪元闭合。对照 H1 纪元：摸底误碰 1 次+终局 1 次（2/2 用尽）——eval_test 缺省翻转修复的完整价值兑现。
- **收官动作**：REPORT.md v1.1（新增 §7 House2 纪元+两纪元对照+跨纪元结论，§6 版本历史追加）；TUNING_GUIDE §3 战史补章（第三纪元）+§4 踩坑清单追加；STATUS 收官更新；session 纪要追加；commit/push。
- 是否进入 REPORT.md：**是**（验收通过触发，v1.1）。

### 执行实录 34（2026-09-10）：House1 dish_washer 纪元立项——事件阈值敏感性设计（前置条件）+ 预注册判读框架
- **本任务角色**：实验/调参教练（新纪元设计）
- **立项依据**：用户选定方向 ①（H2 kettle 纪元已收官，实录 33）；dw 数据已制备并过身份链（实录 23：`ukdale_dw.npz`，mains=meter1/dw=meter6，n=10,685,551/742.1 天，aggOffW 317-406、corr 0.35-0.41 身份链过）——**但纪元锁定悬置于事件口径定夺**（实录 23 预警：20W 阈值把洗涤周期多相位切碎，0.16 kWh/evt vs 典型 1-1.5 kWh/周期）。
- **口径问题本质（立项分析）**：阈值只影响 F1/P/R 与事件统计，**EE 与阈值无关**（能量=功率求和）；prepare/npz 无需重跑。三个层面：
  1. **事件统计**（diagnose evt/day·kWh/evt）：20W 已知切碎（泵相位间隙=多次 rising edge）；
  2. **F1 口径=「运行中」的业务定义**：20W=全周期（含泵 120W med）vs 200W=加热相位（p95 2363W）vs 500W=强加热——口径选择=业务纯度 vs 可学性权衡（泵相位 120W 对 ~350W agg 基线对比度极弱，corr 0.35-0.41 已示 dw 信号占比小）；
  3. **分辨率**：on_frac@20W=0.0222（val 30000→ON≈667）；阈值升高 ON 数下降，F1 量子变粗——须守住 ON≥90（val 30000）下限。
- **预注册判读框架（阈值 20/100/200/500 四跑对照）**：①事件收敛度=evt/day 向真实周期数量级（家用 ~0.5-2 次/天）收敛 + kWh/evt 向周期能量靠拢的程度；②ON 功率分布=med（泵）与 p95（加热）在各阈值的留存（双峰分离点）；③可学性代理=on_frac 与 ON med 对 aggOffW 基线的对比度；④分辨率=val ON 数 ≥90。决策规则：200W 若 evt/day∈[0.5,3] 且 kWh/evt≥0.3 且 ON≥90 → 倾向 200W（加热相位=可检测签名）；100W 与 200W 无差 → 取 100W（更近全周期）；20W 仅作切碎对照、500W 作下界对照。最终口径由数据+业务含义判读定夺（判读时给推荐+理由，用户可否决）。
- **后续设计备忘（摸底时再定）**：dw 周期 1-2h=600-1200 样本 ≫ 窗口（w96-192 仅 9.6-19.2 min）——seq2point 只见相位片段，周期级上下文不可见；粗搜空间窗口维上限考虑放宽（384/512），以摸底结果定。dw 纪元 Test 预算独立 2 次（规则既定）。
- **待用户（阈值敏感性 ×4，秒-分钟级）**：
  ```powershell
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_dw.npz --appliance dish_washer --on-threshold 20
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_dw.npz --appliance dish_washer --on-threshold 100
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_dw.npz --appliance dish_washer --on-threshold 200
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_dw.npz --appliance dish_washer --on-threshold 500
  ```
- **回传要求**：四份完整输出（含表头行——阈值与来源行是口径留痕）。
- 是否进入 REPORT.md：否（口径未定，纪元未锁）。

### 执行实录 35（2026-09-10）：House1 dish_washer 阈值敏感性判读——**口径定夺 200W（加热相位定义），纪元锁定**；摸底交付（val 30000 统一）
- **本任务角色**：实验/调参教练（口径判读 + 摸底设计）
- **用户执行命令（2026-09-10，实录 34 交付版）**：
  ```powershell
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_dw.npz --appliance dish_washer --on-threshold 20
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_dw.npz --appliance dish_washer --on-threshold 100
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_dw.npz --appliance dish_washer --on-threshold 200
  python scripts\diagnose_split.py --npz D:\Work\testPython\datasets\ukdale_dw.npz --appliance dish_washer --on-threshold 500
  ```
- **输出关键数字（evt/day 与 kWh/evt，train/val/test）**：20W：2.53/2.17/2.70，0.160 kWh/evt（切碎，实录 23 预警确认）；100W：2.04/1.46/1.81，0.198；**200W：0.98/0.91/1.02，0.411/0.345/0.404**；500W：0.90/0.76/0.90，0.451/0.409/0.461。ON 分布：@20/100W 双峰（med 120-122/p95 2363）；@200/500W 单峰紧致（med 2330-2337/p95 2383-2387）。on_frac：@20/100W 0.019-0.024；@200/500W **三段四位小数完全一致**（0.0061/0.0045/0.0062）。总能量/kWh/day/corr/aggOffW 四跑完全不变（EE 阈值无关的实证）。
- **判读 1（四判据执行，实录 34 预注册框架）**：①事件收敛度——大断层在 **100→200W**（evt/day 减半 2.04→0.98、medW 122→2335 跳变=泵相位出局），200→500 仅再并 ~10%；②双峰分离点在 100-200W 之间——@200W 起为纯加热签名（2330-2390W 紧致单峰）；③可学性——泵相位 120W **低于总负荷基线本身**（aggOffW 321-408W）→ 从 aggregate 不可学；加热 2335W 对 ~350W 基线强对比 ✓；④分辨率——val ON@30000 = 0.0045×30000 = **135 ≥ 90** ✓（@6000 仅 27）。
- **判读 2（预注册规则命中）**：200W：evt/day 0.91-1.02 ∈[0.5,3] ✓、kWh/evt 0.345-0.411 ≥0.3 ✓、ON 135 ≥90 ✓ → **倾向 200W 三条件全中**；100W 与 200W 差异显著（evt 2.04 vs 0.98、on_frac 3.5×）→ 不取 100W。
- **判读 3（200 vs 500 口径等价，取 200）**：on_frac 三段四位小数一致 → [200,500) 带样本 <1e-4；F1/P/R 为逐样本口径，事件数差异（511 vs 466）只影响 diagnose 统计不影响模型评估 → 两阈值对训练与评估**几乎完全等价**，取 200W（边缘多捕获、阈值落点对噪声更稳健）。
- **判读 4（周期结构算术闭环）**：dw 总能量 0.405 kWh/day ÷ 典型 0.8-1 kWh/周期 = 真实周期 **0.4-0.5 次/天**；×每周期 2 个加热相位 = 0.8-1.0 相位/天，与 @200W 实测 0.91-1.02 **精确吻合** → @200W 事件=加热相位（非整周期）；单相位能量交叉验证 2333W×8.9min=0.347 kWh ✓（=kWh/evt 0.345-0.411 的构成）；加热相位占 dw 总能量 **84%**（14.2W/16.9W 平均功率）→ 200W=能量主体相位，F1 口径与 EE 语义对齐。
- **判读 5（口径业务含义定谳）**：「dw 运行中」在本纪元=**加热相位中**（非全周期）：泵相位不可学（判读 1③）且占 @20W ON 样本的 ~72%（on_frac 0.0222→0.0061）→ @20W 是可学性陷阱（recall 理论天花板 ~0.28）；真周期级事件须推理侧 min-gap 后处理合并（未来工作备忘，不影响本纪元训练与验收）。
- **判读 6（drift 签名记录）**：test 偏重（evt 1.12×/kWh 1.32× vs val），H1 kettle 同向——按跨纪元结论 #4（EE 方向不可预判）仅记录不预测，Test 时观察。
- **纪元锁定宣告**：**House1 dish_washer 纪元锁定**——npz=`ukdale_dw.npz`（实录 23 身份链过+本实录口径定夺），**F1 口径 on_threshold=200W（加热相位定义）**（npz/data_spec 不变，口径在训练 config metrics 字段）；Test 预算 2 次 untouched。
- **摸底设计（configs/baseline_dw.yaml）**：架构/训练=baseline.yaml 同款（d64/h4/L2/ff128/do0.1/bs128/lr5e-4/30/7）；两处纪元适配：①threshold 200；②**val 30000**（dw val ON@6000 仅 27 无分辨率；与搜索口径统一，消除 H2 时代的摸底-搜索口径不可直比警示）；w128 起步（加热相位平均 8.9min < 12.8min 窗口，单相位可见；周期级长窗 384/512 留粗搜）。
- **待用户（摸底 ×3 + val KPI 补读 ×3，一次回传）**：
  ```powershell
  python scripts\train.py --config configs\baseline_dw.yaml --data-path D:\Work\testPython\datasets\ukdale_dw.npz --seed 42 --out reports\base_dw_s42
  python scripts\train.py --config configs\baseline_dw.yaml --data-path D:\Work\testPython\datasets\ukdale_dw.npz --seed 2024 --out reports\base_dw_s2024
  python scripts\train.py --config configs\baseline_dw.yaml --data-path D:\Work\testPython\datasets\ukdale_dw.npz --seed 7 --out reports\base_dw_s7
  python scripts\evaluate.py --run-dir reports\base_dw_s42
  python scripts\evaluate.py --run-dir reports\base_dw_s2024
  python scripts\evaluate.py --run-dir reports\base_dw_s7
  ```
- **回传要求**：三份完整 stdout + 三份 evaluate JSON（本轮直接并回，省一轮往返）。
- 是否进入 REPORT.md：否（口径判读+摸底交付，实验结论待摸底）。
