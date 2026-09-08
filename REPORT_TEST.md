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
- 是否进入 REPORT.md：否（方案与改造本身不是实验结论；待真实 KPI 出现后另行判定）
