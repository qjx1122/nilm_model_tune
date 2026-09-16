# ONBOARDING.md — 新电器接入操作手册（v1，2026-09-16）

> 对应代码基线：REPORT.md v1.4（五纪元 + 外部验证 Stage A 收官）。本手册回答三个问题：
> **①要准备什么数据 ②怎么训练/怎么推理 ③怎么判断能上生产**。
> 全部命令已对照仓库脚本核实（prepare_ukdale / diagnose_split / train / evaluate / tune / infer）；
> 过程判读史见 REPORT_TEST.md 实录 1-58，方法论人话版见 TUNING_GUIDE.md。

---

## 0. 全景：七阶段管道与现有资产

```
数据准备 → 普查&viability → 口径定夺 → 摸底(baseline) → 平移探针(优先)/细搜(备选)
       → Test 预注册验收(预算 2 次) → 推理部署(infer.py) + 漂移监控
```

| 现成可用配方 | 电器 | 验证范围 | Test 成绩 | 外部验证 |
| --- | --- | --- | --- | --- |
| `configs/fine_h2/f5_t9.yaml` | kettle | H1 家族链+H2+H5 | S 0.0252 / F1 0.940 / EE −0.02% | H5 四线全过 ✅ |
| `configs/fine_v5/f4_v2do00_w96.yaml` | kettle | H1 | S 0.0541 / F1 0.894 / EE +5.2% | （由 f5_t9 接棒） |
| `configs/probe_dw_d1.yaml`（源 d1_t11d64） | dish_washer | H1+H2 | 0.0349(+1.33%) / 0.0160(−3.42%) | 相对优势 3/3，绝对 S 线爆 ⚠️ |
| `configs/fine_mw/m1_do01.yaml` | microwave | H2 | S 0.0554 / F1 0.933 / EE +14.15% | 未验证 |

架构：Transformer encoder **seq2point**（窗口→中心点功率），d_model 64 / 1-2 层 / ~0.1M 参数；
输入单通道总表功率（**6 秒网格**），输出目标电器逐点功率（W）+ 事件流（阈值判 ON）。

**核心纪律（跨纪元十四条之最常用）**：判读权威=evaluate/test JSON；赢家诅咒单 seed 不可信；
EE 方向只能 Test 实测；Test 冻结是代码默认（`--test` 显式开关）。

---

## 1. 阶段 0：数据准备

### 1.1 最少数据项

| 数据项 | 要求 | 说明 |
| --- | --- | --- |
| 总表功率 `aggregate` | **必需**，W，数值 | 训练输入。apparent（UK-DALE mains 同款）或 active 均可，但**部署端口径须与训练一致**（口径混用=未验证漂移点） |
| 目标电器子表功率 `target` | **训练必需**，W | 监督信号；部署推理时不需要（见 §7） |
| 时间戳 | **必需**（制备阶段） | 用于 6s 网格重采样与对齐；成品 npz 不带时间戳（等效连续流） |
| 采样间隔 | **≤1min，理想 6-30s** | 事件型电器（壶 2-4min）在 15min AMI 冻结数据上不可见 |
| 记录长度 | **≥90 个 val 段 ON 事件**（viability 红线） | 对应约 3 个月级记录（事件率 ~1-3/day 时）；短记录案例：36.5d→val ON 66 → 排除 |
| 拆分 | 70/15/15 按时间顺序 | train/val/test；窗口不跨段边界（代码保证） |

### 1.2 质量红线（先于一切建模）

1. **混表/共享子表**：子表同时测多个电器（如 REDD 风格 kettle+radio）→ target 不纯，P/R/F1/EE 语义失效 → **整表不可用**（H4 整 house 排除教训）。接入前必查子表标签与共享情况。
2. **表底常驻基线**：子表长期 ~40W 底噪 → kWh/day 几乎全来自基线非事件，事件口径失真（H5 mw 教训）。
3. **缺口密度**：短缺口（≤30min）ffill 可救；长缺口整段剔除后拼接（接缝计数留痕）。缺口过密→有效数据不足。
4. **负功率/异常值**：制备时 clip 到 0 并计数留痕。

### 1.3 制备命令（UK-DALE 源，NILMTK h5）

```powershell
# ① 元数据留痕（表清单/功率类型）
python scripts\parse_nilmtk_metadata.py --h5-path D:\datasets\ukdale.h5 --house 5
# ② 侦察单表覆盖率/采样质量（可选但建议）
python scripts\probe_meter.py --h5-path D:\datasets\ukdale.h5 --house 5 --meter 18
# ③ 列出 house 内全部表（核对表号与功率类型）
python scripts\prepare_ukdale.py --h5-path D:\datasets\ukdale.h5 --house 5 --list-meters
# ④ 制备（示例：H5 kettle=meter 18；mains 相加=--mains-ids 1）
python scripts\prepare_ukdale.py --h5-path D:\datasets\ukdale.h5 --house 5 `
  --mains-ids 1 --appliance kettle --appliance-meter-id 18 `
  --out D:\datasets\ukdale_h5_kettle.npz
```

产物：`*.npz`（`{aggregate, target}` float32 一维等长）+ 同名 `*.data_spec.json`（口径元数据：表号/功率类型/缺口策略/样本数——**所有 KPI 的口径依据，须随数据归档**）。

制备语义（schema v5）：各表先 resample 到统一 6s 网格（bin 均值）再对齐；mains 多表相加；
短缺口整段 ffill（mains ≤30min 默认，电器可调 `--appliance-gap-min`）；长缺口整段剔除后全量拼接。

### 1.4 非 UK-DALE 源（自制 npz 规格）

直接构造 npz：`np.savez(out, aggregate=agg_float32, target=tgt_float32)`（一维等长，W，已重采样到 6s 等步长网格）。
**务必同步手写 data_spec.json**（源/表号/功率类型/步长/缺口策略），否则口径不可追溯。
后续 Stage B 的 `prepare_redd.py` / `prepare_refit.py` 即按此规格输出（8s→6s 须重采样，美国 120V 阈值须重定夺）。

---

## 2. 阶段 1：普查与 viability（一次 diagnose 全出）

```powershell
python scripts\diagnose_split.py --npz D:\datasets\ukdale_h5_kettle.npz --appliance kettle --on-threshold 500
```

输出 train/val/test 三段的：ON 事件数、ON 占比、事件功率中位数（medW）、kWh/evt、与总表相关性 corr。

**判读四步**（口径定夺 = 阈值扫描，dw 案例：20/50/100/200/300/500/1000 各跑一次）：

| 判据 | 方法 | 通过标准（案例） |
| --- | --- | --- |
| ①viability | val ON 事件数 | **≥90**（<90 → F1 无分辨率，诚实排除留档） |
| ②阈值断层 | 事件数-threshold 曲线 | 断崖处上方平台（dw：thr20→200 断崖 235→96，200/500/1000 平台） |
| ③机型指纹 | medW@阈值上 分布 | 紧单峰=单一机型（dw 1661W）；多峰=混机型须分训或注明 |
| ④能量闭环 | kWh/evt ≈ medW×时长×相位数 | 量级对得上（dw 0.36-0.44 ≈ 1661W×13min×2-3 相位） |

定夺的 `on_threshold_watts` 写进后续所有配置（壶 500 / dw·mw 200 的来源即此）。
**阈值不跨电压制度照抄**（美国 120V 壶 ~1500W）。

---

## 3. 阶段 2：摸底（baseline × 3 seeds）

### 3.1 配置模板（逐字段）

```yaml
seed: 42                    # 会被 --seed 覆盖
device: auto                # auto=cuda 优先
data:
  appliance: kettle         # 仅用于日志标识
  window_size: 128          # 窗口（点）＝时间上下文：128×6s=12.8min；壶 96/192 均有胜绩
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  max_samples_train: 30000  # train 子采样上限（linspace 确定性抽中心点）
  max_samples_val: 30000    # val 全量上限（选型依据，不给免费精度）
  max_samples_test: 6000    # test 子采样上限
  # eval_test 不写=缺省 False（Test 冻结）
model:                      # NILMTransformer 构造参数
  input_dim: 1
  d_model: 64
  nhead: 4
  num_layers: 2
  dim_feedforward: 128
  dropout: 0.1
training:
  batch_size: 128
  epochs: 30
  lr: 0.0005
  weight_decay: 0.0001
  patience: 7
  grad_clip: 1.0
  loss: mse
metrics:
  on_threshold_watts: 500.0 # ← 阶段 1 定夺值
```

### 3.2 训练与评估命令

```powershell
python scripts\train.py --config configs\baseline_h5k.yaml --data-path D:\datasets\ukdale_h5_kettle.npz --seed 42 --out reports\h5k_base_s42
python scripts\train.py --config configs\baseline_h5k.yaml --data-path D:\datasets\ukdale_h5_kettle.npz --seed 2024 --out reports\h5k_base_s2024
python scripts\train.py --config configs\baseline_h5k.yaml --data-path D:\datasets\ukdale_h5_kettle.npz --seed 7 --out reports\h5k_base_s7
foreach ($d in "h5k_base_s42","h5k_base_s2024","h5k_base_s7") { python scripts\evaluate.py --run-dir reports\$d }
```

run 目录产物：`best.pt`（best val epoch 权重）/ `history.json`（逐 epoch）/ `result.json`（配置+指标）/ `train.log`。
控制台两级留痕：运行日志 + `logs\console_all.log` 总日志（整份回传即可）。

### 3.3 判读要点

- **选型指标**：baseline 用 val MAE；细搜配置用 composite（S=0.4·MAE/2000+0.4·(1−F1)+0.2·|EE|）——两类配置的 val score 不可直接比。
- **|EE| 硬门槛=出线条件**（与 S 并列）：任一 seed |EE|>15% 即出局（H5 kettle baseline 三 seed 17.61/21.03/15.31% 全超线出局）。
- 记录三 seed 的 S/F1/EE 带，作为后续配对比较的基线。

---

## 4. 阶段 3：平移探针（同名配方优先，多数情况到此为止）

**迁移三级边界**：跨纪元不迁移 / 同 house 跨电器可迁移 / **同名+口径相似+同协议 → 配方逐字平移**（第三级已在 kettle、dw 上三例验证）。

```powershell
# kettle 新 house：直接用 H2/H1 冠军配方（仅换 --data-path）
python scripts\train.py --config configs\fine_h2\f5_t9.yaml --data-path <新npz> --seed 42 --out reports\<new>_f5_s42
# dw 新 house：
python scripts\train.py --config configs\probe_dw_d1.yaml --data-path <新npz> --seed 42 --out reports\<new>_d1_s42
```

**判读（配对不对称比较铁律：同 seed 逐对相减）**：

- 平移 vs baseline，seeds ≥3 配对：ΔS 方向 **3/3 一致 且 |mean ΔS| > 2×SEM** → 显著优；
- 显著优 → 补 seed 至 n=5 锁定档案；打平 → 若有多配方对决，按预注册字典序 **S → F1（量子 1/160 级）→ |EE| → MAE** 裁决；
- 平移全败 → 进入阶段 4 细搜。

---

## 5. 阶段 4：细搜（仅平移失败时）

```powershell
# 粗搜：Optuna 网格（搜索空间在 configs/tuning_*.yaml 定义；eval_test 缺省 False 不碰 Test）
python scripts\tune.py --config configs\tuning_v5.yaml --data-path <新npz> --out reports\tune_new --trials 32
# 细搜：手工批次 configs/fine_<tag>/f0..f7（每配置 3 seeds 配对，沿用 §3 命令模式）
python scripts\summarize_fine.py --runs reports\fine_new_*   # 批次汇总
```

经验先验（五纪元沉淀）：细搜后 **dropout 0 家族**在强信号电器占优（稀疏弱信号 mw 例外 do 0.01）；
窗口 96-192；lr 2e-4~3e-4；composite 选型。

---

## 6. 阶段 5：Test 预注册与验收（每 (dataset,house,appliance) 预算 2 次）

```powershell
# 锁定配方 + fresh seed（从未用于选型的号段）+ --test 显式开关
python scripts\train.py --config configs\fine_h2\f5_t9.yaml --data-path <新npz> --seed 23005 --out reports\<new>_f5_t23005 --test
```

**四线验收**（判读权威=stdout/result.json 的 `test` 块；evaluate.py 不含 test）：

| 线 | 标准 | 备注 |
| --- | --- | --- |
| F1 | ≥ 0.75 | 业务线 |
| Recall | ≥ 0.70 | 业务线 |
| \|EE\| | ≤ 0.15 | 业务线；稀疏电器注意 EE 分母效应（15%≈绝对 kWh 差很小） |
| S_test | ≤ val(n=5 均值)+0.015 | **绝对线仅同分布内有效**——跨 house 场景以业务三线+相对优势为主判据（发现 #4：dw 二连爆即段级漂移击穿此线） |

**失败复盘规则**：#1 失败 → 同配置 fresh seed 二次（不换配置；配置修正仅限数据/口径缺陷且须明示）；二连爆 → 接受失败档案或口径协商（用户决策）。val 优 ≠ test 优（H5 kettle 案例：val 0.0692→test 0.1086，val 0.0916→test 0.0890）——**这就是预算 2 次的设计依据**。

---

## 7. 阶段 6：推理与部署（`scripts/infer.py`，2026-09-16 新增）

### 7.1 场景 A：验证/自查（在训练 npz 上）

```powershell
python scripts\infer.py --run-dir reports\h5k_f5_t23007 --npz D:\datasets\ukdale_h5_kettle.npz --out reports\infer_demo.npz
```

归一化统计量自动取自该 npz 的 train 段（与训练完全一致）；若 npz 含 target 会打印全序列粗对照
（MAE/F1/P/R/EE；**正式验收仍以 `--test` 的 test 段四线为准**，此为全序列含 train 段的粗对照）。

### 7.2 场景 B：新数据/新 house 推理（生产形态）

```powershell
# --stats-from 必须传「训练时的 npz」：x/y 归一化统计量只能来自训练数据，跨数据不可重算
python scripts\infer.py --run-dir reports\h5k_f5_t23007 --npz D:\datasets\new_house.npz `
  --stats-from D:\datasets\ukdale_h5_kettle.npz --out reports\infer_new.npz
```

输出（.npz 或 .csv 按扩展名）：`index / aggregate / pred / pred_on`（有 target 时附 `target/target_on`）。
控制台摘要：ON 占比、连续 ON 段（事件数）、估算 kWh。

### 7.3 部署要点

- **输入**：单通道总表功率序列，6s 等步长（其他步长先重采样）；窗口时间语义 96-192 点=9.6-19.2min 不可压缩。
- **输出后处理**：pred 为逐点回归值可为负（与训练评估同口径）；产品侧可 `max(pred,0)`；事件=连续 `pred≥阈值` 段；能耗=`sum(pred)×6s`。
- **边界对齐**：首尾各 `window//2` 点无预测（seq2point 中心点语义）。
- **边缘可行**：~0.1M 参数、单通道输入、逐窗口一次前向 → 网关/树莓派实时推理成立；训练分钟级 → per-site 现场重训可行。
- **多电器**：每电器独立 run 目录，各跑一次 infer，输出叠加即分项账单（注意：各电器独立拆分，**分项之和 vs 总表一致性未做约束**）。
- **漂移监控**（dw 教训）：上线后持续监控事件率/medW/日均 kWh；test 段事件构成漂移可击穿任何静态验收预期——建议按周/月对账子表（若有）或人工抽检。

---

## 8. 生产准入 Go/No-Go 清单

| # | 项 | 通过标准 |
| --- | --- | --- |
| 1 | 口径档案 | data_spec.json 齐全且与部署端口径（功率类型/步长/电压制度）一致 |
| 2 | viability | 普查 val ON ≥90 且无混表/表底红旗 |
| 3 | 相对优势 | 探针配对显著优（3/3+>2×SEM）或有 n=5 锁定档案 |
| 4 | Test 四线 | F1≥0.75 / R≥0.70 / \|EE\|≤0.15（预算 2 次内至少 1 次全过；绝对 S 线仅同分布内要求） |
| 5 | 多 seed 稳定性 | 选型依据 ≥3 seeds；无单 seed 赢家诅咒迹象 |
| 6 | EE 语义 | 稀疏电器附 EE 分母备注（15% 线的绝对 kWh 含义） |
| 7 | 监控计划 | 事件率/medW/kWh 日志 + 漂移阈值告警 + 重训触发条件 |

---

## 9. 常见坑速查（项目史提炼）

| 坑 | 症状 | 规避 |
| --- | --- | --- |
| 混表子表 | F1 语义失效、EE 异常 | 接入前查 labels 共享（H4 教训） |
| 表底常驻 | kWh/day 全来自基线 | 普查看 implied 常驻功率 |
| 短记录 | val 事件 <90 | viability 红线直接排除 |
| 15min 数据 | 事件不可见 | 换 ≤1min 源 |
| 单 seed 定胜负 | val-test 解耦 | ≥3 seeds 配对；Test 预算 2 次 |
| val EE 低就放心 | test EE 方向翻转 | EE 方向只能 Test 实测 |
| 绝对线跨 house 照抄 | S 线莫名爆 | 业务三线+相对优势为主判据（发现 #4） |
| MAE 选型 | 稀疏电器欠预测 EE | 细搜用 composite 选型 |
| 配置撞名 | 覆盖旧档案 | 新建前 `ls configs/` |
| PowerShell foreach 裸词 | 列表不展开 | `foreach ($d in "a","b")` 带引号 |

---

*维护：本手册随管道演进同步更新；新增脚本/口径变更须过烟测并在此登记。 infer.py 烟测记录：合成数据端到端（训练 3ep→推理 19936 点→事件数与真值一致→跨数据 stats-from 模式→CSV/npz 双输出）2026-09-16 通过。*
