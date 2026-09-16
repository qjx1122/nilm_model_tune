# EDGE_DEPLOYMENT.md — 边缘终端实时推理部署（P0 已交付 + 全景规划）

> 基线：REPORT.md v1.4 / commit 链至 e624daa。本文回答：现有算法如何从「离线 npz 批式推理」
> 落到「边缘总线终端 6400Hz 三相周波流实时推理」。**P0（最小闭环）已于 2026-09-16 交付并通过
> 端到端 parity 测试**；P1/P2 待办见 §8。

---

## 0. 形态跃迁

```
现状：6s 等步长单通道功率 npz → train/evaluate/infer.py（离线批式）
目标：6400Hz 三相 V/I 周波实时流 → 每周波电参量 → 6s 桶 → 模型推理 → 事件流+kWh

边缘终端（6400Hz 三相 V/I 周波，D 接口）
   │ ① 波形特征层（周波→P/Vrms/Irms/S→6s 桶）      ← P0 已实现（nilm_edge.c）
   │ ② 流式推理引擎（环形缓冲+每 6s tick 前向）     ← P0 已实现
   │ ③ 模型导出/嵌入式运行时（纯 C，零依赖）        ← P0 已实现（export_edge_bundle.py）
   │ ④ 接口适配层（动态库 C ABI）                   ← P0 已实现
   ▼
 NilmEdgeResult 流（center_ts / aggregate_w / pred_w / on）
```

关键换算：6400Hz @ 50Hz = 128 点/周波；20ms/包；每 6s = 300 周波；窗口 96/192 点 = 9.6/19.2 min。

## 1. 终端接口约定（2026-09-16 与终端方锁定，Module D 六问）

| # | 项 | 约定 |
| --- | --- | --- |
| D1 | 传输形态 | **动态库调用**：终端主程序加载本库（C ABI） |
| D2 | 数据格式 | **工程量 float（V/A），带时间戳**（秒，double，单调不减） |
| D3 | 分包语义 | **每 128 点×6 通道一包（20ms/包，50 包/s）**；一包=一个整周波，**周波对齐/频率跟踪由终端采集侧负责** |
| D4 | 丢帧标记 | **调用方用上一包数据重推**（本库按时间戳分桶，天然兼容；整 6s 桶缺失由本库 carry 填充，见 §4） |
| D5 | 代码位置 | **终端本机**，主程序加载我们的动态库调用 |
| D6 | 三相同步 | 6 通道**同采同时钟**，交错顺序 `[uA, iA, uB, iB, uC, iC]` |

## 2. P0 交付物（本仓库）

| 文件 | 内容 |
| --- | --- |
| `edge/nilm_edge.h` / `edge/nilm_edge.c` / `edge/Makefile` | 纯 C99 零依赖动态库（libnilm_edge.so；Windows 下同源编译为 dll）：波形特征层 + 6s 桶引擎 + 环形缓冲 + Transformer seq2point 前向（双精度激活）；结果 FIFO 每模型 **2048** 深度（≈3.4h 不轮询容忍） |
| `scripts/export_edge_bundle.py` | 部署包导出：run 目录（best.pt+result.json）+ 训练 npz → `model.bin` + `manifest.json` |
| `scripts/edge_stream_test.py` | **分阶段验证 harness**：S1 合成周波流（内置三相场景→训练→导出→流式推送→15 项功能验收）；S2 录制回放（--mode replay：质量报告+事件指标+--compare-with 确定性对拍） |
| `tests/test_edge_parity.py` | 端到端 parity：tiny 训练链路 + 真实配置随机权重 + 缺口 carry + 暖机（§7） |
| 本文档 | 接口/API/格式/构建/集成/待办 |

## 3. 引擎数据流与语义（nilm_edge.c）

1. **push_packet**：每包按相累加 Σu·i、Σu²、Σi²（128 点全量，不抽样）；
2. **6s 桶**：按 `floor(ts/6)` 分桶；桶结束（时间戳跨界）时换算每相 P=Σui/n、Vrms、Irms、S=Vrms·Irms，三相求和（口径=部署包 power_type，UK-DALE 训练=apparent 视在）；
3. **空桶 carry**：终端断流导致整桶无数据 → 用上一桶值填充（等价训练制备期 ffill 语义）；连续空桶 >300（30min）→ 判长缺口，**清缓冲重置暖机**（等价制备期 long-gap 剔除）；
4. **环形缓冲**：6s 功率值入 1024 深度环形缓冲；每模型 window 点齐即触发一次前向；
5. **前向**：与训练严格同构——窗口 `[center−W/2, center+W/2)`、归一化统计量=训练 npz train 段（导出冻结进部署包）、pe 位置编码=训练同源导出；输出 denorm 为 W；
6. **结果 FIFO**：每模型 2048 深度结果队列（时间序），poll 逐条取出——**不丢 6s 事件段**（S1 实测：长缺口恢复链瞬时入队 ~300 条，256 深度会丢最旧）。

**延迟（中心窗语义，与训练一致）**：预测中心落后最新数据 `(W − W/2 − 1)×6s`：
window=192 → **570s**；window=96 → **282s**。暖机：开机后 W 个 6s 桶内无结果（19.2/9.6min）。
若业务需秒级事件检出 → 因果窗重训（P2，见 §8）。

## 4. C API 参考

```c
#include "nilm_edge.h"

int  nilm_edge_start(void);                      /* 进程一次 */
int  nilm_edge_add_model(const char *bundle);    /* 每电器一个部署包 → model_id */
int  nilm_edge_push_packet(const float *wave, int points, int channels, double ts);
int  nilm_edge_poll(int model_id, NilmEdgeResult *out);  /* FIFO 取结果 */
int  nilm_edge_flush(void);                      /* 可选：停机前结算未满桶 */
void nilm_edge_shutdown(void);
```

```c
typedef struct {
    double center_ts;   /* 预测中心点时间戳（=6s 桶起点） */
    double aggregate_w; /* 该桶三相总功率（W，bundle 口径） */
    double pred_w;      /* 电器功率预测（W，可为负=回归原值；展示层可 clip） */
    int    on;          /* pred_w >= on_threshold_watts */
} NilmEdgeResult;
```

- **多电器**：`add_model` 多次（kettle/dw/mw 各一包），共享同一功率流；power_type 必须一致；
- **线程模型**：P0 非线程安全，全部调用须来自同一线程（终端采集回调线程或统一分发）；
- **ON 语义**：逐 6s 点判定；产品侧事件=连续 ON 段 + 去抖（最小事件时长参数属产品层，P0 未内置）；
- **能耗**：kWh = Σ pred_w × 6s / 3.6e6（调用方累计）。

## 5. 部署包格式与导出

```powershell
python scripts\export_edge_bundle.py --run-dir reports\h5k_f5_t23007 ^
    --stats-from D:\datasets\ukdale_h5_kettle.npz --out reports\edge_bundle_h5k
```

- `--stats-from` 必须传**训练时的 npz**（归一化统计量取其 train 段，与 infer.py/build_splits 同口径；run 目录不落盘统计量——已知改进项）；
- `--power-type`：训练 aggregate 口径（默认 apparent=UK-DALE mains 同款；换有功=未验证口径切换，须先过 E4 实验）；
- 产物：`model.bin`（header: magic NEDG/版本/架构 6 整数/统计量+阈值 5 双精度；随后 float32 权重数组，布局见脚本文档）+ `manifest.json`（来源/口径/延迟/sha256/参数量）。

## 6. 构建与终端集成

```bash
make -C edge            # Linux: libnilm_edge.so（gcc -O2 -std=c99，零外部依赖，仅 libm）
# Windows 终端: clang -shared -O2 -std=c99 nilm_edge.c -o nilm_edge.dll -lm
#               或 MSVC: cl /O2 /LD nilm_edge.c /Fe:nilm_edge.dll
```

终端主程序集成样例（C）：

```c
#include "nilm_edge.h"
nilm_edge_start();
int kettle = nilm_edge_add_model("bundle_kettle/model.bin");
/* 采集回调：每 20ms 一包 */
nilm_edge_push_packet(pkt->wave, 128, 6, pkt->ts);
/* 主循环：任意周期轮询，取空为止 */
NilmEdgeResult r;
while (nilm_edge_poll(kettle, &r) == 1)
    printf("t=%.0f agg=%.1fW pred=%.1fW %s\n", r.center_ts, r.aggregate_w, r.pred_w, r.on?"ON":"off");
```

资源占用（w192/d64/L2 单模型量级）：权重 ~0.4MB + 工作区 ~1MB；每 6s tick 一次前向
~20 MFLOPs（ARM Cortex-A 级数十 ms）；6s 桶引擎每包 768 次乘加×3 相累计。多模型线性叠加。

## 7. P0 验收（parity 测试，2026-09-16 通过）

| 用例 | 内容 | 结果 |
| --- | --- | --- |
| A tiny 训练链路 | 合成 npz → train.py（3ep）→ export → 60 桶波形流（含 3 桶缺口 carry） | n=45（45/45 全对上）；**Δagg rel 3.97e-15**；**Δpred max 0.00005W**；ON 全一致 ✓ |
| B 真实配置随机权重 | F4 架构（w96/d64/nhead8/L2/ff128）→ export → 104 桶流 | n=9（9/9）；Δagg rel 3.77e-15；Δpred max 0.00009W ✓ |
| C 暖机 | 窗口未满 poll | 返回 0 ✓ |

复跑：`python tests/test_edge_parity.py`（自动 make；依赖 numpy/torch）。
**注意**：parity=数值等价（C 引擎 vs torch 参考在同一 6s 功率序列上），**不等于业务精度**——
业务精度须过 §8 E1 黄金集（真实站点波形+子表真值四线验收）。

### 7.1 分阶段验证计划与结果（2026-09-16）

**S1 合成周波流（✅ 已通过，2026-09-16）**——`python scripts/edge_stream_test.py --mode synth --out-dir reports/edge_s1 --record rec.npz`

内置三相场景（基线+冰箱/电视背景负荷+kettle/dw 事件+噪声谐波；20s D4 短缺口+31min 长缺口）→
6 天 6s 序列渲染训练 tiny 双模型 → 导出部署包 → 4h=720,000 包流式推送 → 15 项验收全过：

| 验收项 | 结果 |
| --- | --- |
| 结果条数（语义推算精确命中） | 2264/2264（含长缺口 carry 300 + 重置再暖机 63 的完整推算） |
| 暖机/末端中心/长缺口重置+再暖机/短缺口 D4 连续 | 全 ✓（首中心桶 232、末端滞后 186s、恢复后首中心 1542 精确命中） |
| 特征层功率 vs 场景解析值 | max rel err **8.81e-04** ≤ 0.005 |
| kettle 点级/事件级 F1 | **1.0000 / 1.0000**（368/368 点、12/12 事件、kWh −0.04%） |
| dish_washer 点级/事件级 F1 | **0.9882 / 1.0000**（kWh +0.36%、起始偏移 −1 桶） |
| 性能 | **581× 实时**，引擎 1.03 ms/tick（双模型并行） |

S1 的过程价值：首跑暴露并修复 **3 个真 bug**——①引擎 carry 计数被 carry 推送自身清零→长缺口
永不重置（nilm_edge.c 已修：仅真实数据包到达才清零）；②harness 波形生成器只写 6/768 列
（布局错）；③样本时间轴误用周波间隔 20ms 而非采样间隔 1/6400s（sin 相位恒定）。
「先合成后现场」的分阶段设计立竿见影。

**S2 现场录制回放（harness 就绪+自证通过；现场数据待录）**

```powershell
python scripts\edge_stream_test.py --mode replay --wave-file 现场录制.npz ^
    --bundle bundle_kettle\model.bin --bundle bundle_dw\model.bin ^
    --out-dir reports/edge_s2 [--realtime] [--compare-with 基准report.json]
```

- 录制文件格式（NPZ v1）：`wave` float32 [n,768]（每行一包，128 点×6 通道交错
  [uA,iA,uB,iB,uC,iC]）+ `ts` float64 [n]（秒，单调不减）+ 可选 `truth_on_<appliance>`
  int8 [n]（受控切换实验真值）+ 可选 `meta` json。长录制按小时分文件，`--wave-file` 可多次；
- 输出：录制质量报告（包数/时长/疑似丢帧/Vrms 三相）+ 模型结果（事件段/kWh；有真值则 F1）；
- **自证（2026-09-16 通过）**：S1 录制样例段（59,950 包）回放，与 S1 直推结果**逐位一致**
  （kettle/dw 各 137 条交集 0 不一致）——「生成器直推」与「录制文件回放」两路径等价；
- 现场录制建议：段内含目标电器若干次真实使用（kettle ≥5 次、dw ≥2 次）；有条件时加装临时
  子表或受控切换（truth_on 键）→ 即成本地黄金集（§8 E1 的现场版）。

**S3 终端实测（待现场）**：终端加载 libnilm_edge（§6 集成样例）→ 与 S2 同口径验收；联调清单 §9。

## 8. 剩余待办（P0 未覆盖 = 上轮差距分析映射）

**P1（可信上线，按序）**
1. **E1 黄金集**（最大缺口）：至少一个真实站点——6400Hz 三相周波录制 + 临时子表真值 →
   边缘输出 vs 真值四线验收（F1≥0.75 / R≥0.70 / |EE|≤0.15 + 相对优势）；无此步不得宣称生产可用；
2. **权重来源决策**（本项目最大未验证前提）：五纪元全部是「同 house 有子表重训」；边缘是
   「无标签零样本迁移」。选项：纯零样本先跑 / 临时子表校准验收（建议至少此档）/ per-site 微调；
3. **E4 口径实验**：apparent vs active 输入对拍（黄金集上定夺）；中国 220V 电器 medW 普查
   （阈值重定夺，预注册 #3：kettle 500W 大概率可用但须确认）；
4. **ONNX 备选路径**（可选）：若终端已有 onnxruntime 运行时，可改走 ONNX 导出替换自研 C 前向
   （当前纯 C 路径已过 parity，此项仅为生态兼容备选）；
5. **监控运维**：事件率/medW/日均 kWh 滚动统计+告警（dw 教训：段级漂移可击穿静态验收）；
   看门狗/输出守卫（NaN/常数检测）；部署包版本管理+OTA。

**P2（增强）**
6. 因果窗重训（秒级事件检出，=新纪元全套验收）；
7. 6400Hz 谐波/V-I 轨迹特征利用（现有模型只用 6s 功率，波形信息是富矿）；
8. 多电器分项账单一致性（分项和 vs 总表对账约束，未验证）；
9. int8 量化（当前资源占用已轻，非必需）。

## 9. 集成联调清单（终端方确认项）

- [ ] 终端 OS/架构与编译工具链（Linux/Windows？ARM/x86？交叉编译由谁执行）；
- [ ] 动态库加载方式与符号可见性（dlopen/LoadLibrary 或链接期）；
- [ ] `ts` 时钟源与单位（Unix 秒？单调钟？跨重启是否回退——回退会打乱分桶，需确认）；
- [ ] 采集侧周波对齐精度与频率跟踪范围（D3 已约定由终端负责，需实测确认 128 点/周波误差）；
- [ ] 丢帧重推的实现位置与时间戳语义（重推包用原时间戳还是新时间戳？建议原时间戳+原数据）；
- [ ] 轮询线程与采集线程的关系（P0 单线程约定是否满足；否则需加锁版）；
- [ ] 站点是否有分布式光伏（负功率语义：引擎不 clip，产品层如何呈现）；
- [ ] E1 黄金集站点选择与子表安装计划。
