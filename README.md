# Transformer-NILM：UK-DALE + PyTorch 完整训练与调优项目

本项目面向 Windows + Conda + PyTorch，采用公开 UK-DALE 数据集构建一个可复现的 Transformer Encoder Seq2Point NILM 实验。

> 重要：本项目不会伪造实验结果。当前执行环境没有 UK-DALE 原始/处理数据文件，因此本次只能完成代码、配置、文档和 CPU smoke test；真实 UK-DALE 指标必须在下载数据后运行 `run_real.ps1` 得到。

## 1. 实验目标

输入：
- 家庭总负荷 Aggregate

输出：
- 指定电器 Kettle 的功率

模型：
- Input Projection
- Sinusoidal Positional Encoding
- Transformer Encoder
- Center-point Regression Head

评价：
- MAE
- RMSE
- R²
- SAE
- Precision / Recall / F1（基于 ON/OFF 阈值）
- Energy Error

调优：
- learning rate
- d_model
- nhead
- num_layers
- dim_feedforward
- dropout
- weight_decay
- batch_size
- window_size

## 2. 数据来源

本项目优先支持 NILMbench 公开的 UK-DALE 处理数据，以及标准 UK-DALE HDF5/CSV 数据。

UK-DALE 官方：
https://jack-kelly.com/data/

UK-DALE 论文：
https://www.nature.com/articles/sdata20157

NILMbench：
https://github.com/nilmtk/nilmbench

处理后的 UK-DALE：
https://huggingface.co/datasets/Pybunny/nilmbench-ukdale

### 推荐路径

对于本项目的低频 Seq2Point 教学实验，推荐使用 UK-DALE 6-second 数据，而不是 16 kHz 波形。原始 UK-DALE 的 6 秒数据包含 whole-home 与 appliance-level readings。

如果已经有 `ukdale.h5`，可以先执行：

```powershell
conda activate transformer_nilm
python scripts\inspect_h5.py --path D:\datasets\ukdale.h5
```

## 3. Windows + Conda

```powershell
conda create -n transformer_nilm python=3.11 -y
conda activate transformer_nilm
pip install -r requirements.txt
```

GPU 用户请先按自己的 CUDA 版本安装对应 PyTorch，再安装其余依赖。

## 4. 项目结构

```text
transformer_nilm_project/
├── configs/
│   ├── baseline.yaml
│   └── tuning.yaml
├── data/
├── checkpoints/
├── logs/
├── reports/
├── src/
│   ├── data.py
│   ├── model.py
│   ├── metrics.py
│   ├── trainer.py
│   └── experiment.py
├── scripts/
│   ├── inspect_h5.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
│   └── run_smoke.py
├── tests/
├── requirements.txt
├── run_baseline.ps1
├── run_tuning.ps1
├── run_real.ps1
└── README.md
```

## 5. 先验证代码

```powershell
python scripts\run_smoke.py
```

Smoke test 使用可控的合成信号，只用于验证代码链路：
数据 -> Dataset -> Transformer -> Train -> Validation -> Test -> Metrics -> Checkpoint。

它不是 UK-DALE 科学实验结果。

## 6. Baseline

配置见 `configs/baseline.yaml`。

典型参数：

```yaml
window_size: 128
d_model: 64
nhead: 4
num_layers: 2
dim_feedforward: 128
dropout: 0.1
lr: 0.0005
weight_decay: 0.0001
batch_size: 128
epochs: 30
```

运行：

```powershell
python scripts\train.py --config configs\baseline.yaml --data-path D:\datasets\ukdale.h5
python scripts\evaluate.py --run-dir reports\baseline
```

## 7. 调优

不要直接拿 Test 指标调参。

流程：

```text
Train
  ↓
Validation
  ↓
调整超参数
  ↓
重新训练
  ↓
Validation 最优
  ↓
锁定配置
  ↓
Test 一次
```

运行：

```powershell
python scripts\tune.py --config configs\tuning.yaml --data-path D:\datasets\ukdale.h5
```

调优结果会保存到：

```text
reports/tuning_summary.csv
reports/best_config.yaml
```

## 8. 指标驱动的调优规则

### Train 差 + Val 差
优先检查：
1. 数据对齐
2. normalization
3. learning rate
4. window
5. 模型容量

### Train 好 + Val 差
说明过拟合：
1. dropout
2. weight decay
3. 减少层数
4. 数据增强
5. early stopping

### Train/Val 好 + Test 差
优先检查：
1. 时间分布变化
2. household/domain shift
3. 预处理一致性
4. 测试家庭是否与训练家庭不同

### Loss 震荡
优先降低 LR。

### Loss 下降非常慢
尝试提高 LR，但必须观察 Validation。

### MAE 好但 RMSE 很差
说明存在少量大误差/异常事件，应重点检查事件启动、关闭和大功率区间。

## 9. 真实 UK-DALE 实验

建议第一阶段：

```text
Train: House 1 前 70% 时间
Val:   House 1 中间 15%
Test:  House 1 后 15%
Appliance: kettle
```

第二阶段再做真正更有意义的跨家庭：

```text
Train: House 1
Test:  House 2
```

不要把跨家庭 Test 用于反复调参。

## 10. 实验记录

每次实验保存：

- experiment_id
- dataset
- appliance
- split
- window_size
- d_model
- nhead
- num_layers
- dropout
- lr
- weight_decay
- batch_size
- best_epoch
- train/val/test metrics
- checkpoint
- runtime
- git/config hash

## 11. 下一步

当真实 UK-DALE 实验跑通后，可以继续扩展：

```text
Transformer
├── Kettle
├── Microwave
├── Fridge
├── Dishwasher
└── Washing Machine
```

再扩展：

```text
LSTM
GRU
TCN
CNN
CNN-LSTM
Informer
PatchTST
```

最后接入 AI Agent：

```text
Training Agent
   ↓
Log Agent
   ↓
Metric Agent
   ↓
Diagnosis Agent
   ↓
Hyperparameter Agent
   ↓
Retraining Agent
   ↓
Experiment Manager
   ↓
Report Agent
```
