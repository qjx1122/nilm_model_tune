$ErrorActionPreference = "Stop"
conda activate transformer_nilm
# 烟雾/链路验证（合成信号，8 trials 快速跑通，结果不作数）
python scripts\tune.py --config configs\tuning.yaml --synthetic --trials 8 --out reports\tuning_smoke

# 真实调参（Test 冻结在 config 默认即生效，按 val 综合分排名）：
# python scripts\tune.py --config configs\tuning.yaml --data-path D:\datasets\ukdale_prepared.npz --out reports\tuning_p1
