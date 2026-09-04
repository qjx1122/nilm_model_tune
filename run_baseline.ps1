$ErrorActionPreference = "Stop"
conda activate transformer_nilm
python scripts\run_smoke.py
# Real data:
# python scripts\train.py --config configs\baseline.yaml --data-path D:\datasets\ukdale_prepared.npz --out reports\baseline
