$ErrorActionPreference = "Stop"
conda activate transformer_nilm
python scripts\tune.py --config configs\tuning.yaml --synthetic --out reports\tuning_smoke
