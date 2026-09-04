import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.data import make_synthetic_signal
from src.experiment import train_experiment

cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text(encoding="utf-8"))
cfg["data"]["window_size"] = 64
cfg["data"]["max_samples_train"] = 3000
cfg["data"]["max_samples_val"] = 600
cfg["data"]["max_samples_test"] = 600
cfg["model"]["d_model"] = 32
cfg["model"]["nhead"] = 4
cfg["model"]["num_layers"] = 1
cfg["model"]["dim_feedforward"] = 64
cfg["training"]["epochs"] = 3
cfg["training"]["patience"] = 2
cfg["training"]["batch_size"] = 64

x, y = make_synthetic_signal(10000)
result = train_experiment(x, y, cfg, "reports/smoke")
print(result)
print("\nSMOKE TEST PASSED. These metrics are NOT UK-DALE results.")
