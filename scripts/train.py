import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.data import make_synthetic_signal, load_simple_npz
from src.experiment import train_experiment

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--data-path", default="")
p.add_argument("--out", default="reports/baseline")
p.add_argument("--synthetic", action="store_true")
p.add_argument("--seed", type=int, default=None,
               help="覆盖配置里的 seed（细搜多种子复跑用；否则同一 yaml 每次结果相同）")
args = p.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
if args.seed is not None:
    cfg["seed"] = args.seed
if args.synthetic or not args.data_path:
    x, y = make_synthetic_signal(20000, cfg.get("seed", 42))
    print("WARNING: synthetic mode. This is not a UK-DALE result.")
else:
    # Expected NPZ format: aggregate, target.
    x, y = load_simple_npz(args.data_path)

result = train_experiment(x, y, cfg, args.out)
print(result)
