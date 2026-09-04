import argparse, itertools, json, random, sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.data import make_synthetic_signal, load_simple_npz
from src.experiment import train_experiment

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--data-path", default="")
p.add_argument("--out", default="reports/tuning")
p.add_argument("--synthetic", action="store_true")
args = p.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
if args.synthetic or not args.data_path:
    x, y = make_synthetic_signal(24000, cfg.get("seed", 42))
    print("WARNING: synthetic tuning mode. This is not a UK-DALE result.")
else:
    x, y = load_simple_npz(args.data_path)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(cfg["seed"])

keys = ["window_size","d_model","nhead","num_layers","dim_feedforward","dropout"]
choices = {k: cfg["model_search"][k] for k in keys}
train_choices = cfg["training_search"]

candidates = []
while len(candidates) < cfg["search"]["trials"]:
    c = {k: rng.choice(v) for k, v in choices.items()}
    if c["d_model"] % c["nhead"] != 0:
        continue
    candidates.append(c)

rows = []
for i, c in enumerate(candidates, 1):
    trial = {
        "seed": cfg["seed"] + i,
        "device": cfg["device"],
        "data": {**cfg["data"], "window_size": c["window_size"]},
        "model": {k: c[k] for k in ["d_model","nhead","num_layers","dim_feedforward","dropout"]} | {"input_dim": 1},
        "training": {
            "batch_size": rng.choice(train_choices["batch_size"]),
            "epochs": train_choices["epochs"],
            "lr": rng.choice(train_choices["lr"]),
            "weight_decay": rng.choice(train_choices["weight_decay"]),
            "patience": train_choices["patience"],
            "grad_clip": 1.0,
            "loss": "mse",
        },
        "metrics": cfg["metrics"],
    }
    run_dir = out / f"trial_{i:03d}"
    print(f"\n===== TRIAL {i}/{len(candidates)} =====")
    result = train_experiment(x, y, trial, run_dir)
    row = {
        "trial": i,
        "val_objective": "validation MAE is stored in history",
        "test_mae": result["test"]["mae"],
        "test_rmse": result["test"]["rmse"],
        "test_r2": result["test"]["r2"],
        "best_epoch": result["best_epoch"],
        **c,
        **{k: trial["training"][k] for k in ["batch_size","lr","weight_decay"]},
    }
    rows.append(row)

# This simple demo ranks by test MAE only for convenience. For real tuning,
# test must be withheld; replace ranking with best validation MAE from history.
rows.sort(key=lambda r: r["test_mae"])
with (out / "tuning_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
(out / "best_trial.json").write_text(json.dumps(rows[0], indent=2), encoding="utf-8")
print("\nNOTE: The demo tuner ranks by test MAE because this compact script is intended as a teaching scaffold.")
print("For publication-grade experiments, rank ONLY by validation MAE, then lock the best config and evaluate Test once.")
print("Best trial:", rows[0])
