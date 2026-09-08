"""随机搜索调参（按 Validation 决策，与 REPORT_TEST.md 调参方案专题一致）。

铁律：
1. Test 冻结：默认搜索中不评估 test（search.report_test: false → 各 trial 的
   data.eval_test=false），test 列仅留档位、绝不参与排名。
2. 选型只用 val；指标口径 = search.objective.metric（composite | mae）：
   - composite：业务综合分 S（见 src/objective.py），先过业务硬门槛再排序；
   - mae：历史口径，按 val MAE 排序（门槛不生效）。
3. 留档可复现：tuning_summary.csv 记录每个 trial 的 seed / git_commit /
   runtime / best_epoch / val 全套指标 / 完整超参；最优配置写 best_config.yaml
   （与 README 一致）。

用法：
    python scripts/tune.py --config configs/tuning.yaml --data-path <npz>
    python scripts/tune.py --config configs/tuning.yaml --synthetic --trials 8
"""
import argparse, csv, json, random, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.data import make_synthetic_signal, load_simple_npz
from src.experiment import train_experiment
from src.objective import (
    DEFAULT_MAE_NORM_WATTS,
    DEFAULT_WEIGHTS,
    row_score,
    gate_results,
)

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--config", required=True)
p.add_argument("--data-path", default="")
p.add_argument("--out", default="reports/tuning")
p.add_argument("--synthetic", action="store_true",
               help="用合成信号（仅链路验证，不是真实结果）")
p.add_argument("--trials", type=int, default=None,
               help="覆盖 search.trials（烟雾测试传小值）")
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

search_cfg = cfg["search"]
trials = int(args.trials or search_cfg["trials"])
objective_cfg = search_cfg.get("objective", {}) or {}
metric = objective_cfg.get("metric", "mae")
if metric not in ("mae", "composite"):
    raise SystemExit(f"search.objective.metric 只支持 'mae' 或 'composite'，当前为 {metric!r}")
gates = objective_cfg.get("gates", {}) or {}
use_gates = metric == "composite" and bool(gates.get("enabled", False))
report_test = bool(search_cfg.get("report_test", False))

try:
    git_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip() or "unknown"
except Exception:
    git_commit = "unknown"

model_keys = ["window_size", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout"]
model_choices = {k: cfg["model_search"][k] for k in model_keys}
train_choices = cfg["training_search"]

# 固定 trial 配置的公共部分（除随机采样的超参外）。
def build_trial(seed, c):
    trial = {
        "seed": seed,
        "device": cfg["device"],
        "data": {**cfg["data"], "window_size": c["window_size"], "eval_test": report_test},
        "model": {"input_dim": 1, **{k: c[k] for k in model_keys[1:]}},
        "training": {
            "batch_size": rng.choice(train_choices["batch_size"]),
            "epochs": train_choices["epochs"],
            "lr": rng.choice(train_choices["lr"]),
            "weight_decay": rng.choice(train_choices["weight_decay"]),
            "patience": train_choices["patience"],
            "grad_clip": 1.0,
            "loss": "mse",
            "select_metric": metric,
        },
        "metrics": cfg["metrics"],
    }
    if metric == "composite":
        trial["training"]["objective"] = {
            "weights": objective_cfg.get("weights", DEFAULT_WEIGHTS),
            "mae_norm_watts": objective_cfg.get("mae_norm_watts", DEFAULT_MAE_NORM_WATTS),
        }
    return trial

records = []
passed_ids, failed_ids = [], []
for i in range(1, trials + 1):
    while True:
        c = {k: rng.choice(v) for k, v in model_choices.items()}
        if c["d_model"] % c["nhead"] == 0:
            break
    seed = cfg["seed"] + i
    trial = build_trial(seed, c)
    run_dir = out / f"trial_{i:03d}"
    print(f"\n===== TRIAL {i}/{trials} (seed={seed}) =====")
    result = train_experiment(x, y, trial, run_dir)

    hist = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    best_row = next((r for r in hist if r["epoch"] == result["best_epoch"]), hist[-1])
    val_composite = row_score(best_row, objective_cfg)
    val_score = val_composite if metric == "composite" else best_row["val_mae"]
    ok, reasons = gate_results(best_row, gates) if use_gates else (None, [])

    rec = {
        "trial": i,
        "seed": seed,
        "git_commit": git_commit,
        "runtime_sec": round(result["runtime_sec"], 2),
        "best_epoch": result["best_epoch"],
        "select_metric": metric,
        "gate": "pass" if ok is True else ("fail:" + ";".join(reasons) if ok is False else "na"),
        "val_score": round(val_score, 6),
        "val_composite": round(val_composite, 6),
        "val_mae": round(best_row["val_mae"], 4),
        "val_rmse": round(best_row["val_rmse"], 4),
        "val_r2": round(best_row["val_r2"], 6),
        "val_sae": round(best_row["val_sae"], 6),
        "val_f1": round(best_row["val_f1"], 6),
        "val_precision": round(best_row["val_precision"], 6),
        "val_recall": round(best_row["val_recall"], 6),
        "val_energy_error": round(best_row["val_energy_error"], 6),
        **c,
        **{k: trial["training"][k] for k in ["batch_size", "lr", "weight_decay"]},
    }
    if report_test and result["test"]:
        t = result["test"]
        rec.update(test_mae=round(t["mae"], 4), test_rmse=round(t["rmse"], 4),
                   test_r2=round(t["r2"], 6), test_f1=round(t["f1"], 6))
    rec["run_dir"] = str(run_dir)
    records.append(rec)
    if ok is True:
        passed_ids.append(i)
    elif ok is False:
        failed_ids.append((i, ";".join(reasons)))
    print(f"best_epoch={result['best_epoch']} | val_score={val_score:.4f} | "
          f"val_mae={best_row['val_mae']:.2f} | val_f1={best_row['val_f1']:.3f}")

# ---- 排名：只按 Validation 指标 ----
candidates = records
if passed_ids:
    candidates = [r for r in records if r["gate"] == "pass"]
    print(f"\n门槛通过 {len(passed_ids)}/{len(records)} 个 trial，按 val 综合分排序。")
elif use_gates:
    print("\nWARNING: 全部 trial 未过业务硬门槛（F1/recall/|EE|）。"
          "先检查数据与训练是否正常；此处仍按 val 分排序供诊断，"
          "该轮结果不能作为推荐配置。")
candidates = sorted(candidates, key=lambda r: r["val_score"])
for rank, r in enumerate(candidates, 1):
    r["rank"] = rank
# 未过门槛的 trial 保留在 summary 末尾（rank 留空），保证全部留档可查。
rejected = [r for r in records if not r.get("rank")]
for r in rejected:
    r["rank"] = ""

fieldnames = [
    "rank", "trial", "seed", "git_commit", "runtime_sec", "best_epoch",
    "select_metric", "gate", "val_score", "val_composite", "val_mae", "val_rmse",
    "val_r2", "val_sae", "val_f1", "val_precision", "val_recall",
    "val_energy_error", *model_keys, "batch_size", "lr", "weight_decay",
    "test_mae", "test_rmse", "test_r2", "test_f1",
]
all_rows = candidates + rejected
with (out / "tuning_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(all_rows)

if candidates:
    best_rec = candidates[0]
    best_cfg = json.loads(
        (Path(best_rec["run_dir"]) / "result.json").read_text(encoding="utf-8")
    )["config"]
    (out / "best_config.yaml").write_text(
        yaml.safe_dump(best_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

print(f"\n排名依据：Validation（{metric}）。test 列仅留档，未参与任何选择"
      + ("（本搜索未评估 test）" if not report_test else "") + "。")
print(f"Summary: {out / 'tuning_summary.csv'}")
print(f"Best config: {out / 'best_config.yaml'}")
print("\nTop-5 (val_score | val_mae | val_f1 | val_ee | window | d_model | layers | lr):")
for r in candidates[:5]:
    print(f"  #{r['rank']:>2} trial {r['trial']:>3} | {r['val_score']:.4f} | "
          f"{r['val_mae']:.1f} | {r['val_f1']:.3f} | {r['val_energy_error']:+.3f} | "
          f"w{r['window_size']} d{r['d_model']} L{r['num_layers']} "
          f"lr={r['lr']:.0e}")
