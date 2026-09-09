"""细搜批次汇总：扫描 runs 目录下 <变体>_s<种子> 子目录，按变体输出 mean±std 表。

每个子目录由 scripts/train.py 产出（history.json 含 select_metric 与 val_score）。
对每个 run：取 history 中 val_score 最小（最优）的 epoch 行作代表（与早停/checkpoint
判据同口径），再按变体聚合 mean±std。输出按平均 val_score 升序，便于直接回贴判读。

目录命名：<任意变体名>_s<种子>（如 v0_anchor_s1000、c1_lr2e4_s6000 均可识别）。

用法：
    python scripts/summarize_fine.py --runs-dir reports/fine
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

METS = ["val_score", "val_mae", "val_rmse", "val_f1", "val_precision",
        "val_recall", "val_energy_error"]
RUN_NAME = re.compile(r"^(.+)_s(\d+)$")


def best_row(run_dir):
    hist = json.loads((Path(run_dir) / "history.json").read_text(encoding="utf-8"))
    if not hist:
        raise ValueError(f"{run_dir}/history.json 为空")
    return min(hist, key=lambda r: r["val_score"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="reports/fine")
    args = ap.parse_args()

    dirs = sorted(Path(args.runs_dir).iterdir())
    runs = []
    for d in dirs:
        if d.is_dir() and RUN_NAME.match(d.name) and (d / "history.json").exists():
            runs.append(d)
    print(f"扫描到 {len(runs)} 个 run 目录（{args.runs_dir}）")
    if not runs:
        raise SystemExit("没找到 <变体>_s<种子> 目录。")

    groups = {}
    for run in runs:
        try:
            var, seed = RUN_NAME.match(run.name).groups()
            groups.setdefault(var, []).append(best_row(run))
        except Exception as e:
            print(f"WARN: 读取 {run} 失败: {e}", file=sys.stderr)

    if not groups:
        raise SystemExit("没有任何可汇总的 run。")

    # 每变体：n / 各指标 mean±std / best_epoch 均值
    stats = {}
    for var, rows in groups.items():
        n = len(rows)
        per = {}
        for key in METS:
            vals = [r[key] for r in rows]
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if n > 1 else 0.0
            per[key] = (m, s)
        ep = statistics.mean([r["epoch"] for r in rows])
        stats[var] = (n, per, ep)

    hdr = f"{'variant':<16}{'n':>3} " + " ".join(f"{m[4:]:>10}" for m in METS) + f"{'best_ep':>8}"
    print(hdr)
    print("-" * len(hdr))
    for var in sorted(stats, key=lambda v: stats[v][1]["val_score"][0]):
        n, per, ep = stats[var]
        line = f"{var:<16}{n:>3} " + " ".join(f"{per[k][0]:10.4f}" for k in METS) + f"{ep:8.1f}"
        print(line)

    # 详表（便于回贴判读）
    print("\n=== 详表（mean±std，可直接回贴）===")
    for var in sorted(stats, key=lambda v: stats[v][1]["val_score"][0]):
        n, per, ep = stats[var]
        parts = [f"{var} (n={n})"]
        for key in METS:
            m, s = per[key]
            parts.append(f"{key[4:]}={m:+.4f}±{s:.4f}")
        print(" | ".join(parts) + f" | best_ep={ep:.1f}")


if __name__ == "__main__":
    main()
