"""推理脚本：用已训练 run 目录（best.pt + result.json）对总表数据做单电器功率拆分。

用法（PowerShell）：
  # 1) 在训练用的同一 npz 上推理（演示/回归验证；归一化统计量取自其 train 段，与训练完全一致）：
  python scripts\\infer.py --run-dir reports\\h5k_f5_t23007 --npz D:\\...\\ukdale_h5_kettle.npz --out reports\\infer_demo.npz

  # 2) 跨 house / 新数据推理：--stats-from 必须传训练时的 npz
  #    （x/y 归一化统计量只能来自训练数据 train 段，跨数据不可重算——见 docs/ONBOARDING.md §7）：
  python scripts\\infer.py --run-dir reports\\h5k_f5_t23007 --npz D:\\...\\new_house.npz ^
      --stats-from D:\\...\\ukdale_h5_kettle.npz --out reports\\infer_new.npz

输出：
  --out *.npz：{index, aggregate, pred, pred_on[, target, target_on]}（pred_on = pred >= on 阈值）
  --out *.csv：同列文本（大文件慎用：百万行级）
  控制台摘要：窗口/阈值/统计量/预测事件数（连续 ON 段）/估算 kWh；
  若 npz 含非零 target（评估场景），另打印 MAE/F1/P/R/EE 对照（与训练同口径）。

注意：
  - pred 为窗口中心点逐点功率（seq2point），首尾各 window//2 个点无预测；
  - 事件/能耗统计用 6s 步长（与 schema v5 一致）；其他步长数据请先重采样到 6s 网格。
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from runlog import setup_run_log
from src.model import NILMTransformer
from src.metrics import regression_metrics

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True, help="训练输出目录（须含 best.pt + result.json）")
p.add_argument("--npz", required=True, help="待推理数据：npz 含 aggregate（可含 target 供对照）")
p.add_argument("--stats-from", default="",
               help="归一化统计量来源 npz（须含 aggregate+target，取其 train 段）。"
                    "缺省=--npz 自身；跨数据推理必须传训练时的 npz")
p.add_argument("--out", default="reports/infer_out.npz", help="输出路径（.npz 或 .csv，按扩展名）")
p.add_argument("--batch", type=int, default=4096, help="推理批大小（显存/内存调节）")
p.add_argument("--sample-period-sec", type=float, default=6.0, help="采样步长（kWh 换算用）")
p.add_argument("--no-log", action="store_true", help="禁用控制台输出留痕")
args = p.parse_args()

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
setup_run_log(out_path.with_suffix(".log"), enabled=not args.no_log)

run_dir = Path(args.run_dir)
result_path = run_dir / "result.json"
if not result_path.exists() or not (run_dir / "best.pt").exists():
    raise SystemExit(f"run 目录不完整（需要 result.json + best.pt）：{run_dir}")
cfg = json.loads(result_path.read_text(encoding="utf-8"))["config"]
window = int(cfg["data"]["window_size"])
train_ratio = float(cfg["data"].get("train_ratio", 0.7))
threshold = float(cfg["metrics"]["on_threshold_watts"])


def _load_npz(path):
    d = np.load(path)
    if "aggregate" not in d:
        raise SystemExit(f"npz 缺 aggregate 键：{path}（实际 keys={list(d.keys())}）")
    agg = np.asarray(d["aggregate"], dtype=np.float64)
    tgt = np.asarray(d["target"], dtype=np.float64) if "target" in d else None
    return agg, tgt


def _train_stats(path):
    """镜像 build_splits：统计量只拟合 train 段（前 train_ratio 比例）。"""
    agg, tgt = _load_npz(path)
    if tgt is None:
        raise SystemExit(f"--stats-from 的 npz 必须含 target（y 统计量必需）：{path}")
    a = int(len(agg) * train_ratio)
    return (float(agg[:a].mean()), float(agg[:a].std() + 1e-6),
            float(tgt[:a].mean()), float(tgt[:a].std() + 1e-6))


stats_npz = args.stats_from or args.npz
x_mean, x_std, y_mean, y_std = _train_stats(stats_npz)
print(f"归一化统计量（来源 {stats_npz} train 段，train_ratio={train_ratio}）：")
print(f"  x_mean={x_mean:.2f} x_std={x_std:.2f} | y_mean={y_mean:.2f} y_std={y_std:.2f}")

agg, tgt = _load_npz(args.npz)
n = len(agg)
half = window // 2
if n < window:
    raise SystemExit(f"数据长度 {n} < 窗口 {window}，无法推理")
centers = np.arange(half, n - half, dtype=np.int64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NILMTransformer(**cfg["model"]).to(device)
model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
model.eval()

pred = np.empty(len(centers), dtype=np.float64)
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(centers), args.batch):
        cs = centers[i:i + args.batch]
        idx = cs[:, None] + np.arange(-half, window - half)[None, :]  # [B, W]
        x = (agg[idx] - x_mean) / x_std
        xb = torch.from_numpy(x[:, :, None].astype(np.float32)).to(device)
        pb = model(xb).cpu().numpy().astype(np.float64)
        pred[i:i + len(cs)] = pb * y_std + y_mean
print(f"推理完成：{len(centers)} 点 / {len(centers) * args.sample_period_sec / 86400:.1f} 天，"
      f"window={window}，device={device}，耗时 {time.time() - t0:.1f}s")

pred_on = pred >= threshold
kwh = float(pred.sum() * args.sample_period_sec / 3.6e6)
events = int(np.sum(pred_on[1:] & ~pred_on[:-1]) + (1 if pred_on.size and pred_on[0] else 0))
print(f"预测摘要：ON 点占比 {pred_on.mean() * 100:.3f}% | 连续 ON 段（事件）≈ {events} | "
      f"估算能耗 {kwh:.3f} kWh（步长 {args.sample_period_sec:g}s）")

cols = {"index": centers, "aggregate": agg[centers], "pred": pred, "pred_on": pred_on}
if tgt is not None and float(np.abs(tgt).sum()) > 0:
    t = tgt[centers]
    cols["target"] = t
    cols["target_on"] = t >= threshold
    m = regression_metrics(t, pred, threshold)
    print("对照真实子表（评估场景）：")
    print(f"  MAE={m['mae']:.2f}W | F1={m['f1']:.4f} | P={m['precision']:.4f} | R={m['recall']:.4f} | "
          f"EE={m['energy_error'] * 100:+.2f}%")
    print("  （注：这是全序列含 train/val 段的粗对照，正式验收以 train.py --test 的 test 段四线为准）")

if out_path.suffix == ".csv":
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(cols.keys()))
        rows = np.column_stack([cols[k].astype(float) for k in cols])
        w.writerows(rows)
else:
    np.savez(out_path, **cols)
print(f"已写出：{out_path}")
