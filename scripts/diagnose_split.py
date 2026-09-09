"""数据分段诊断：量化 train/val/test 三段 kettle 事件形态与能耗差异（纯数据，无需模型）。

用途：出现「val 好 / test 差」时先做此诊断（对应 README §8 已知失败模式「时间分布变化」），
判断是 test 段异常还是真实漂移。与 data.build_splits 同切分比例（0.70/0.15/0.15），
按原始连续段统计（不做 max_samples 抽样）。

输出每段：样本数/天数、ON 事件数与日均事件数、ON 占比、ON 功率分布（mean/median/p95）、
总能耗与日均能耗、总负荷均值/p95。事件按 target >= on_threshold 定义（默认 500W）。

用法：
    python scripts/diagnose_split.py --npz D:\\datasets\\ukdale_prepared.npz
    python scripts/diagnose_split.py --npz <npz> --on-threshold 500 --ratios 0.7 0.15 0.15
"""
import argparse
import json
from pathlib import Path

import numpy as np

SAMPLE_SEC = 6.0


def on_events(x, thr):
    above = x >= thr
    edges = np.diff(above.astype(np.int8))
    return int((edges == 1).sum()), int(above.sum())


def segment_stats(seg, name, thr):
    n = len(seg)
    if n == 0:
        return {"segment": name, "n_samples": 0, "note": "empty"}
    events, on_samp = on_events(seg, thr)
    on_pows = seg[seg >= thr]
    day = n * SAMPLE_SEC / 86400.0
    energy_kwh = float(seg.sum() * SAMPLE_SEC / 3600.0)
    return {
        "segment": name,
        "n_samples": n,
        "days": round(day, 2),
        "on_events": events,
        "events_per_day": round(events / day, 2) if day else 0.0,
        "on_fraction": round(on_samp / n, 4) if n else 0.0,
        "mean_on_power_w": round(float(on_pows.mean()), 1) if len(on_pows) else 0.0,
        "median_on_power_w": round(float(np.median(on_pows)), 1) if len(on_pows) else 0.0,
        "on_p95_w": round(float(np.percentile(on_pows, 95)), 1) if len(on_pows) else 0.0,
        "energy_kwh": round(energy_kwh, 1),
        "kwh_per_day": round(energy_kwh / day, 2) if day else 0.0,
        "agg_mean_w": round(float(seg.mean()), 1),
        "agg_p95_w": round(float(np.percentile(seg, 95)), 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--on-threshold", type=float, default=500.0)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    help="train/val/test 比例（与 build_splits 一致）")
    args = ap.parse_args()

    z = np.load(args.npz)
    target = z["target"]
    agg = z["aggregate"]
    a = int(len(target) * args.ratios[0])
    b = int(len(target) * (args.ratios[0] + args.ratios[1]))
    names = ["train", "val", "test"]
    slices = [target[:a], target[a:b], target[b:]]
    print(f"on_threshold={args.on_threshold}W  npz={Path(args.npz).name}  "
          f"n_total={len(target)}（{len(target)*SAMPLE_SEC/86400:.1f} 天）")
    print(f"{'segment':<7}{'n':>9}{'days':>7}{'on_evt':>8}{'evt/day':>9}"
          f"{'on_frac':>9}{'meanW':>8}{'medW':>8}{'p95W':>8}{'kWh':>9}{'kWh/day':>9}"
          f"{'aggW':>8}{'aggp95':>8}")
    for seg, nm in zip(slices, names):
        s = segment_stats(seg, nm, args.on_threshold)
        if "note" in s:
            print(f"{nm:<7} empty")
            continue
        print(f"{s['segment']:<7}{s['n_samples']:>9}{s['days']:>7}{s['on_events']:>8}"
              f"{s['events_per_day']:>9}{s['on_fraction']:>9.4f}{s['mean_on_power_w']:>8}"
              f"{s['median_on_power_w']:>8}{s['on_p95_w']:>8}{s['energy_kwh']:>9}"
              f"{s['kwh_per_day']:>9}{s['agg_mean_w']:>8}{s['agg_p95_w']:>8}")
    # 判读提示
    t, v = segment_stats(target[b:], "test", args.on_threshold), segment_stats(
        target[a:b], "val", args.on_threshold)
    print("\n判读提示：若 test 的 evt/day、mean/median ON 功率或 kWh/day 明显偏离 val/train，"
          "即为时间分布漂移证据（漏报/EE 偏差的根因方向）。")


if __name__ == "__main__":
    main()
