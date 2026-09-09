"""数据分段诊断：量化 train/val/test 三段电器事件形态与能耗差异（纯数据，无需模型）。
电器无关（--appliance 标签 + 默认阈值表，实录 22；kettle 为项目冻结口径）。

用途：出现「val 好 / test 差」时先做此诊断（对应 README §8 已知失败模式「时间分布变化」），
判断是 test 段异常还是真实漂移。与 data.build_splits 同切分比例（0.70/0.15/0.15），
按原始连续段统计（不做 max_samples 抽样）。

输出每段：样本数/天数、ON 事件数与日均事件数、ON 占比、ON 功率分布（mean/median/p95）、
总能耗与日均能耗、总负荷均值/p95。事件按 target >= on_threshold 定义。

阈值来源：显式 --on-threshold 优先；否则按 --appliance 查默认表
（kettle 500W / fridge·freezer 50W / dish_washer·washer_dryer·washing_machine 20W /
microwave 200W / boiler 100W；未收录电器回退 500W）。

用法：
    python scripts/diagnose_split.py --npz D:\\datasets\\ukdale_prepared.npz
    python scripts/diagnose_split.py --npz <npz> --appliance dish_washer
    python scripts/diagnose_split.py --npz <npz> --on-threshold 500 --ratios 0.7 0.15 0.15
"""
import argparse
import json
from pathlib import Path

import numpy as np

SAMPLE_SEC = 6.0
APPLIANCE_DEFAULT_THRESHOLD_W = {
    "kettle": 500.0, "fridge": 50.0, "freezer": 50.0,
    "dish_washer": 20.0, "washer_dryer": 20.0, "washing_machine": 20.0,
    "microwave": 200.0, "boiler": 100.0,
}


def on_events(x, thr):
    above = x >= thr
    edges = np.diff(above.astype(np.int8))
    return int((edges == 1).sum()), int(above.sum())


def segment_stats(seg, agg_seg, name, thr):
    n = len(seg)
    if n == 0:
        return {"segment": name, "n_samples": 0, "note": "empty"}
    events, on_samp = on_events(seg, thr)
    on_pows = seg[seg >= thr]
    day = n * SAMPLE_SEC / 86400.0
    energy_kwh = float(seg.sum() * SAMPLE_SEC / 3600.0 / 1000.0)  # W→kWh（/1000）
    off = seg < thr
    agg_off_mean = float(agg_seg[off].mean()) if off.any() else 0.0
    corr = float(np.corrcoef(agg_seg, seg)[0, 1]) if np.std(agg_seg) > 0 else 0.0
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
        "kwh_per_day": round(energy_kwh / day, 3) if day else 0.0,
        "agg_mean_w": round(float(agg_seg.mean()), 1),
        "agg_p95_w": round(float(np.percentile(agg_seg, 95)), 1),
        "agg_off_mean_w": round(agg_off_mean, 1),
        "corr_agg_target": round(corr, 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--appliance", default="kettle",
                    help="电器名（显示 + 默认阈值表查找；kettle=冻结口径 500W）")
    ap.add_argument("--on-threshold", type=float, default=None,
                    help="ON 事件阈值 W（显式给出则优先于 --appliance 默认表）")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    help="train/val/test 比例（与 build_splits 一致）")
    args = ap.parse_args()

    if args.on_threshold is not None:
        thr, src = float(args.on_threshold), "explicit"
    else:
        thr = APPLIANCE_DEFAULT_THRESHOLD_W.get(args.appliance, 500.0)
        src = (f"--appliance {args.appliance} 默认表"
               if args.appliance in APPLIANCE_DEFAULT_THRESHOLD_W else "未收录电器回退 500W")

    z = np.load(args.npz)
    target = z["target"]
    agg = z["aggregate"]
    a = int(len(target) * args.ratios[0])
    b = int(len(target) * (args.ratios[0] + args.ratios[1]))
    names = ["train", "val", "test"]
    slices = [(target[:a], agg[:a]), (target[a:b], agg[a:b]), (target[b:], agg[b:])]
    print(f"appliance={args.appliance}  on_threshold={thr}W（{src}）  "
          f"npz={Path(args.npz).name}  n_total={len(target)}（{len(target)*SAMPLE_SEC/86400:.1f} 天）")
    print(f"{'segment':<7}{'n':>9}{'days':>7}{'on_evt':>8}{'evt/day':>9}"
          f"{'on_frac':>9}{'meanW':>8}{'medW':>8}{'p95W':>8}{'kWh':>9}{'kWh/day':>9}"
          f"{'aggW':>8}{'aggp95':>8}{'aggOffW':>9}{'corr':>8}")
    for (t, ag), nm in zip(slices, names):
        s = segment_stats(t, ag, nm, thr)
        if "note" in s:
            print(f"{nm:<7} empty")
            continue
        print(f"{s['segment']:<7}{s['n_samples']:>9}{s['days']:>7}{s['on_events']:>8}"
              f"{s['events_per_day']:>9}{s['on_fraction']:>9.4f}{s['mean_on_power_w']:>8}"
              f"{s['median_on_power_w']:>8}{s['on_p95_w']:>8}{s['energy_kwh']:>9}"
              f"{s['kwh_per_day']:>9}{s['agg_mean_w']:>8}{s['agg_p95_w']:>8}"
              f"{s['agg_off_mean_w']:>9}{s['corr_agg_target']:>8}")
    # 判读提示
    print(f"\n判读提示（appliance={args.appliance}）：")
    print("1) test 的 evt/day、kWh/day 明显高于 train/val → 时间分布漂移证据（见执行实录 5）。")
    print(f"2) agg_off_mean_w（{args.appliance} 关断时的总负荷均值）若 ≈0 且 corr_agg_target≈1 → "
          f"aggregate 几乎只含 {args.appliance}，不是真实总负荷（数据制备红旗），"
          "NILM 前提失效，先修数据再谈模型。")
    print("3) 常开型电器（fridge/freezer 等 on_frac≈1 属正常）不适用 evt/day 解读，"
          "以功率形态与 kWh/day 为主。")


if __name__ == "__main__":
    main()
