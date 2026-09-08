"""UK-DALE HDF5 → aggregate/target NPZ 制备脚本（REPORT_TEST.md 调参方案·阶段 0 规格）。

支持布局（本脚本契约，先跑 --list-meters 核对）：
    /building{house}/elec/meter{id}/power   数据集形状 (N, 2)
    第 0 列时间戳（秒或纳秒，自动识别；纳秒 >1e14 自动 /1e9），第 1 列功率（W）。
UK-DALE 官方/NILMTK 转换文件的常见布局即此形式；布局不符时报错并打印键树，
不会靠猜硬解。

数据规格（写进 data_spec.json 留痕，作为以后所有 KPI 的口径依据）：
1. mains 各表（--mains-ids，默认 1,2）按时间并集对齐；短缺口（<= mains-gap-min
   分钟）ffill，剩余 NaN 行剔除。
2. kettle（--kettle-meter-id）：短缺口（<= kettle-gap-min 分钟）填 0（关断即 0），
   更长缺口视为不可信区间。
3. 剔除任一列仍为 NaN / 不可信区间的行后，取最长连续段（不跨大缺口拼接）。
4. 负功率视为测量伪迹 clip 到 0（计数并留痕）；单位统一 W。

输出：
    <out>.npz             {aggregate, target} float32 一维等长
    <out 同名>.data_spec.json  口径元数据（来源/表号/时间范围/缺口策略/样本数）

用法（示例）：
    python scripts\\prepare_ukdale.py --h5-path D:\\datasets\\ukdale.h5 --list-meters
    python scripts\\prepare_ukdale.py --h5-path D:\\datasets\\ukdale.h5 ^
        --mains-ids 1,2 --kettle-meter-id <上一步输出的表号> ^
        --out D:\\datasets\\ukdale_prepared.npz
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SECONDS_PER_SAMPLE = 6  # UK-DALE 低频 6 秒


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def _list_groups(f, prefix="building"):
    return sorted(k for k in f.keys() if str(k).lower().startswith(prefix))


def find_building(f, house):
    """定位 building 组：优先 building{house}，否则前缀匹配。"""
    cand = [f"building{house}", f"building_{house}", f"house_{house}", f"house{house}"]
    for name in cand:
        if name in f:
            return f[name], name
    groups = _list_groups(f)
    hits = [k for k in groups if str(house) in k]
    if len(hits) == 1:
        return f[hits[0]], hits[0]
    raise RuntimeError(
        f"找不到 building{house}（候选组：{groups or '(空)'}）。请先用 --list-meters 或 "
        f"scripts/inspect_h5.py 查看文件结构，再按实际布局传参。")


def meter_groups(f, building_group):
    b = building_group if not isinstance(building_group, str) else f[building_group]
    elec = b["elec"] if "elec" in b else b
    meters = {}
    for key in elec.keys():
        s = str(key)
        if s.lower().startswith("meter"):
            try:
                meters[int(s[len("meter"):])] = elec[key]
            except ValueError:
                continue
    return meters


def read_power_series(f, meter_group):
    """读取 (N,2) power 数据集 → pandas Series（UTC DatetimeIndex，单位 W）。"""
    keys = list(meter_group.keys())
    ds_name = next((k for k in ("power", "power_series") if k in meter_group), None)
    if ds_name is None:
        raise RuntimeError(f"meter 组 {meter_group.name} 下没有 power/power_series，"
                           f"实际键：{keys}。布局与本脚本契约不符，请 --list-meters 核对。")
    arr = np.asarray(meter_group[ds_name][()], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError(f"{meter_group.name}/{ds_name} 形状 {arr.shape} 不是 (N,2)"
                           "（时间戳, 功率），布局与本脚本契约不符。")
    ts, val = arr[:, 0], arr[:, 1]
    if ts.max() > 1e14:  # 纳秒时间戳 → 秒
        ts = ts / 1e9
    s = pd.Series(val, index=pd.to_datetime(ts, unit="s"))
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s = s[np.isfinite(s.values)]
    return s


def _meter_summary(f, meter_group):
    try:
        s = read_power_series(f, meter_group)
        attrs = dict(meter_group.attrs)
        app = attrs.get("appliance") or attrs.get("metadata") or ""
        if not isinstance(app, str):
            app = json.dumps(app, ensure_ascii=False)[:120]
        return {"id": int(meter_group.name.rstrip("/").split("meter")[-1]),
                "n": len(s), "start": str(s.index.min()), "end": str(s.index.max()),
                "attrs": str(app)[:120]}
    except Exception as e:
        return {"id": int(meter_group.name.rstrip("/").split("meter")[-1]),
                "error": str(e)}


def cmd_list_meters(f, house):
    bg, name = find_building(f, house)
    meters = meter_groups(f, bg)
    if not meters:
        raise RuntimeError(f"{name} 下没有 meter* 组（elec 内键：{list(bg['elec'].keys()) if 'elec' in bg else list(bg.keys())}）。")
    print(f"house: {name} | meters: {len(meters)}")
    print(f"{'id':>9} {'n_samples':>10}  {'start':<22} {'end':<22} attrs")
    for mid in sorted(meters):
        m = _meter_summary(f, meters[mid])
        if "error" in m:
            print(f"meter {mid:<3}  读取失败: {m['error']}")
        else:
            print(f"meter {mid:<3} {m['n']:>10}  {m['start']:<22} {m['end']:<22} {m['attrs']}")
    print("\n提示：House 的 mains 通常为小号表（如 1/2），kettle 表号以各 house 元数据为准。")


def _combine_mains(f, meters, mains_ids):
    found, series = [], []
    for mid in mains_ids:
        if mid in meters:
            s = read_power_series(f, meters[mid])
            found.append(mid)
            series.append(s)
        else:
            print(f"WARNING: mains meter {mid} 不存在，跳过（现有：{sorted(meters)}）")
    if not series:
        raise RuntimeError("没有可用的 mains 表（--mains-ids 与实际表号不符）。")
    # 多相/多表总负荷相加；两表时间戳须同时存在（inner），缺口策略统一在 prepare() 处理。
    df = pd.concat(series, axis=1, join="inner")
    return df.sum(axis=1), found


def prepare(f, house, mains_ids, kettle_id, out, mains_gap_min, kettle_gap_min):
    bg, bname = find_building(f, house)
    meters = meter_groups(f, bg)
    if not meters:
        raise RuntimeError(f"{bname} 下没有 meter* 组。")
    if kettle_id not in meters:
        raise RuntimeError(f"kettle meter {kettle_id} 不存在（实际表号：{sorted(meters)}）。"
                           "先跑 --list-meters 确认。")

    agg, used_mains = _combine_mains(f, meters, mains_ids)
    kettle = read_power_series(f, meters[kettle_id])

    # 短缺口补齐策略（可配置，均写入 data_spec.json 留痕）。
    mains_fill_limit = int(mains_gap_min * 60 / SECONDS_PER_SAMPLE)
    kettle_fill_limit = int(kettle_gap_min * 60 / SECONDS_PER_SAMPLE)
    n_kettle_nan_before = int(kettle.isna().sum())

    df = pd.DataFrame({"aggregate": agg, "target": kettle}).sort_index()
    df["aggregate"] = df["aggregate"].ffill(limit=mains_fill_limit)
    df["target"] = df["target"].fillna(0.0, limit=kettle_fill_limit)  # 关断即 0
    n_kettle_nan_after = int(df["target"].isna().sum())
    n_agg_nan_after = int(df["aggregate"].isna().sum())

    # 剩余 NaN（长缺口）→ 拆段，取最长连续段，不跨缺口拼接。
    mask = df[["aggregate", "target"]].notna().all(axis=1)
    df = df[mask]
    if len(df) == 0:
        raise RuntimeError("对齐后无任何有效行——请检查表号/采样周期是否匹配。")
    seg_boundaries = np.flatnonzero(~mask.to_numpy())
    seg_len = np.diff(np.concatenate([[0], seg_boundaries + 1, [len(df)]]))
    seg_start = np.concatenate([[0], seg_boundaries + 1])
    seg_end = np.concatenate([seg_boundaries + 1, [len(df)]])
    keep = int(np.argmax(seg_len))
    df = df.iloc[seg_start[keep]:seg_end[keep]]

    agg_v = df["aggregate"].to_numpy()
    tgt_v = df["target"].to_numpy()
    n_neg_agg = int((agg_v < 0).sum())
    n_neg_tgt = int((tgt_v < 0).sum())
    agg_v = np.clip(agg_v, 0.0, None)
    tgt_v = np.clip(tgt_v, 0.0, None)

    # 采样周期抽查（6s 数据允许轻微抖动）。
    med_dt = np.median(np.diff(df.index.astype(np.int64).to_numpy())) / 1e9
    if not (4 <= med_dt <= 8):
        print(f"WARNING: 中位采样间隔 {med_dt:.1f}s 偏离 6s，请确认数据为 6 秒采样。")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, aggregate=agg_v.astype(np.float32),
             target=tgt_v.astype(np.float32))

    spec = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "source_h5": str(f.filename),
        "building": bname,
        "mains_meter_ids_requested": mains_ids,
        "mains_meter_ids_used": used_mains,
        "kettle_meter_id": kettle_id,
        "unit": "W",
        "sample_period_sec": 6,
        "median_sample_gap_sec": round(float(med_dt), 3),
        "gap_policy": {
            "aggregate_ffill_min": mains_gap_min,
            "target_fill0_min": kettle_gap_min,
            "kettle_nan_dropped_after_policy": n_kettle_nan_after,
            "aggregate_nan_dropped_after_policy": n_agg_nan_after,
            "kettle_nan_before_policy": n_kettle_nan_before,
        },
        "segment_policy": "longest_contiguous_no_cross_gap",
        "segment_samples_kept": len(df),
        "time_range_iso": [str(df.index.min()), str(df.index.max())],
        "negative_clipped_to_zero": {"aggregate": n_neg_agg, "target": n_neg_tgt},
        "n_output": len(df),
    }
    spec_path = str(out_path.with_suffix(".data_spec.json"))
    Path(spec_path).write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OK: {out_path}  n={len(df)}  mains_used={used_mains}  "
          f"kettle={kettle_id}  时间范围 {spec['time_range_iso']}")
    print(f"缺口处理: kettle NaN {n_kettle_nan_before} → 策略后 {n_kettle_nan_after} → 分段剔除剩余；"
          f"负值 clip: agg {n_neg_agg} / target {n_neg_tgt}")
    print(f"数据口径留痕: {spec_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5-path", required=True, help="UK-DALE HDF5 文件路径")
    ap.add_argument("--house", type=int, default=1)
    ap.add_argument("--mains-ids", default="1,2", help="总负荷表号，逗号分隔（相加）")
    ap.add_argument("--kettle-meter-id", type=int, default=None, help="kettle 表号")
    ap.add_argument("--out", default="ukdale_prepared.npz", help="输出 npz 路径")
    ap.add_argument("--mains-gap-min", type=float, default=30.0,
                    help="aggregate 短缺口最大补全长（分钟，ffill）")
    ap.add_argument("--kettle-gap-min", type=float, default=5.0,
                    help="target 短缺口最大补全长（分钟，填 0）")
    ap.add_argument("--list-meters", action="store_true",
                    help="只打印 house 下各表号/样本数/时间范围，不生成 npz")
    args = ap.parse_args()

    if args.list_meters:
        with h5py.File(args.h5_path, "r") as f:
            cmd_list_meters(f, args.house)
        return
    if args.kettle_meter_id is None:
        raise SystemExit("需要 --kettle-meter-id（先跑 --list-meters 查看表号）。")

    mains_ids = [int(s) for s in args.mains_ids.split(",") if s.strip()]
    with h5py.File(args.h5_path, "r") as f:
        prepare(f, args.house, mains_ids, args.kettle_meter_id, args.out,
                args.mains_gap_min, args.kettle_gap_min)


if __name__ == "__main__":
    main()
