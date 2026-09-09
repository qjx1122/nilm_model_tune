"""UK-DALE HDF5 → aggregate/target NPZ 制备脚本（REPORT_TEST.md 调参方案·阶段 0 规格）。

支持两类 meter 布局（自动识别）：
  A. 直接数据集：meter 组内 /building{house}/elec/meter{id}/power 形状 (N,2)
     [时间戳(秒或纳秒), 功率(W)]
  B. NILMTK pandas 表：meter 组内含 _i_table/table（pd.read_hdf 读取），
     列名为 ('power','active') 或 ('power','apparent') —— 自动优先 active，
     找不到才用 apparent（并 WARNING + 留痕）。时间戳 index 自动按秒/纳秒换算。

数据规格（写进 data_spec.json 留痕，作为以后所有 KPI 的口径依据）：
0. 各表先 resample 到统一 6 秒网格（bin 内均值）再对齐——真实 UK-DALE 各表
   采样时刻有秒级相位差（如 meter1 在 :15 秒、meter10 在 :18 秒），精确时间戳
   join 几乎拼不上（执行实录 9：n=345）。已在网格上的数据为恒等变换。
1. mains 各表（--mains-ids）按时间对齐相加；短缺口（<= mains-gap-min 分钟）
   整段 ffill，更长缺口整段剔除。
2. 电器子表（--appliance-meter-id，兼容别名 --kettle-meter-id）：短缺口
   （<= appliance-gap-min 分钟）整段 ffill
   （前值：壶 ~99% 时间关断 → 前值=0 与「关断即 0」语义一致；煮沸中掉线
   保持 ~2300W 不断流——v4 曾整段补 0，把煮沸事件切成 35s 碎片、evt/day
   翻倍 5.56→12.55，实录 12），更长缺口整段剔除。
3. 剩余 NaN（长缺口）行剔除，其余按时间顺序全量拼接（跨缺口接缝计数留痕；
   npz 本不带时间戳，等效连续流。实录 10：真实 UK-DALE 缺口密布，最长无缺口
   段仅 ~3.4h，"只取最长连续段"策略已废弃；段统计须在统一索引空间计算）。
4. 负功率视为测量伪迹 clip 到 0（计数并留痕）；单位统一 W。

输出：
    <out>.npz             {aggregate, target} float32 一维等长
    <out 同名>.data_spec.json  口径元数据（来源/表号/功率类型/时间范围/缺口策略/样本数）

用法（示例，NILMTK 格式 ukdale.h5）：
    python scripts\\prepare_ukdale.py --h5-path D:\\datasets\\ukdale.h5 --list-meters
    # House1 kettle（项目冻结口径 v5，旧命令不变）：
    python scripts\\prepare_ukdale.py --h5-path D:\\datasets\\ukdale.h5 ^
        --mains-ids 1 --kettle-meter-id 10 ^
        --out D:\\datasets\\ukdale_prepared.npz
    # 其他 house / 电器（泛化口径，实录 22）：
    python scripts\\prepare_ukdale.py --h5-path D:\\datasets\\ukdale.h5 --house 1 ^
        --mains-ids 1 --appliance-meter-id 6 --appliance dish_washer ^
        --out D:\\datasets\\ukdale_dw.npz
    （mains 若为 apparent、电器表为 active，脚本自动处理并在 data_spec.json 留痕；
    House1 地面真相：mains 只有 meter1 单表，meter2 系锅炉回路，见 REPORT_TEST.md 实录 8）
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
POWER_TYPES = ("active", "apparent")


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


def _normalize_ts_index(idx):
    """把时间戳 index 转成 UTC DatetimeIndex。

    兼容三类输入：① int64 秒/纳秒戳；② naive datetime；③ tz-aware datetime
    （真实 NILMTK ukdale.h5 的 index 自带 Europe/London 时区）。
    注意：np.issubdtype 无法解析 tz-aware dtype，会抛
    TypeError: Cannot interpret 'datetime64[..., tz]' —— 必须先处理 DatetimeIndex。
    """
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            return idx.tz_localize("UTC")
        return idx.tz_convert("UTC")
    if not isinstance(idx, pd.Index):
        idx = pd.Index(idx)
    try:
        is_int = np.issubdtype(idx.dtype, np.integer)
    except TypeError:
        # pandas 扩展 dtype（如 tz-aware 经普通 Index 包裹）→ 直接 to_datetime
        return pd.to_datetime(idx, utc=True)
    if not is_int:
        return pd.to_datetime(idx, utc=True)
    arr = idx.to_numpy()
    if len(arr) and arr.max() > 1e14:  # 纳秒
        arr = arr / 1e9
    return pd.to_datetime(arr, unit="s", utc=True)


def _table_power_columns(df):
    """从 pd.read_hdf 结果里找出可用功率列：返回 (power_type -> Series 或列名)。"""
    cols = list(df.columns)
    avail = {}
    for pt in POWER_TYPES:
        try:
            if isinstance(df.columns, pd.MultiIndex) and ("power", pt) in df.columns:
                avail[pt] = df[("power", pt)]
            else:
                # 单层列：找 'power' 或该类型名
                cand = [c for c in cols if str(c).lower() == pt]
                if cand:
                    avail[pt] = df[cand[0]]
        except Exception:
            continue
    return avail


def read_power_series(f, meter_group, prefer="active"):
    """读取 meter 的功率序列 → (Series[UTC], 实际功率类型)。

    布局 B（NILMTK pandas 表）：pd.read_hdf 成功即用；读取后的处理错误直接抛出
    （不静默吞，便于定位真因）。仅当 read_hdf 本身失败（非 pandas 表路径）时
    回退布局 A（直接数据集 power/power_series (N,2)）。
    """
    # 布局 B：pandas HDFStore 表
    try:
        df = pd.read_hdf(f.filename, meter_group.name)
    except Exception:
        df = None  # 非 pandas 表路径 → 走布局 A
    if df is not None and hasattr(df, "columns"):
        if len(df) == 0:
            raise RuntimeError(f"{meter_group.name} 是空 pandas 表")
        avail = _table_power_columns(df)
        if not avail:
            raise RuntimeError(f"{meter_group.name} pandas 表无法识别功率列，"
                               f"列={list(df.columns)[:8]}")
        pt = prefer if prefer in avail else next(iter(avail))
        if pt != prefer:
            print(f"WARNING: {meter_group.name} 无 {prefer} 列，使用 {pt}"
                  f"（可用列：{list(avail)}）")
        s = avail[pt]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.astype(np.float64).copy()
        s.index = _normalize_ts_index(df.index)
        s = s[~s.index.duplicated(keep="first")].sort_index()
        s = s[np.isfinite(s.values)]
        return s, pt

    # 布局 A：直接数据集 (N,2)
    keys = list(meter_group.keys())
    ds_name = next((k for k in ("power", "power_series") if k in meter_group), None)
    if ds_name is None:
        raise RuntimeError(f"meter 组 {meter_group.name} 不是 pandas 表也没有 "
                           f"power/power_series，实际键：{keys}。请先跑 --list-meters 核对。")
    arr = np.asarray(meter_group[ds_name][()], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError(f"{meter_group.name}/{ds_name} 形状 {arr.shape} 不是 (N,2)"
                           "（时间戳, 功率），布局与本脚本契约不符。")
    ts, val = arr[:, 0], arr[:, 1]
    idx = _normalize_ts_index(pd.Index(ts.astype(np.int64) if np.all(ts == ts.astype(np.int64)) else ts))
    s = pd.Series(val, index=idx)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s = s[np.isfinite(s.values)]
    return s, "active"  # 契约 A 无法判断，默认标注 active（历史行为）


def _to_6s_grid(s):
    """把功率序列归一化到统一 6 秒网格（bin 内均值）。

    bin 划分显式以 epoch 为原点（origin="epoch"，不依赖 pandas 默认值随版本
    变化），左右表落在同一网格上；已在网格上的数据为恒等变换（每 bin 恰 1 个
    样本，均值即原值）。
    注意：只用于 prepare 对齐，--list-meters 仍显示原始 n_samples。
    """
    return s.resample(f"{SECONDS_PER_SAMPLE}s", origin="epoch").mean()


def _bridge_short_gaps(s, limit, fill_value=None):
    """整段桥接：长度 <= limit 的 NaN 缺口段整段补值，更长缺口整段保留 NaN。

    fill_value=None → 沿用缺口前最后值（ffill；aggregate 与 kettle 均用此
    语义，见规格第 2 条）。返回 (新 Series, 桥接格数)。
    注意：不能用 fillna(value, limit=N) 表达本语义——其 limit 是全轴总限额
    （pandas 文档：method 未指定时按整轴计），真实数据 240 万缺口格只被填了
    49 格（实录 11）；ffill(limit=N) 则是每段只填头部 N 格。两者皆非整段语义。
    """
    is_na = s.isna()
    if not is_na.any():
        return s, 0
    run_id = (is_na != is_na.shift()).cumsum()
    run_len = is_na.groupby(run_id).transform("sum")  # NaN 段长度（非 NaN 段为 0）
    short = is_na & (run_len <= limit)
    if fill_value is None:
        s = s.mask(short, s.ffill())
    else:
        s = s.mask(short, fill_value)
    return s, int(short.sum())


def _meter_summary(f, meter_group):
    try:
        s, pt = read_power_series(f, meter_group)
        return {"id": int(meter_group.name.rstrip("/").split("meter")[-1]),
                "n": len(s), "start": str(s.index.min()), "end": str(s.index.max()),
                "power_type": pt}
    except Exception as e:
        return {"id": int(meter_group.name.rstrip("/").split("meter")[-1]),
                "error": str(e)}


def cmd_list_meters(f, house):
    bg, name = find_building(f, house)
    meters = meter_groups(f, bg)
    if not meters:
        raise RuntimeError(f"{name} 下没有 meter* 组（elec 内键：{list(bg['elec'].keys()) if 'elec' in bg else list(bg.keys())}）。")
    print(f"house: {name} | meters: {len(meters)}")
    print(f"{'id':>9} {'n_samples':>10}  {'start':<22} {'end':<22} {'type':<9}")
    for mid in sorted(meters):
        m = _meter_summary(f, meters[mid])
        if "error" in m:
            print(f"meter {mid:<3}  读取失败: {m['error']}")
        else:
            print(f"meter {mid:<3} {m['n']:>10}  {m['start']:<22} {m['end']:<22} {m['power_type']:<9}")
    print("\n提示：功率类型 active=有功（NILM 分解应用），apparent=视在（总表常见）。"
          "kettle 应以 metadata（parse_nilmtk_metadata.py）为准。")


def _combine_mains(f, meters, mains_ids):
    found, series, types = [], [], []
    for mid in mains_ids:
        if mid in meters:
            s, pt = read_power_series(f, meters[mid])
            found.append(mid)
            series.append(_to_6s_grid(s))
            types.append(pt)
        else:
            print(f"WARNING: mains meter {mid} 不存在，跳过（现有：{sorted(meters)}）")
    if not series:
        raise RuntimeError("没有可用的 mains 表（--mains-ids 与实际表号不符）。")
    # 多相/多表总负荷相加（已统一 6s 网格，inner 即网格交集）。
    # 缺口策略统一在 prepare() 处理。
    # min_count=1：整行 NaN（各表全缺）须保持 NaN 交给缺口策略——
    # 默认 skipna 会把缺口静默写成 0W 假零（实录 10，回归测试桥接断言拦截）。
    df = pd.concat(series, axis=1, join="inner")
    return df.sum(axis=1, min_count=1), found, types


def prepare(f, house, mains_ids, appliance_id, out, mains_gap_min,
            appliance_gap_min, appliance="kettle"):
    """电器无关制备；appliance 仅为标签/留痕（默认 kettle = v5 冻结口径，行为不变）。"""
    bg, bname = find_building(f, house)
    meters = meter_groups(f, bg)
    if not meters:
        raise RuntimeError(f"{bname} 下没有 meter* 组。")
    if appliance_id not in meters:
        raise RuntimeError(f"appliance meter {appliance_id} 不存在（实际表号：{sorted(meters)}）。"
                           "先跑 --list-meters 确认。")

    agg, used_mains, mains_types = _combine_mains(f, meters, mains_ids)
    appl, appl_type = read_power_series(f, meters[appliance_id])
    appl = _to_6s_grid(appl)
    print(f"对齐: mains 网格点 {len(agg)} / {appliance} 网格点 {len(appl)}"
          f"（6s 网格 resample 后）")

    # 短缺口补齐策略（可配置，均写入 data_spec.json 留痕）。
    mains_fill_limit = int(mains_gap_min * 60 / SECONDS_PER_SAMPLE)
    appl_fill_limit = int(appliance_gap_min * 60 / SECONDS_PER_SAMPLE)
    n_appl_nan_before = int(appl.isna().sum())

    df = pd.DataFrame({"aggregate": agg, "target": appl}).sort_index()
    # 整段桥接（v4，实录 11）：短缺口整段补值，长缺口整段保留 NaN 交给剔除。
    df["aggregate"], n_agg_bridged = _bridge_short_gaps(df["aggregate"], mains_fill_limit)
    df["target"], n_appl_bridged = _bridge_short_gaps(df["target"], appl_fill_limit)  # ffill：关断时前值=0；运行中不断流
    n_appl_nan_after = int(df["target"].isna().sum())
    n_agg_nan_after = int(df["aggregate"].isna().sum())

    # 剩余 NaN（长缺口）→ 剔除后全量拼接（接缝计数留痕）。
    # 回归（实录 10）：v2 曾在 df[mask] 过滤后仍用过滤前的位置编号算段长，
    # 末段被低估"剔除行数-1"，选段不可信；现统一在过滤前索引空间计算。
    mask = df[["aggregate", "target"]].notna().all(axis=1).to_numpy()
    if not mask.any():
        raise RuntimeError("对齐后无任何有效行——请检查表号/采样周期是否匹配。")
    d = np.diff(np.concatenate([[0], mask.astype(np.int8)]))
    seg_start = np.flatnonzero(d == 1)
    seg_end = np.flatnonzero(d == -1)
    if len(seg_end) < len(seg_start):
        seg_end = np.append(seg_end, len(mask))
    seg_len = seg_end - seg_start
    n_segments = int(len(seg_start))
    n_breaks = n_segments - 1
    largest_seg = int(seg_len.max())
    n_union = int(len(mask))
    df = df[mask]  # 保留全部有效行，按时间顺序拼接

    agg_v = df["aggregate"].to_numpy()
    tgt_v = df["target"].to_numpy()
    n_neg_agg = int((agg_v < 0).sum())
    n_neg_tgt = int((tgt_v < 0).sum())
    agg_v = np.clip(agg_v, 0.0, None)
    tgt_v = np.clip(tgt_v, 0.0, None)

    # 采样周期抽查（只统计段内间隔，跨缺口接缝不计入；Timedelta 口径
    # 避免 asi8 的 ns/us 单位陷阱——pandas 3 的 asi8 随 index 单位变）。
    if len(df) > 1:
        _is_seg_start = np.zeros(len(df), dtype=bool)
        _is_seg_start[0] = True
        _is_seg_start[np.cumsum(seg_len)[:-1]] = True  # 各段在过滤后数组中的起点
        _secs = df.index.to_series().diff().dt.total_seconds().to_numpy()
        med_dt = float(np.median(_secs[~_is_seg_start]))
    else:
        med_dt = 6.0
    if not (4 <= med_dt <= 8):
        print(f"WARNING: 中位采样间隔 {med_dt:.1f}s 偏离 6s，请确认数据为 6 秒采样。")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, aggregate=agg_v.astype(np.float32),
             target=tgt_v.astype(np.float32))

    spec = {
        "schema_version": 5,  # v5：kettle 短缺口 ffill（实录 12）；v4：整段桥接（实录 11）；v3：全量拼接（实录 10）；v2：6s 网格（实录 9）
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "source_h5": str(f.filename),
        "building": bname,
        "mains_meter_ids_requested": mains_ids,
        "mains_meter_ids_used": used_mains,
        "mains_power_types_used": mains_types,
        "appliance": appliance,
        "appliance_meter_id": appliance_id,
        "appliance_power_type_used": appl_type,
        "resample_policy": "mean_to_6s_grid_epoch_origin",
        "unit": "W",
        "sample_period_sec": 6,
        "median_sample_gap_sec": round(float(med_dt), 3),
        "gap_policy": {
            "aggregate_ffill_min": mains_gap_min,
            "target_fill0_min": appliance_gap_min,  # legacy 键名（v1-v5 沿用）
            "appliance_gap_min": appliance_gap_min,
            "aggregate_policy": "ffill_whole_gaps_le_threshold",
            "target_policy": "ffill_whole_gaps_le_threshold",
            "long_gap_policy": "drop_whole_gap",
            "aggregate_cells_bridged": n_agg_bridged,
            "appliance_cells_bridged": n_appl_bridged,
            "appliance_nan_dropped_after_policy": n_appl_nan_after,
            "aggregate_nan_dropped_after_policy": n_agg_nan_after,
            "appliance_nan_before_policy": n_appl_nan_before,
        },
        "segment_policy": "concat_all_segments_logged_breaks",
        "n_segments": n_segments,
        "n_concat_breaks": n_breaks,
        "largest_segment_samples": largest_seg,
        "union_grid_samples": n_union,
        "dropped_gap_samples": n_union - len(df),
        "segment_samples_kept": len(df),
        "time_range_iso": [str(df.index.min()), str(df.index.max())],
        "negative_clipped_to_zero": {"aggregate": n_neg_agg, "target": n_neg_tgt},
        "n_output": len(df),
    }
    if appliance == "kettle":
        # v5 冻结口径 legacy 键（向后兼容旧读取方与既有测试；非 kettle 不产生）
        spec["kettle_meter_id"] = appliance_id
        spec["kettle_power_type_used"] = appl_type
        spec["gap_policy"]["kettle_nan_dropped_after_policy"] = n_appl_nan_after
        spec["gap_policy"]["kettle_nan_before_policy"] = n_appl_nan_before
        spec["gap_policy"]["kettle_cells_bridged"] = n_appl_bridged

    spec_path = str(out_path.with_suffix(".data_spec.json"))
    Path(spec_path).write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OK: {out_path}  n={len(df)}  mains_used={used_mains}  "
          f"{appliance}={appliance_id}  时间范围 {spec['time_range_iso']}")
    print(f"缺口处理: 桥接 agg {n_agg_bridged} 格 / {appliance} {n_appl_bridged} 格"
          f"（均 ffill 前值整段桥接）；长缺口整段剔除 {n_union - len(df)} 格"
          f"（跨缺口拼接 {n_breaks} 处，最大连续段 {largest_seg} 样本 ≈ {largest_seg / 14400:.1f} 天）；"
          f"负值 clip: agg {n_neg_agg} / target {n_neg_tgt}")
    print(f"数据口径留痕: {spec_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5-path", required=True, help="UK-DALE HDF5 文件路径")
    ap.add_argument("--house", type=int, default=1)
    ap.add_argument("--mains-ids", default="1",
                    help="总负荷表号，逗号分隔（相加）；House1 只有 meter1 单表，"
                         "其他 house 先 --list-meters + parse_nilmtk_metadata.py 核对")
    ap.add_argument("--appliance", default="kettle",
                    help="电器名（标签，写入 data_spec 与输出；默认 kettle=项目冻结口径）")
    ap.add_argument("--appliance-meter-id", type=int, default=None,
                    help="电器子表号（通用名；先 --list-meters / parse_nilmtk_metadata 确认）")
    ap.add_argument("--kettle-meter-id", type=int, default=None,
                    help="（兼容别名）等价 --appliance-meter-id，v5 冻结命令继续可用")
    ap.add_argument("--out", default="ukdale_prepared.npz", help="输出 npz 路径")
    ap.add_argument("--mains-gap-min", type=float, default=30.0,
                    help="aggregate 短缺口整段 ffill 阈值（分钟）")
    ap.add_argument("--appliance-gap-min", type=float, default=None,
                    help="电器短缺口整段 ffill 阈值（分钟，默认 5.0）")
    ap.add_argument("--kettle-gap-min", type=float, default=None,
                    help="（兼容别名）等价 --appliance-gap-min")
    ap.add_argument("--list-meters", action="store_true",
                    help="只打印 house 下各表号/样本数/时间范围，不生成 npz")
    args = ap.parse_args()

    if args.list_meters:
        with h5py.File(args.h5_path, "r") as f:
            cmd_list_meters(f, args.house)
        return

    # 通用名 + 兼容别名解析（冲突即报错，不接受静默覆盖）
    if (args.appliance_meter_id is not None and args.kettle_meter_id is not None
            and args.appliance_meter_id != args.kettle_meter_id):
        raise SystemExit("--appliance-meter-id 与 --kettle-meter-id 只能给一个（或相同的值）。")
    appliance_id = (args.appliance_meter_id if args.appliance_meter_id is not None
                    else args.kettle_meter_id)
    if appliance_id is None:
        raise SystemExit("需要 --appliance-meter-id（旧名 --kettle-meter-id 亦可；"
                         "先跑 --list-meters 查看表号）。")
    if (args.appliance_gap_min is not None and args.kettle_gap_min is not None
            and args.appliance_gap_min != args.kettle_gap_min):
        raise SystemExit("--appliance-gap-min 与 --kettle-gap-min 只能给一个（或相同的值）。")
    appliance_gap_min = (args.appliance_gap_min if args.appliance_gap_min is not None
                         else (args.kettle_gap_min if args.kettle_gap_min is not None else 5.0))

    mains_ids = [int(s) for s in args.mains_ids.split(",") if s.strip()]
    with h5py.File(args.h5_path, "r") as f:
        prepare(f, args.house, mains_ids, appliance_id, args.out,
                args.mains_gap_min, appliance_gap_min, appliance=args.appliance)


if __name__ == "__main__":
    main()
