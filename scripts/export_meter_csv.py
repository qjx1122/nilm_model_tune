"""export_meter_csv.py — UK-DALE h5 按表导出 CSV（每表一个文件：时间 + 功率）。

用法（PowerShell）：
  python scripts\\export_meter_csv.py --h5-path D:\\datasets\\ukdale.h5 --house 5 --meter 18
  python scripts\\export_meter_csv.py --h5-path D:\\datasets\\ukdale.h5 --house 5 --meter 18 22 23
  python scripts\\export_meter_csv.py --h5-path D:\\datasets\\ukdale.h5 --house 1 --meter 1 `
      --start "2015-04-01 00:00:00" --end "2015-04-03 00:00:00" --out-dir D:\\exports
  python scripts\\export_meter_csv.py --h5-path D:\\datasets\\ukdale.h5 --house 1 --meter 1 2 3 `
      --common-span          # 仅导出各表公共时间段（数据范围交集）

输出：
  <out-dir>\\house_{house}_meter_{meter}.csv（两列：time, power_w）
  - time：UTC（ISO "YYYY-MM-DD HH:MM:SS"，源索引为 Europe/London 时区，与 prepare 管道
    同口径统一转 UTC）；power_w：功率（W）。
  - 默认重采样到 6 秒网格（bin 均值，与 prepare 口径一致；--resample-sec 0 = 导出原始采样）。
  - 非有限值（NaN/inf）剔除；重采样产生的缺口格剔除——两者计数均打印。
  - 负值**不 clip**（与训练制备不同：本脚本忠实导出原始数据，负值计数仅提示）。

时间段语义：
  - --start / --end：UTC 时间（"YYYY-MM-DD HH:MM:SS" 或带时区偏移的 ISO；naive 按 UTC
    解析，与导出列一致）；可只给一端；**边界含端点**；
  - --common-span（多表）：仅导出各表**数据范围的交集** [max(各表首样本), min(各表末样本)]，
    再与 --start/--end 求交；交集为空则报错并列出各表范围；单表时提示无效（=全量）；
  - 切片发生在网格化**之后**（边界格为整格 bin 均值；--resample-sec 0 时为逐样本精确过滤）。

读表语义与 scripts/prepare_ukdale.py 完全同源（import 其 find_building/meter_groups/
read_power_series）：NILMTK pandas 表布局，功率列自动优先 active，无 active 用 apparent
（WARNING 提示）；--power-type 可强制指定。

依赖：h5py pandas numpy（与 prepare 相同）。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 仓库根（runlog）
sys.path.insert(0, str(Path(__file__).resolve().parent))       # scripts/（prepare_ukdale）

import h5py
import pandas as pd

from runlog import setup_run_log, default_log_path
from prepare_ukdale import find_building, meter_groups, read_power_series


def _parse_ts(x, opt):
    """解析用户时间参数 → UTC Timestamp（naive 按 UTC，与导出列一致）。"""
    try:
        t = pd.Timestamp(x)
    except Exception:
        raise SystemExit(f"{opt} 无法解析：{x!r}（示例 '2017-07-14 02:40:00'，含时区偏移的 ISO 亦可）")
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


p = argparse.ArgumentParser(description="按 house+表号导出 UK-DALE h5 表数据为 CSV（时间+功率）")
p.add_argument("--h5-path", required=True, help="UK-DALE HDF5 文件路径")
p.add_argument("--house", type=int, required=True, help="house 号（building 组）")
p.add_argument("--meter", type=int, nargs="+", required=True,
               help="表号（可多个，如 --meter 18 22 23）")
p.add_argument("--out-dir", default=".", help="输出目录（缺省当前目录）")
p.add_argument("--start", default="", help="起始时间（UTC，含端点；可只给一端）")
p.add_argument("--end", default="", help="结束时间（UTC，含端点；可只给一端）")
p.add_argument("--common-span", action="store_true",
               help="多表导出时仅导出各表公共时间段（数据范围交集，再与 --start/--end 求交）")
p.add_argument("--power-type", choices=["auto", "active", "apparent"], default="auto",
               help="功率列选择（auto=优先 active，无则 apparent 并提示；缺省 auto）")
p.add_argument("--resample-sec", type=int, default=6,
               help="重采样网格秒数（bin 均值，与 prepare 口径一致）；0=原始采样（缺省 6）")
p.add_argument("--float-format", default="%.3f", help="功率列格式（缺省 %.3f；传空串=完整精度）")
p.add_argument("--no-log", action="store_true", help="禁用控制台输出留痕")
args = p.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
setup_run_log(default_log_path("export_meter_csv"), enabled=not args.no_log)

prefer = "active" if args.power_type == "auto" else args.power_type
if args.power_type != "auto":
    print(f"NOTE: 强制功率类型 {args.power_type}（表内无该列时报错）")

with h5py.File(args.h5_path, "r") as f:
    bg, bname = find_building(f, args.house)
    meters = meter_groups(f, bg)
    if not meters:
        raise SystemExit(f"{bname} 下没有 meter* 组。先用 "
                         f"python scripts\\prepare_ukdale.py --h5-path {args.h5_path} "
                         f"--house {args.house} --list-meters 查看可用表。")
    missing = [m for m in args.meter if m not in meters]
    if missing:
        raise SystemExit(f"表号 {missing} 不存在。{bname} 可用表：{sorted(meters)}")

    print(f"h5: {args.h5_path} | house: {bname} | 表: {sorted(args.meter)} | "
          f"功率类型: {args.power_type} | 网格: "
          f"{f'{args.resample_sec}s' if args.resample_sec else '原始采样'} | 输出: {out_dir}")

    # 先读全部表序列（--common-span 需先知晓各表数据范围）
    loaded = {}
    for mid in args.meter:
        loaded[mid] = read_power_series(f, meters[mid], prefer=prefer)

    # ── 导出范围解析：--start/--end ∩ --common-span ──
    start_t = _parse_ts(args.start, "--start") if args.start else None
    end_t = _parse_ts(args.end, "--end") if args.end else None
    span_src = []
    if args.common_span:
        if len(args.meter) > 1:
            lo = max(s.index.min() for s, _ in loaded.values())
            hi = min(s.index.max() for s, _ in loaded.values())
            if lo > hi:
                spans = {mid: f"[{s.index.min()} → {s.index.max()}]"
                         for mid, (s, _) in loaded.items()}
                raise SystemExit(f"--common-span：各表数据范围无公共时间段：{spans}")
            start_t = lo if start_t is None else max(start_t, lo)
            end_t = hi if end_t is None else min(end_t, hi)
            span_src.append(f"--common-span（各表交集 [{lo} → {hi}]）")
        else:
            print("NOTE: --common-span 仅多表导出时生效（单表导出全量）")
    if start_t is not None and end_t is not None and start_t > end_t:
        raise SystemExit(f"导出范围为空：start {start_t} > end {end_t}"
                         + (f"（与 --common-span 交集后）" if args.common_span else ""))
    if start_t is not None or end_t is not None:
        if not span_src:
            span_src.append("--start/--end")
        print(f"导出范围（含端点）: [{start_t if start_t is not None else '(-∞)'} → "
              f"{end_t if end_t is not None else '(+∞)'}]（来源：{' ∩ '.join(span_src)}）")

    for mid in args.meter:
        s, pt = loaded[mid]
        full_lo, full_hi = s.index.min(), s.index.max()
        n_raw = len(s)                       # read_power_series 已剔除非有限值
        if args.resample_sec > 0:
            s = s.resample(f"{args.resample_sec}s", origin="epoch").mean()
        n_grid = len(s)
        if start_t is not None and end_t is not None:
            s = s.loc[start_t:end_t]         # 边界含端点
        elif start_t is not None:
            s = s.loc[start_t:]
        elif end_t is not None:
            s = s.loc[:end_t]
        n_rng = len(s)
        n_gap = n_rng - int(s.dropna().size)  # 范围内网格上无样本的缺口格
        s = s.dropna()
        if s.empty:
            raise SystemExit(f"表 {mid} 在导出范围内无数据（范围 "
                             f"[{start_t} → {end_t}]，该表数据 [{full_lo} → {full_hi}]）")
        neg = int((s < 0).sum())
        out = out_dir / f"house_{args.house}_meter_{mid}.csv"
        df = pd.DataFrame({"time": s.index.strftime("%Y-%m-%d %H:%M:%S"),
                           "power_w": s.values})
        kw = {"index": False, "encoding": "utf-8"}
        if args.float_format:
            kw["float_format"] = args.float_format
        df.to_csv(out, **kw)
        dt = s.index.to_series().diff().median()
        if args.resample_sec:
            src = f"原始样本 {n_raw} → 网格 {n_grid}"
            if n_grid != n_rng:
                src += f" → 范围内 {n_rng}"
            src += f"（缺口格 {n_gap}）"
        else:
            src = f"原始样本 {n_raw}" + (f" → 范围内 {n_rng}" if n_raw != n_rng else "")
        print(f"  ✓ house_{args.house}_meter_{mid}.csv | {len(s)} 行 | {pt} | "
              f"{s.index.min()} → {s.index.max()} | 中位间隔 {dt.total_seconds():g}s | "
              f"功率 min/mean/max = {s.min():.1f}/{s.mean():.1f}/{s.max():.1f} W | "
              f"负值 {neg} 格（不 clip，忠实导出）| {src} | {out.stat().st_size / 1e6:.1f} MB")

print("导出完成" + (f"（{len(args.meter)} 个表）" if len(args.meter) > 1 else "") +
      f" → {out_dir}")
