"""raw_to_recnpz.py — 周波录制 .raw（recorder.c 落盘）→ NPZ v1（edge_stream_test 回放格式）。

现场工作流（docs/EDGE_DEPLOYMENT.md §7.2）：
  终端双写（nilm_edge_push_packet + nilm_rec_packet 同点调用）→ .raw
  → 本脚本转换 → edge_stream_test.py --mode replay 回放验证 / 黄金集留存。

.raw v1 格式（小端）：
  头 64B：magic "NILMRAW1" | int32×4 [version, points, channels, 保留]
          | double×3 [fs, f0, 保留] | 保留 16B
  记录定长 append：wave float32 × points*channels + ts float64
  （截断尾记录按长度识别自动丢弃并警告）

用法：
  python scripts\\raw_to_recnpz.py --raw rec1.raw rec2.raw --out rec.npz
  python scripts\\raw_to_recnpz.py --raw rec.raw --out rec.npz --events events.csv
  python scripts\\raw_to_recnpz.py --raw rec.raw --out rec --split-hours 1   # 按 1h 分卷 rec_000.npz…

events.csv（可选，真值标注=受控切换实验）：
  每行：appliance,start,end   （epoch 秒；end 开区间）
  示例：kettle,1760000000.0,1760000180.5
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HDR_SZ = 64
MAGIC = b"NILMRAW1"


def read_raw(path):
    """读取单个 .raw → (records 结构化数组, header dict)。截断尾记录自动丢弃。"""
    raw = Path(path).read_bytes()
    if len(raw) < HDR_SZ or raw[:8] != MAGIC:
        raise SystemExit(f"{path}: 非 .raw v1 文件（magic 不符）")
    ver, points, channels = np.frombuffer(raw, "<i4", 3, offset=8)
    fs, f0 = np.frombuffer(raw, "<f8", 2, offset=24)
    if ver != 1:
        raise SystemExit(f"{path}: 版本 {ver} 不支持（应为 1）")
    dt = np.dtype([("wave", "<f4", (int(points) * int(channels),)), ("ts", "<f8")])
    body, rem = len(raw) - HDR_SZ, -1
    if body % dt.itemsize != 0:
        rem = body % dt.itemsize
        print(f"WARNING {path}: 尾部 {rem}B 非整记录（疑似截断/未 close），已丢弃", file=sys.stderr)
    recs = np.frombuffer(raw, dtype=dt, count=body // dt.itemsize, offset=HDR_SZ)
    return recs, {"version": int(ver), "points": int(points), "channels": int(channels),
                  "fs": float(fs), "f0": float(f0)}, rem


def main():
    ap = argparse.ArgumentParser(description=".raw v1 → NPZ v1 转换")
    ap.add_argument("--raw", nargs="+", required=True, help=".raw 文件（按时间序，可多个）")
    ap.add_argument("--out", required=True, help="输出 npz 路径（--split-hours 时为路径主干）")
    ap.add_argument("--events", default="", help="真值 CSV：appliance,start,end（epoch 秒，end 开区间）")
    ap.add_argument("--split-hours", type=float, default=0.0, help=">0：按此时长（h）分卷输出 <out>_000.npz…")
    args = ap.parse_args()

    parts, hdr0, src_meta = [], None, []
    for p in args.raw:
        recs, hdr, _ = read_raw(p)
        if hdr0 is None:
            hdr0 = hdr
        elif (hdr["points"], hdr["channels"]) != (hdr0["points"], hdr0["channels"]):
            raise SystemExit(f"{p}: points/channels 与首文件不一致，不能拼接")
        parts.append(recs)
        src_meta.append({"file": str(p), "n_packets": int(len(recs))})
        print(f"  读取 {p}: {len(recs)} 包（points={hdr['points']} channels={hdr['channels']}）")
    recs = np.concatenate(parts) if len(parts) > 1 else parts[0]
    n = len(recs)
    if n == 0:
        raise SystemExit("无有效记录")
    ts = recs["ts"]
    wave = recs["wave"]

    # 质量摘要
    period = hdr0["points"] / hdr0["fs"]
    d = np.diff(ts)
    dropouts = int(np.sum(d > period * 1.05 + 1e-9))
    mono = bool(np.all(d >= 0))
    vrms = [None, None, None]
    if hdr0["channels"] == 6:
        w = wave[:: max(1, n // 1200)].reshape(-1, hdr0["channels"]).astype(np.float64)
        vrms = [round(float(np.sqrt((w[:, k * 2] ** 2).mean())), 1) for k in range(3)]
    span = float(ts[-1] - ts[0])
    print(f"合计 {n} 包 / {span / 3600:.2f}h | 时间戳单调={mono} | 疑似丢帧 {dropouts} 处"
          f"（>{period * 1.05 * 1000:.0f}ms）| Vrms {vrms}")
    if not mono:
        print("WARNING: 时间戳非单调——回放引擎按契约要求单调（D4 重推须带原时间戳）", file=sys.stderr)

    # 真值展开
    truth = {}
    if args.events:
        with open(args.events, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or row[0].strip().startswith("#") or row[0].strip().lower() == "appliance":
                    continue
                name, s, e = row[0].strip(), float(row[1]), float(row[2])
                flag = ((ts >= s) & (ts < e)).astype(np.int8)
                if name in truth:
                    truth[name] = np.maximum(truth[name], flag)
                else:
                    truth[name] = flag
                print(f"  事件标注 {name}: [{s}, {e}) → {int(flag.sum())} 包 ON")

    meta = {
        "format": "nilm-edge wave recording v1",
        "converted_from_raw": True,
        "source_raw": src_meta,
        "n_packets": int(n),
        "span_sec": span,
        "points_per_packet": hdr0["points"],
        "channels": hdr0["channels"],
        "channel_order": "[uA,iA,uB,iB,uC,iC]" if hdr0["channels"] == 6 else None,
        "fs": hdr0["fs"],
        "f0": hdr0["f0"],
        "ts_monotonic": mono,
        "dropouts_gt_period": dropouts,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    def write_npz(path, idx, note=""):
        kw = {"wave": wave[idx].astype(np.float32), "ts": ts[idx].astype(np.float64),
              "meta": json.dumps({**meta, "n_packets": int(len(idx)), "note": note},
                                 ensure_ascii=False)}
        for name, flag in truth.items():
            kw[f"truth_on_{name}"] = flag[idx]
        np.savez(path, **kw)
        print(f"  写出 {path}（{len(idx)} 包）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.split_hours and args.split_hours > 0:
        if truth:
            print("WARNING: --split-hours 与 --events 同用时真值按各卷独立展开（已按包切片，语义不变）")
        win = args.split_hours * 3600.0
        t0 = float(ts[0])
        bids = np.floor((ts - t0) / win).astype(np.int64)
        for b in range(int(bids[0]), int(bids[-1]) + 1):
            idx = np.nonzero(bids == b)[0]
            if len(idx):
                write_npz(out.with_name(f"{out.stem}_{b:03d}.npz"), idx,
                          note=f"split {args.split_hours}h #{b}")
    else:
        write_npz(out, np.arange(n))
    print("转换完成。回放：python scripts/edge_stream_test.py --mode replay "
          f"--wave-file {out}{'_000.npz…' if args.split_hours and args.split_hours > 0 else ''} "
          "--bundle <model.bin> --out-dir reports/edge_s2")


if __name__ == "__main__":
    main()
