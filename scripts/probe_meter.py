"""单 meter pandas 表原始真相探查（零过滤 dump，与 read_power_series 视图对照）。

背景（执行实录 24）：House2 kettle 制备打印「网格点 3,377,557」大于 meter8
时间跨度的理论上限（2013-04-16→10-10 ≈ 2,539,200 格），且并集算术
（n+剔除=3,377,557）显示 target 侧存在 mains 网格之外的时间戳（~172 格在
mains 起点 2013-02-17 16:17:34 之前、1 格在终点之后）——提示表内可能夹带
mains 覆盖之前的 finite 杂散行；但 --list-meters 曾显示 meter8 起点为
2013-04-16，两者互斥（沙箱孪生复现已证实：若有杂散行，list-meters 也会
显示杂散时间）。本脚本直接 dump 表原始 index/value 统计以定谳。

用法：
    python scripts\\probe_meter.py --h5-path <ukdale.h5> --house 2 --meter 8 ^
        --cutoff "2013-04-16 21:18:09"
"""
import argparse

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5-path", required=True)
    p.add_argument("--house", type=int, required=True)
    p.add_argument("--meter", type=int, required=True)
    p.add_argument("--cutoff", default=None,
                   help="对照截断时间戳（naive 按表时区解释）：统计其前后行数与截断前明细")
    a = p.parse_args()

    key = f"/building{a.house}/elec/meter{a.meter}"
    df = pd.read_hdf(a.h5_path, key)
    idx = df.index
    val = df.iloc[:, 0].astype("float64")

    print(f"key={key}  rows={len(df)}  cols={[str(c) for c in df.columns]}")
    print(f"index: tz={getattr(idx, 'tz', None)}  min={idx.min()}  max={idx.max()}")
    print(f"NaT时间戳行={int(pd.isna(idx).sum())}  重复时间戳={int(idx.duplicated().sum())}"
          f"  单调递增={bool(idx.is_monotonic_increasing)}")
    finite = np.isfinite(val)
    print(f"值: NaN={int(np.isnan(val).sum())}  inf={int(np.isinf(val).sum())}"
          f"  finite={int(finite.sum())}  min={np.nanmin(val):.1f}  max={np.nanmax(val):.1f}")

    if a.cutoff:
        ts = pd.Timestamp(a.cutoff)
        if getattr(idx, "tz", None) is not None and ts.tzinfo is None:
            ts = ts.tz_localize(idx.tz)
        pre = idx < ts
        print(f"cutoff={ts}  之前行数={int(pre.sum())}  之后行数={int((~pre).sum())}")
        if pre.any():
            print("截断点之前前 5 行：")
            print(df[pre].head(5).to_string())
            pv = val[pre.to_numpy()]
            print(f"截断前值: finite={int(np.isfinite(pv).sum())}  NaN={int(np.isnan(pv).sum())}"
                  f"  min={np.nanmin(pv):.1f}  max={np.nanmax(pv):.1f}")


if __name__ == "__main__":
    main()
