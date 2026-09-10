"""单 meter pandas 表原始真相探查（零过滤 dump，与 read_power_series 视图对照）。

背景（执行实录 24→25 定谳）：House2 kettle 制备打印「kettle 网格点 3,377,557」
曾按 Apr16 起点判为超跨度异常。probe 实测（实录 25）：meter8 表头夹带 8 行
2013-02-17 16:00:22 起的 0W 杂散（安装测试残留），表真实跨度 Feb 17 16:00:22
→Oct 10 06:15:56 的 6s 满跨度格数恰=3,377,557（与 prepare 打印一字不差）；
杂散全落在 mains 覆盖之前的 172 格头部区，制备时整段剔除，npz 不受影响。
旧「与 list-meters Apr16 起点互斥」系实录 24 转写误差：list-meters 的 start=
read_power_series 原始 min（isfinite 不滤 0W 行），本文件必显 Feb 17 16:00:22。

v2（实录 25）：修复 cutoff 分支 ndarray 无 to_numpy 的 crash；截断前行 ≤20 行
全量打印；新增截断后前 5 行与「6s 满跨度格数」行（floor 公式与 prepare 的
resample('6s', origin='epoch') 满跨度一致，已沙箱对拍验证）。

用法：
    python scripts\\probe_meter.py --h5-path <ukdale.h5> --house 2 --meter 8 ^
        --cutoff "2013-04-16 21:18:09"
    python scripts\\probe_meter.py --h5-path <ukdale.h5> --house 2 --meter 1
"""
import argparse

import numpy as np
import pandas as pd

GRID_SECONDS = 6  # 与 prepare_ukdale.SECONDS_PER_SAMPLE 一致


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5-path", required=True)
    p.add_argument("--house", type=int, required=True)
    p.add_argument("--meter", type=int, required=True)
    p.add_argument("--cutoff", default=None,
                   help="对照截断时间戳（naive 按表时区解释）：统计其前后行数、"
                        "截断前明细与截断后前 5 行")
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
    if len(df):
        b0 = idx.min().floor(f"{GRID_SECONDS}s")
        b1 = idx.max().floor(f"{GRID_SECONDS}s")
        n_bins = int((b1 - b0) / pd.Timedelta(seconds=GRID_SECONDS)) + 1
        print(f"6s 网格满跨度格数(epoch origin, 按原始 min/max) = {n_bins}"
              f"（prepare 的 resample 跨度即此；若极值行非 finite 会略小）")

    if a.cutoff:
        ts = pd.Timestamp(a.cutoff)
        if getattr(idx, "tz", None) is not None and ts.tzinfo is None:
            ts = ts.tz_localize(idx.tz)
        pre = np.asarray(idx < ts)  # DatetimeIndex 比较返回 ndarray（实录 25 crash 修复）
        post = ~pre
        print(f"cutoff={ts}  之前行数={int(pre.sum())}  之后行数={int(post.sum())}")
        if pre.any():
            npre = int(pre.sum())
            label = f"全部 {npre} 行" if npre <= 20 else "前 5 行"
            print(f"截断点之前行明细（{label}）：")
            print((df[pre] if npre <= 20 else df[pre].head(5)).to_string())
            pv = val[pre]
            print(f"截断前值: finite={int(np.isfinite(pv).sum())}  NaN={int(np.isnan(pv).sum())}"
                  f"  min={np.nanmin(pv):.1f}  max={np.nanmax(pv):.1f}")
        if post.any():
            print("截断点之后前 5 行：")
            print(df[post].head(5).to_string())


if __name__ == "__main__":
    main()
