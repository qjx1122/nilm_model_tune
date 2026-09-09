"""探查 NILMTK / 自定义 HDF5 布局（递归结构 + attrs + pandas table 兼容读）。

用途：prepare_ukdale.py 报「布局与契约不符」时，先跑本脚本摸清文件真实结构，
再决定怎么适配。兼容两类 meter 存储：
  A. 直接数据：组内含 power / power_series 数据集（(N,2)）
  B. pandas 表（键含 _i_table / table）：尝试用 pd.read_hdf 读列名与首尾时间戳

用法：
    python scripts/inspect_h5.py --path <ukdale.h5>
    python scripts/inspect_h5.py --path <ukdale.h5> --max-meters 60
"""
import argparse
import sys

import h5py
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None


def _fmt_attrs(attrs, limit=8):
    parts = []
    for k, v in list(attrs.items())[:limit]:
        s = str(v)
        if len(s) > 60:
            s = s[:60] + "…"
        parts.append(f"{k}={s}")
    return ", ".join(parts) if parts else "(无 attrs)"


def _first_rows_of_table(f, meter_path):
    """尝试把 meter 组当 pandas HDFStore 表读取，返回列名与首尾行。"""
    if pd is None:
        return None, "pandas 未安装，跳过 pd.read_hdf"
    try:
        df = pd.read_hdf(f.filename, meter_path)
        if df is None or len(df) == 0:
            return None, "表为空"
        cols = [str(c) for c in df.columns]
        head = df.head(3)
        tail = df.tail(2)
        lines = [f"  columns={cols}  shape={df.shape}"]
        lines.append("  head:")
        for _i, row in head.iterrows():
            lines.append(f"    {list(row)}")
        lines.append("  tail:")
        for _i, row in tail.iterrows():
            lines.append(f"    {list(row)}")
        return "\n".join(lines), None
    except Exception as e:
        return None, f"pd.read_hdf 失败: {type(e).__name__}: {e}"


def inspect(path, max_meters=60):
    with h5py.File(path, "r") as f:
        print(f"== 顶层键 ==")
        for k in f.keys():
            print(" ", k, "(group)" if isinstance(f[k], h5py.Group) else "(dataset)")
        print(f"\n顶层 attrs: {_fmt_attrs(f.attrs, 12)}")

        for bname in sorted(k for k in f.keys()
                            if isinstance(f[k], h5py.Group)
                            and str(k).lower().startswith("building")):
            bg = f[bname]
            print(f"\n== {bname} ==")
            print(f"  attrs: {_fmt_attrs(bg.attrs, 12)}")
            elec = bg["elec"] if "elec" in bg else None
            if elec is not None:
                print(f"  {bname}/elec attrs: {_fmt_attrs(elec.attrs, 12)}")
                meters = sorted(
                    (k for k in elec.keys()
                     if isinstance(elec[k], h5py.Group) and str(k).startswith("meter")),
                    key=lambda s: int(s[5:]))
                print(f"  meter 数: {len(meters)}（展示前 {max_meters} 个）")
                for mk in meters[:max_meters]:
                    mg = elec[mk]
                    mid = mk[5:]
                    print(f"  meter {mid}: attrs: {_fmt_attrs(mg.attrs, 8)}")
                    child_keys = list(mg.keys())
                    print(f"    子键: {child_keys}")
                    ds_info = []
                    for ck in child_keys:
                        c = mg[ck]
                        if isinstance(c, h5py.Dataset):
                            ds_info.append(f"{ck} shape={c.shape} dtype={c.dtype}")
                    for line in ds_info:
                        print("    dataset:", line)
                    if any(s in child_keys for s in ("power", "power_series")):
                        pass  # 契约 A，prepare 可直接读
                    if "_i_table" in child_keys or "table" in child_keys:
                        detail, err = _first_rows_of_table(f, f"{bname}/elec/{mk}")
                        print("    [pandas 表]")
                        if err:
                            print(f"    {err}")
                        else:
                            print(detail)
            else:
                print(f"  {bname} 下无 elec 组，子键: {list(bg.keys())[:20]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", required=True)
    ap.add_argument("--max-meters", type=int, default=60)
    args = ap.parse_args()
    inspect(args.path, args.max_meters)


if __name__ == "__main__":
    main()
