"""解析 NILMTK HDF5 的 metadata（attrs 里 pickle 序列化的 dict），打印 building 表→电器映射。

NILMTK 转换后的 h5，meter 组 attrs 里常存 pickle 的 metadata dict，格式大致为：
    {u'appliances': [{u'type': u'kettle', u'instance': 1, u'meters': [10], ...}, ...],
     u'metadata': {...}}  （顶层 attrs 还可能是 subjects 等）
本脚本尽量稳健解析：能反序列化则递归找 appliance 定义（含 type/instance/meters），
再按 meter id 反查。解析失败会打印原始片段 + traceback，方便人工判读。

用法：
    python scripts/parse_nilmtk_metadata.py --h5-path <ukdale.h5> [--house 1]
"""
import argparse
import pickle
import pprint
import sys
from pathlib import Path

import h5py


def _decode_attrs(attrs):
    """把 bytes / 其它可 pickle 的值尽力 decode。"""
    out = {}
    for k, v in attrs.items():
        if isinstance(v, bytes):
            try:
                v = pickle.loads(v)  # NILMTK 常用
            except Exception:
                try:
                    v = v.decode("utf-8", "replace")
                except Exception:
                    pass
        out[k] = v
    return out


def _walk_appliances(obj, meter_to_app, out):
    """递归遍历解出的 dict，找 appliance 定义（含 meters 列表）。"""
    if isinstance(obj, dict):
        if "type" in obj and "meters" in obj:
            typ = obj["type"]
            if isinstance(typ, (list, tuple)) and typ:
                typ = typ[0] if isinstance(typ[0], str) else str(typ[0])
            inst = obj.get("instance")
            label = obj.get("label") or obj.get("name") or ""
            for m in obj["meters"]:
                meter_to_app.setdefault(int(m), []).append((typ, inst, label))
        for v in obj.values():
            _walk_appliances(v, meter_to_app, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_appliances(v, meter_to_app, out)


def parse(house_name, meter_groups, all_attrs):
    meter_to_app = {}
    extra = []
    for path, attrs in all_attrs:
        dec = _decode_attrs(attrs)
        for k, v in dec.items():
            if isinstance(v, (dict, list, tuple)):
                _walk_appliances(v, meter_to_app, {})
        # 收集不易解析的文本片段（供人读）
        for k, v in dec.items():
            if isinstance(v, bytes):
                extra.append(f"{path}.{k}: {v[:120]!r}")
    return meter_to_app, extra


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5-path", required=True)
    ap.add_argument("--house", type=int, default=1)
    args = ap.parse_args()

    target = f"building{args.house}"
    with h5py.File(args.h5_path, "r") as f:
        if target not in f:
            raise SystemExit(f"{target} 不在文件中（顶层：{list(f.keys())}）")
        b = f[target]

        meter_groups = {}
        all_attrs = [(target, b.attrs)]
        if "elec" in b:
            all_attrs.append((f"{target}/elec", b["elec"].attrs))
            for k in b["elec"].keys():
                g = b["elec"][k]
                if isinstance(g, h5py.Group) and str(k).startswith("meter"):
                    meter_groups[int(k[5:])] = g
                    all_attrs.append((f"{target}/elec/{k}", g.attrs))

        meter_to_app, extra = parse(target, meter_groups, all_attrs)

        print(f"== building{args.house} 电器映射（来自 pickle metadata）==")
        if meter_to_app:
            for mid in sorted(meter_to_app):
                for (typ, inst, label) in meter_to_app[mid]:
                    print(f"  meter {mid:<3} -> {typ} (instance={inst}, label={label!r})")
        else:
            print("  未能从 attrs 反序列化出 appliance 定义（见下方原始片段，可能需人工判读）")

        if extra:
            print("\n== 未解码的 attrs 片段（供人工判读）==")
            for e in extra[:10]:
                print(" ", e)

        print("\n提示：若上面映射为空，试试把某个 meter 组 attrs 的 metadata 内容贴出来人工看；"
              "mains 通常是 apparent 列（如 meter1/2/3 量级最大的前几个）。")
        print("制备命令模板：python scripts\\prepare_ukdale.py --h5-path <ukdale.h5> --house N "
              "--mains-ids <mains表号> --appliance-meter-id <电器表号> "
              "--appliance <电器名> --out <npz>")


if __name__ == "__main__":
    main()
