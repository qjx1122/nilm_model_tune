"""导出边缘部署包：run 目录（best.pt + result.json）+ 训练 npz → model.bin + manifest.json。

用法（PowerShell）：
  python scripts\\export_edge_bundle.py --run-dir reports\\h5k_f5_t23007 ^
      --stats-from D:\\datasets\\ukdale_h5_kettle.npz --out reports\\edge_bundle_h5k

说明：
  - 归一化统计量（x/y mean/std）取自 --stats-from 训练 npz 的 train 段（与 infer.py / build_splits
    完全同口径）——run 目录本身不落盘统计量，故必须提供训练时的 npz。
  - power_type：训练数据口径（UK-DALE mains=apparent 视在功率）。边缘引擎按此口径聚合三相功率。

model.bin 布局（小端；C 引擎 edge/nilm_edge.c 按同序读取）：
  header: magic "NEDG"(4B) | int32 ver=1 | int32×6 [d_model, nhead, num_layers, dim_ff,
          window, power_type(0=apparent,1=active)] | float64×5 [x_mean, x_std, y_mean,
          y_std, on_threshold_watts]
  float32 数组依次：
    proj_w[d] proj_b[d] pe[window*d]
    每层（num_layers 个）：ln1_w[d] ln1_b[d] qkvw[3d*d] qkvb[3d] outw[d*d] outb[d]
                          ln2_w[d] ln2_b[d] l1w[ff*d] l1b[ff] l2w[d*ff] l2b[d]
    尾：fn_w[d] fn_b[d] h1w[(d/2)*d] h1b[d/2] h2w[d/2] h2b[1]
  （权重布局与 torch 一致：Linear W 为 [out,in] 行主序；qkv 为 in_proj 合并 [3d,d]）
"""
import argparse
import hashlib
import json
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from runlog import setup_run_log
from src.model import NILMTransformer

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True, help="训练输出目录（best.pt + result.json）")
p.add_argument("--stats-from", required=True, help="训练时的 npz（统计量取其 train 段）")
p.add_argument("--out", required=True, help="部署包输出目录")
p.add_argument("--power-type", choices=["apparent", "active"], default="apparent",
               help="训练数据 aggregate 口径（默认 apparent=UK-DALE mains 同款）")
p.add_argument("--name", default="", help="部署包名称（缺省用 appliance 名）")
p.add_argument("--no-log", action="store_true")
args = p.parse_args()

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
setup_run_log(out_dir / "export.log", enabled=not args.no_log)

run_dir = Path(args.run_dir)
result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
cfg = result["config"]
mcfg, dcfg = cfg["model"], cfg["data"]
W = int(dcfg["window_size"])
thr = float(cfg["metrics"]["on_threshold_watts"])
d, nh, L, ff = mcfg["d_model"], mcfg["nhead"], mcfg["num_layers"], mcfg["dim_feedforward"]
half = d // 2
if d % nh != 0:
    raise SystemExit(f"d_model {d} 不能被 nhead {nh} 整除")
if not (1 < W <= 1024):
    raise SystemExit(f"window {W} 超出边缘引擎环形缓冲（2..1024）")

# ---- 归一化统计量（镜像 build_splits：仅 train 段） ----
dn = np.load(args.stats_from)
agg = np.asarray(dn["aggregate"], dtype=np.float64)
tgt = np.asarray(dn["target"], dtype=np.float64)
c = int(len(agg) * float(dcfg.get("train_ratio", 0.7)))
x_mean, x_std = float(agg[:c].mean()), float(agg[:c].std() + 1e-6)
y_mean, y_std = float(tgt[:c].mean()), float(tgt[:c].std() + 1e-6)
print(f"统计量（{args.stats_from} train 段）：x {x_mean:.2f}/{x_std:.2f} | y {y_mean:.2f}/{y_std:.2f}")

# ---- 权重导出 ----
model = NILMTransformer(**mcfg)
model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
model.eval()
sd = model.state_dict()
pe = model.pos.pe[0, :W, :].numpy().astype(np.float32)  # [W, d]（非持久 buffer，构造确定）


def g(name):
    return sd[name].detach().numpy().astype(np.float32).ravel()


arrays = [g("proj.weight"), g("proj.bias"), pe.ravel()]
for i in range(L):
    pre = f"encoder.layers.{i}."
    arrays += [
        g(pre + "norm1.weight"), g(pre + "norm1.bias"),
        g(pre + "self_attn.in_proj_weight"), g(pre + "self_attn.in_proj_bias"),
        g(pre + "self_attn.out_proj.weight"), g(pre + "self_attn.out_proj.bias"),
        g(pre + "norm2.weight"), g(pre + "norm2.bias"),
        g(pre + "linear1.weight"), g(pre + "linear1.bias"),
        g(pre + "linear2.weight"), g(pre + "linear2.bias"),
    ]
arrays += [g("norm.weight"), g("norm.bias"),
           g("head.0.weight"), g("head.0.bias"), g("head.2.weight"), g("head.2.bias")]
blob = np.concatenate(arrays)
n_params = int(blob.size) + 4 * pe.size // 4  # pe 已含
n_params = int(blob.size)

hdr = struct.pack("<4s i 6i 5d", b"NEDG", 1, d, nh, L, ff, W,
                  0 if args.power_type == "apparent" else 1,
                  x_mean, x_std, y_mean, y_std, thr)
(out_dir / "model.bin").write_bytes(hdr + blob.tobytes())


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).resolve().parents[1],
                                       text=True).strip()
    except Exception:
        return "unknown"


sha = hashlib.sha256((out_dir / "model.bin").read_bytes()).hexdigest()
manifest = {
    "name": args.name or dcfg.get("appliance", "model"),
    "appliance": dcfg.get("appliance", ""),
    "schema": "nilm-edge bundle v1",
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_commit": _git(),
    "source_run_dir": str(run_dir),
    "stats_from": str(args.stats_from),
    "arch": {"d_model": d, "nhead": nh, "num_layers": L, "dim_feedforward": ff, "window": W},
    "norm_stats": {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std},
    "on_threshold_watts": thr,
    "power_type": args.power_type,
    "tick_sec": 6.0,
    "latency_sec": (W - W // 2 - 1) * 6,  # 中心窗语义：预测中心落后最新数据
    "param_count": n_params,
    "model_bin_sha256": sha,
    "integration": "见 docs/EDGE_DEPLOYMENT.md（C API / 构建说明 / 联调清单）",
}
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

# ---- 回读校验 ----
raw = (out_dir / "model.bin").read_bytes()
magic, ver, hd, hh, hl, hff, hw, ptype = struct.unpack_from("<4s i 6i", raw, 0)
rx_mean, rx_std, ry_mean, ry_std, rthr = struct.unpack_from("<5d", raw, 32)
assert magic == b"NEDG" and ver == 1 and (hd, hh, hl, hff, hw) == (d, nh, L, ff, W)
assert (rx_mean, rx_std, ry_mean, ry_std, rthr) == (x_mean, x_std, y_mean, y_std, thr)
back = np.frombuffer(raw, dtype="<f4", offset=72)
assert back.size == blob.size and np.array_equal(back, blob)
print(f"部署包导出完成：{out_dir}")
print(f"  model.bin {len(raw)} B（{n_params} 参数）| manifest.json | "
      f"window={W}（延迟 {(W - W // 2 - 1) * 6:g}s）| power_type={args.power_type} | thr={thr:g}W")
print(f"  sha256={sha[:16]}…")
