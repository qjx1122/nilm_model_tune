"""端到端 parity 测试：C 边缘引擎（edge/libnilm_edge.so）vs Python/torch 参考实现。

覆盖：
  A. tiny 训练链路：合成 npz → train.py 训练 → export_edge_bundle.py 导出 → 波形流推送
     （含 3 桶时间缺口=carry 语义验证）→ C 结果 vs torch 参考逐点对拍；
  B. 真实配置随机权重（F4 架构 d64/nhead8/L2/ff128/w96）：仅验证数值等价（parity 不要求精度）；
  C. 暖机语义：窗口未满时 poll 返回 0。

波形：3 相 50Hz/6400Hz，每包 128 点×6 通道（[uA,iA,uB,iB,uC,iC] 交错 float 工程量），
20ms/包；基线负载每相 100W PF=0.9（使视在≠有功，验证 power_type 选择），事件负载 2000W 纯阻性（A 相）。

运行：python tests/test_edge_parity.py（自动 make；依赖 numpy/torch）。
"""
import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import NILMTransformer  # noqa: E402

FS, F0, NPT = 6400.0, 50.0, 128
T0 = 996.0  # 6s 桶边界对齐起点（996 = 166×6）


class Result(ctypes.Structure):
    _fields_ = [("center_ts", ctypes.c_double), ("aggregate_w", ctypes.c_double),
                ("pred_w", ctypes.c_double), ("on", ctypes.c_int), ("reserved", ctypes.c_int)]


def build_lib():
    """构建并加载边缘库（跨平台，无 make 依赖；详见 edge/edgebuild.py）。"""
    sys.path.insert(0, str(ROOT / "edge"))
    from edgebuild import build as build_library
    lib = ctypes.CDLL(str(build_library()))
    lib.nilm_edge_start.restype = ctypes.c_int
    lib.nilm_edge_add_model.argtypes = [ctypes.c_char_p]
    lib.nilm_edge_add_model.restype = ctypes.c_int
    lib.nilm_edge_push_packet.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                          ctypes.c_int, ctypes.c_double]
    lib.nilm_edge_push_packet.restype = ctypes.c_int
    lib.nilm_edge_poll.argtypes = [ctypes.c_int, ctypes.POINTER(Result)]
    lib.nilm_edge_poll.restype = ctypes.c_int
    lib.nilm_edge_flush.restype = ctypes.c_int
    lib.nilm_edge_shutdown.restype = None
    lib.nilm_edge_last_error.restype = ctypes.c_char_p
    return lib


# ---------------- 波形与参考实现 ----------------

def packet_wave(k, on):
    """第 k 个周波（ts=T0+k/50）。on=事件负载是否开启（A 相 2000W 纯阻性）。"""
    t = (np.arange(NPT) + k * NPT) / FS
    w = np.empty((NPT, 6), dtype=np.float32)
    theta = np.arccos(0.9)  # 基线 PF=0.9 滞后
    for ph, phi in enumerate([0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0]):
        u = 220.0 * np.sqrt(2.0) * np.sin(2.0 * np.pi * F0 * t + phi)
        i = (100.0 / 220.0) * np.sqrt(2.0) * np.sin(2.0 * np.pi * F0 * t + phi - theta)
        if on and ph == 0:
            i = i + (2000.0 / 220.0) * np.sqrt(2.0) * np.sin(2.0 * np.pi * F0 * t + phi)
        w[:, ph * 2] = u
        w[:, ph * 2 + 1] = i
    return w


def run_stream(lib, mid, n_cycles, on_fn, hole_after=None, hole_cycles=0):
    """推送 n_cycles 个周波包，逐包 poll 收集结果。

    hole_after：在该周波后跳过 hole_cycles 个周波（模拟终端断流；空桶由引擎 carry 填充）。
    返回 (results, series)：series=参考端重建的 6s 桶序列（含 carry），与引擎同语义。
    """
    results = []
    buckets = {}  # bucket_id -> [sum_ui3, sum_u2_3, sum_i2_3, n]
    order = []    # 首末桶之间的全部桶 id（含空桶）
    for k in range(n_cycles):
        if hole_after is not None and hole_after < k <= hole_after + hole_cycles:
            continue  # 缺口：不推包
        ts = T0 + k / 50.0
        w = packet_wave(k, on_fn(k))
        assert lib.nilm_edge_push_packet(
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), NPT, 6, ts) == 0
        r = Result()
        while lib.nilm_edge_poll(mid, ctypes.byref(r)) == 1:
            results.append((r.center_ts, r.aggregate_w, r.pred_w, r.on))
        b = int(np.floor(ts / 6.0))
        acc = buckets.setdefault(b, [np.zeros(3), np.zeros(3), np.zeros(3), 0])
        u = w[:, 0::2].astype(np.float64)
        i = w[:, 1::2].astype(np.float64)
        acc[0] += (u * i).sum(axis=0)
        acc[1] += (u * u).sum(axis=0)
        acc[2] += (i * i).sum(axis=0)
        acc[3] += NPT
        if not order or order[-1] != b:
            order.append(b)
    lib.nilm_edge_flush()
    r = Result()
    while lib.nilm_edge_poll(mid, ctypes.byref(r)) == 1:
        results.append((r.center_ts, r.aggregate_w, r.pred_w, r.on))
    # 参考序列：首末桶之间逐桶（空桶=carry 上一值，同引擎语义）
    series = []
    last = None
    for b in range(order[0], order[-1] + 1):
        if b in buckets and buckets[b][3] > 0:
            s_ui, s_u2, s_i2, n = buckets[b]
            S = np.sqrt(s_u2 / n) * np.sqrt(s_i2 / n)
            v = float(S.sum())  # apparent 口径
            last = v
        else:
            v = last  # carry（首桶前无空桶，last 必非 None）
        series.append((b * 6.0, v))
    return results, series


def torch_ref(model, series, stats, window):
    """对序列每个可推理中心做 torch 参考 forward，返回 {center_ts: (pred, agg)}。"""
    x_mean, x_std, y_mean, y_std = stats
    vals = np.array([v for _, v in series])
    tss = np.array([t for t, _ in series])
    half = window // 2
    out = {}
    model.eval()
    with torch.no_grad():
        for c in range(half, len(vals) - (window - half) + 1):
            win = vals[c - half: c - half + window]
            x = ((win - x_mean) / x_std).astype(np.float32)
            y = float(model(torch.from_numpy(x)[None, :, None]))
            out[float(tss[c])] = (y * y_std + y_mean, float(vals[c]))
    return out


def compare(results, ref, stats, thr, tag):
    x_mean, x_std, y_mean, y_std = stats
    assert results, f"{tag}: 引擎无结果"
    d_agg = d_pred = 0.0
    flag_mismatch = 0
    for c_ts, agg, pred, on in results:
        assert c_ts in ref, f"{tag}: 中心 {c_ts} 不在参考范围"
        rp, ra = ref[c_ts]
        d_agg = max(d_agg, abs(agg - ra) / max(1.0, abs(ra)))
        d_pred = max(d_pred, abs(pred - rp))
        if abs(rp - thr) > 1e-3 and on != int(rp >= thr):
            flag_mismatch += 1
    n = len(results)
    n_ref = len(ref)
    assert n == n_ref, f"{tag}: 结果数 {n} != 参考 {n_ref}"
    assert d_agg < 1e-9, f"{tag}: 功率层偏差 {d_agg:.2e}"
    tol = max(0.5, 0.002 * abs(y_std))
    assert d_pred < tol, f"{tag}: 预测偏差 {d_pred:.4f}W 超容差 {tol:.4f}W"
    assert flag_mismatch == 0, f"{tag}: ON 判定不一致 {flag_mismatch} 处"
    print(f"  [{tag}] n={n} | Δagg rel {d_agg:.2e} | Δpred max {d_pred:.5f}W（容差 {tol:.3f}W）| ON 全一致 ✓")


def train_stats(npz_path, train_ratio=0.7):
    d = np.load(npz_path)
    agg = np.asarray(d["aggregate"], dtype=np.float64)
    tgt = np.asarray(d["target"], dtype=np.float64)
    c = int(len(agg) * train_ratio)
    return (float(agg[:c].mean()), float(agg[:c].std() + 1e-6),
            float(tgt[:c].mean()), float(tgt[:c].std() + 1e-6))


# ---------------- 用例 ----------------

def test_tiny_trained(tmp):
    """A：tiny 训练链路 + 缺口 carry + 暖机。"""
    rng = np.random.default_rng(42)
    n = 20000
    base = np.clip(300 + np.cumsum(rng.normal(0, 2, n)), 150, 600)
    tgt = np.zeros(n)
    for _ in range(60):
        s = rng.integers(100, n - 100)
        tgt[s:s + int(rng.integers(20, 50))] = 2000.0
    npz = tmp / "synth.npz"
    np.savez(npz, aggregate=(base + tgt + rng.normal(0, 5, n)).astype(np.float32),
             target=tgt.astype(np.float32))
    cfg = tmp / "cfg.yaml"
    cfg.write_text(
        "seed: 42\ndevice: cpu\ndata:\n  appliance: testapp\n  window_size: 16\n"
        "  train_ratio: 0.7\n  val_ratio: 0.15\n  test_ratio: 0.15\n"
        "  max_samples_train: 4000\n  max_samples_val: 2000\n  max_samples_test: 1000\n"
        "model:\n  input_dim: 1\n  d_model: 8\n  nhead: 2\n  num_layers: 1\n"
        "  dim_feedforward: 16\n  dropout: 0.0\ntraining:\n  batch_size: 256\n"
        "  epochs: 3\n  lr: 0.001\n  weight_decay: 0.0001\n  patience: 2\n"
        "  grad_clip: 1.0\n  loss: mse\nmetrics:\n  on_threshold_watts: 1000.0\n")
    run = tmp / "tiny_run"
    subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--config", str(cfg),
                    "--data-path", str(npz), "--out", str(run), "--no-log"],
                   check=True, capture_output=True)
    bundle = tmp / "bundle_tiny"
    subprocess.run([sys.executable, str(ROOT / "scripts/export_edge_bundle.py"),
                    "--run-dir", str(run), "--stats-from", str(npz),
                    "--out", str(bundle), "--no-log"], check=True, capture_output=True)

    lib = build_lib()
    assert lib.nilm_edge_start() == 0
    mid = lib.nilm_edge_add_model(str(bundle / "model.bin").encode())
    assert mid >= 0, lib.nilm_edge_last_error()

    # 暖机：窗口 16 → 前 15 桶无结果
    r = Result()
    assert lib.nilm_edge_poll(mid, ctypes.byref(r)) == 0, "暖期内不应有结果"

    n_buckets, hole_after, hole_buckets = 60, 20 * 300, 3
    n_cycles = n_buckets * 300
    on_fn = lambda k: (int((T0 + k / 50.0) / 6.0) % 5) in (1, 2)
    results, series = run_stream(lib, mid, n_cycles, on_fn,
                                 hole_after=hole_after, hole_cycles=hole_buckets * 300)
    lib.nilm_edge_shutdown()

    # 序列长度 = 桶数（含 3 个 carry 桶）
    assert len(series) == n_buckets, f"序列 {len(series)} != {n_buckets}"
    cfgm = json.loads((run / "result.json").read_text())["config"]
    model = NILMTransformer(**cfgm["model"])
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    stats = train_stats(npz)
    ref = torch_ref(model, series, stats, 16)
    compare(results, ref, stats, 1000.0, "tiny 训练链路（含 3 桶缺口 carry）")


def test_real_arch_random(tmp):
    """B：真实配置（F4 架构 w96/d64/nhead8/L2/ff128）随机权重 parity。"""
    mcfg = {"input_dim": 1, "d_model": 64, "nhead": 8, "num_layers": 2,
            "dim_feedforward": 128, "dropout": 0.0}
    torch.manual_seed(123)
    model = NILMTransformer(**mcfg)
    run = tmp / "fx_run"
    run.mkdir(parents=True)
    torch.save(model.state_dict(), run / "best.pt")
    cfg = {"seed": 123, "device": "cpu",
           "data": {"appliance": "fixture_kettle", "window_size": 96, "train_ratio": 0.7,
                    "val_ratio": 0.15, "test_ratio": 0.15, "max_samples_train": 1000,
                    "max_samples_val": 1000, "max_samples_test": 500},
           "model": mcfg, "training": {"batch_size": 64, "epochs": 1, "lr": 3e-4,
                                       "weight_decay": 1e-4, "patience": 1, "grad_clip": 1.0,
                                       "loss": "mse"},
           "metrics": {"on_threshold_watts": 500.0}}
    (run / "result.json").write_text(json.dumps(
        {"best_epoch": 1, "device": "cpu", "runtime_sec": 0.0, "test": None,
         "config": cfg, "n_train": 0, "n_val": 0, "n_test": 0}))
    npz = tmp / "synth.npz"
    if not npz.exists():
        rng = np.random.default_rng(42)
        n = 20000
        base = np.clip(300 + np.cumsum(rng.normal(0, 2, n)), 150, 600)
        tgt = np.zeros(n)
        for _ in range(60):
            s = rng.integers(100, n - 100)
            tgt[s:s + int(rng.integers(20, 50))] = 2000.0
        np.savez(npz, aggregate=(base + tgt + rng.normal(0, 5, n)).astype(np.float32),
                 target=tgt.astype(np.float32))
    bundle = tmp / "bundle_fx"
    subprocess.run([sys.executable, str(ROOT / "scripts/export_edge_bundle.py"),
                    "--run-dir", str(run), "--stats-from", str(npz),
                    "--out", str(bundle), "--no-log"], check=True, capture_output=True)

    lib = build_lib()
    assert lib.nilm_edge_start() == 0
    mid = lib.nilm_edge_add_model(str(bundle / "model.bin").encode())
    assert mid >= 0, lib.nilm_edge_last_error()
    n_buckets = 104  # 96 窗 + 8
    on_fn = lambda k: (int((T0 + k / 50.0) / 6.0) % 4) in (1, 2)
    results, series = run_stream(lib, mid, n_buckets * 300, on_fn)
    lib.nilm_edge_shutdown()

    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    stats = train_stats(npz)
    ref = torch_ref(model, series, stats, 96)
    compare(results, ref, stats, 500.0, "真实配置随机权重 w96/d64/L2")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="nilm_edge_parity_"))
    print(f"工作目录：{tmp}")
    test_tiny_trained(tmp)
    test_real_arch_random(tmp)
    print("\n全部 parity 测试通过 ✅")


if __name__ == "__main__":
    main()
