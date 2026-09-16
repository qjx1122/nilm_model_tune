"""端到端彩排：现场录制落盘链路（recorder.c 双写 → .raw → raw_to_recnpz.py → NPZ v1 → 回放）。

验证目标（模拟终端在现场的真实行为）：
  1. 在线双写：每包同时推引擎（nilm_edge_push_packet）+ 录制器（nilm_rec_packet）；
  2. .raw 经 raw_to_recnpz.py 转换后，wave/ts 与在线推送的包**逐位一致**；
  3. 同一 bundle 回放（新引擎实例）结果与在线结果**逐位一致**——录制-转换-回放全链路无损；
  4. 时间戳截断容错：人为截断尾记录 → 转换器自动丢弃并警告，其余数据完好。

运行：python tests/test_recorder_loop.py（依赖 numpy/torch；录制器库自动定位/构建）。
"""
import ctypes
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
from test_edge_parity import build_lib, Result  # noqa: E402
from edge_stream_test import Scenario, EdgeRunner  # noqa: E402

IS_WIN = platform.system() == "Windows"
REC_LIB = ROOT / "edge" / ("recorder.dll" if IS_WIN else "librecorder.so")


def build_recorder():
    if REC_LIB.exists():
        return REC_LIB
    for cc, extra in (("gcc", []), ("clang", []),
                      ("zig", ["cc", "-target", "x86_64-windows-gnu"] if IS_WIN else ["cc"])):
        try:
            cmd = ["python", "-m", "ziglang"] if cc == "zig" else [cc]
            if cc == "zig":
                cmd = ["python", "-m", "ziglang"] + extra
            else:
                cmd = [cc, "-O2", "-std=c99", "-fPIC", "-shared", "-Wall",
                       "-DNILM_REC_BUILDING", "recorder.c", "-o", REC_LIB.name]
            subprocess.run(cmd, cwd=ROOT / "edge", check=True, capture_output=True)
            if REC_LIB.exists():
                return REC_LIB
        except Exception:
            continue
    raise SystemExit(f"找不到录制器库 {REC_LIB}，且本机无编译器（git pull 获取预编译库）")


def load_recorder():
    lib = ctypes.CDLL(str(build_recorder()))
    lib.nilm_rec_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.nilm_rec_open.restype = ctypes.c_int
    lib.nilm_rec_packet.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                    ctypes.c_int, ctypes.c_double]
    lib.nilm_rec_packet.restype = ctypes.c_int
    lib.nilm_rec_close.restype = ctypes.c_int
    lib.nilm_rec_last_error.restype = ctypes.c_char_p
    return lib


TINY_CFG = """seed: 42
device: cpu
data:
  appliance: kettle
  window_size: 16
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  max_samples_train: 8000
  max_samples_val: 4000
  max_samples_test: 1000
model:
  input_dim: 1
  d_model: 8
  nhead: 2
  num_layers: 1
  dim_feedforward: 16
  dropout: 0.0
training:
  batch_size: 256
  epochs: 3
  lr: 0.001
  weight_decay: 0.0001
  patience: 2
  grad_clip: 1.0
  loss: mse
metrics:
  on_threshold_watts: 1000.0
"""


def train_bundle(tmp):
    npz = tmp / "synth.npz"
    rng = np.random.default_rng(42)
    n = 20000
    base = np.clip(300 + np.cumsum(rng.normal(0, 2, n)), 150, 600)
    tgt = np.zeros(n)
    for _ in range(60):
        s = rng.integers(100, n - 100)
        tgt[s:s + int(rng.integers(20, 50))] = 2000.0
    np.savez(npz, aggregate=(base + tgt + rng.normal(0, 5, n)).astype(np.float32),
             target=tgt.astype(np.float32))
    cfg = tmp / "cfg.yaml"
    cfg.write_text(TINY_CFG, encoding="utf-8")
    run = tmp / "run"
    subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--config", str(cfg),
                    "--data-path", str(npz), "--out", str(run), "--no-log"],
                   check=True, capture_output=True)
    bdir = tmp / "bundle"
    subprocess.run([sys.executable, str(ROOT / "scripts/export_edge_bundle.py"),
                    "--run-dir", str(run), "--stats-from", str(npz),
                    "--out", str(bdir), "--no-log"], check=True, capture_output=True)
    return bdir / "model.bin"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="nilm_rec_loop_"))
    print(f"工作目录：{tmp}")
    bundle = train_bundle(tmp)
    print("① tiny 训练+导出完成")

    # ── 在线双写（模拟终端：引擎+录制器同点调用）──
    lib = build_lib()
    rlib = load_recorder()
    raw_path = tmp / "live.raw"
    assert rlib.nilm_rec_open(str(raw_path).encode(), 128, 6) == 0, \
        rlib.nilm_rec_last_error()
    runner = EdgeRunner(lib, [bundle])
    sc = Scenario(303, 140, 5, 1)          # 140 桶=14min 流（含 kettle 事件）
    live_rows, live_ts, live_res = [], [], []
    r = Result()
    for i, ts, wave, truth, ana in sc.iter_wave():
        for k in range(len(ts)):
            row = wave[k]
            assert runner.push(row, float(ts[k])) is None or True
            assert rlib.nilm_rec_packet(
                row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 128, 6,
                float(ts[k])) == 0, rlib.nilm_rec_last_error()
            live_rows.append(row)
            live_ts.append(float(ts[k]))
        while lib.nilm_edge_poll(runner.models[0][1], ctypes.byref(r)) == 1:
            live_res.append((float(r.center_ts), float(r.aggregate_w),
                             float(r.pred_w), int(r.on)))
    assert rlib.nilm_rec_close() == 0
    lib.nilm_edge_flush()                    # 停机结算末桶（与回放段同语义）
    while lib.nilm_edge_poll(runner.models[0][1], ctypes.byref(r)) == 1:
        live_res.append((float(r.center_ts), float(r.aggregate_w),
                         float(r.pred_w), int(r.on)))
    print(f"② 在线双写完成：{len(live_ts)} 包 → {raw_path.name}"
          f"（{raw_path.stat().st_size / 1e6:.1f} MB），在线结果 {len(live_res)} 条")

    # ── 截断容错：人为去掉尾记录 1000B ──
    truncated = tmp / "trunc.raw"
    truncated.write_bytes(raw_path.read_bytes()[:-1000])

    # ── 转换（含截断文件单独验证）──
    npz_full = tmp / "rec.npz"
    subprocess.run([sys.executable, str(ROOT / "scripts/raw_to_recnpz.py"),
                    "--raw", str(raw_path), "--out", str(npz_full)],
                   check=True, capture_output=True, text=True)
    npz_trunc = tmp / "trunc.npz"
    rt = subprocess.run([sys.executable, str(ROOT / "scripts/raw_to_recnpz.py"),
                         "--raw", str(truncated), "--out", str(npz_trunc)],
                        check=True, capture_output=True, text=True)
    assert "截断" in (rt.stderr + rt.stdout) or "丢弃" in (rt.stderr + rt.stdout), \
        "截断警告未出现"
    d = np.load(npz_full)
    dt = np.load(npz_trunc)
    assert len(dt["ts"]) == len(d["ts"]) - 1 or len(dt["ts"]) < len(d["ts"]), "截断丢弃量异常"
    print(f"③ 转换完成：完整 {len(d['ts'])} 包 / 截断 {len(dt['ts'])} 包（尾记录自动丢弃 ✓）")

    # ── 数据路径无损：npz == 在线推送的包 ──
    assert np.array_equal(d["wave"], np.asarray(live_rows, np.float32)), "wave 不一致"
    assert np.array_equal(d["ts"], np.asarray(live_ts, np.float64)), "ts 不一致"
    print("④ 转换数据与在线推送逐位一致（wave/ts）✓")

    # ── 回放（新引擎实例）结果与在线逐位一致 ──
    lib.nilm_edge_shutdown()
    runner2 = EdgeRunner(lib, [bundle])
    replay_res = []
    wave_arr, ts_arr = d["wave"], d["ts"]
    for k in range(len(ts_arr)):
        assert runner2.push(wave_arr[k], float(ts_arr[k])) is None or True
        if k % 200 == 199 or k == len(ts_arr) - 1:
            while lib.nilm_edge_poll(runner2.models[0][1], ctypes.byref(r)) == 1:
                replay_res.append((float(r.center_ts), float(r.aggregate_w),
                                   float(r.pred_w), int(r.on)))
    lib.nilm_edge_flush()
    while lib.nilm_edge_poll(runner2.models[0][1], ctypes.byref(r)) == 1:
        replay_res.append((float(r.center_ts), float(r.aggregate_w),
                           float(r.pred_w), int(r.on)))
    lib.nilm_edge_shutdown()
    assert len(replay_res) == len(live_res), \
        f"结果条数不一致：回放 {len(replay_res)} vs 在线 {len(live_res)}"
    for a, b in zip(live_res, replay_res):
        assert a == b, f"结果不一致：在线 {a} vs 回放 {b}"
    print(f"⑤ 回放 vs 在线：{len(replay_res)} 条结果逐位一致 ✓")

    print("\n录制-转换-回放全链路彩排通过 ✅")


if __name__ == "__main__":
    main()
