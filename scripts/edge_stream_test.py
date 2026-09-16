"""edge_stream_test.py — 边缘流式推理分阶段验证 harness（S1 合成 / S2 现场录制回放）。

三阶段验证计划（docs/EDGE_DEPLOYMENT.md §7）：
  S1 合成周波流：内置三相场景（基线+背景负荷+kettle/dw 事件+噪声谐波+短缺口 D4 填充+长缺口重置）
      → 渲染 6s 训练序列 → train.py 训练 tiny 模型 → export_edge_bundle.py 导出 →
      周波包流式推送（模拟终端 20ms/包）→ 功能验收（特征层精度/事件 F1/kWh/暖机/延迟/缺口语义/性能）。
  S2 现场录制回放：--mode replay --wave-file <录制.npz> [--bundle ...]，
      同一推送路径回放录制周波（全速或 --realtime 实时 pacing），输出质量报告+模型结果；
      有真值键（truth_on_<name>）时做事件指标；--compare-with 对拍此前报告（确定性回归）。
  S3 终端实测：待现场（联调清单见文档 §9）。

用法：
  # S1 全流程（训练+回放+验收+录制样例段）：
  python scripts/edge_stream_test.py --mode synth --out-dir reports/edge_s1 --record rec.npz
  # S2 回放录制文件（可与 S1 报告逐条对拍验证确定性）：
  python scripts/edge_stream_test.py --mode replay --wave-file rec.npz \
      --bundle reports/edge_s1/bundle_kettle/model.bin --bundle reports/edge_s1/bundle_dw/model.bin \
      --out-dir reports/edge_s1/replay --compare-with reports/edge_s1/report.json

录制文件格式（NPZ v1，供终端/采集侧生成）：
  wave: float32 [n, 768]  每行一包：128 点×6 通道交错 [uA,iA,uB,iB,uC,iC]，工程量 V/A
  ts:   float64 [n]       每包时间戳（秒，单调不减）
  truth_on_<appliance>: int8 [n]（可选）每周波目标电器真值（受控切换实验时提供）
  meta: json 字符串（可选）站点/时间/通道映射等
"""
import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_edge_parity import build_lib, Result  # noqa: E402

FS, F0, NPT, CH = 6400.0, 50.0, 128, 6
TICK = 6.0
CPB = int(round(TICK * F0))          # 300 周波/桶
PHI = [0.0, -2 * np.pi / 3, 2 * np.pi / 3]
V0 = 220.0
B0 = 200                             # 流起始绝对桶（非零起点+首桶不完整，测部分桶）

# ---- S1 功能验收线（业务精度验收属 S2/S3 黄金集，此处为功能/一致性线） ----
ACC_ANALYTIC = 0.005     # 特征层：引擎 6s 功率 vs 场景解析值 最大相对误差
ACC_F1_POINT = 0.90      # 点级 F1（kettle / dish_washer 各自）
ACC_F1_EVENT = 0.90      # 事件级 F1
ACC_KWH = 0.05           # kWh 相对误差（有预测覆盖的桶区间）
MIN_X_REALTIME = 50.0    # 全速回放至少 50× 实时


# ═══════════════ 场景模型（6s 渲染与周波渲染共享同一负荷状态机，确定性种子） ═══════════════

class Scenario:
    """三相负荷场景。

    负荷：基线每相 (180W, 60var)；冰箱 150W PF0.8（B 相，20min 开/40min 关）；
    电视 120W PF0.95（C 相，随机块）；kettle 2200W 纯阻（A 相，20-40 桶=2-4min 事件）；
    dish_washer 1600W 加热相位（A 相，[42 开/18 关]×3=180 桶事件）。
    缺口：短缺口 20s（D4 语义=终端用上一包数据在原时间槽重推）；长缺口 31min（终端死寂→
    引擎 carry 300 桶→重置→再暖机）。事件调度避开缺口窗口 ±40 桶。
    """

    KETTLE_W, DW_W = 2200.0, 1600.0

    def __init__(self, seed, n_buckets, ev_kettle, ev_dw, gap_short=None, gap_long=None):
        self.n = n_buckets
        self.gap_short = gap_short            # (start_bucket, 秒) 或 None
        self.gap_long = gap_long              # (start_bucket, 秒) 或 None
        self.gap_long_buckets = int(np.ceil(gap_long[1] / TICK)) if gap_long else 0
        rng = np.random.default_rng(seed)
        tv = np.zeros(n_buckets, bool)
        b = 0
        while b < n_buckets:
            on = int(rng.integers(30, 120))
            off = int(rng.integers(60, 300))
            tv[b:b + on] = True
            b += on + off
        self.tv = tv
        forbid = []
        if gap_short:
            forbid.append((gap_short[0] - 40,
                           gap_short[0] + int(np.ceil(gap_short[1] / TICK)) + 40))
        if gap_long:
            forbid.append((gap_long[0] - 40, gap_long[0] + self.gap_long_buckets + 40))
        self.k_events = self._sched(rng, ev_kettle, (20, 40), forbid)
        self.d_events = self._sched(rng, ev_dw, (180, 181), forbid)
        self.rng_w = np.random.default_rng(seed + 1)   # 波形噪声（迭代序确定）
        self.rng_6 = np.random.default_rng(seed + 2)   # 6s 渲染噪声

    def _sched(self, rng, n_events, dur, forbid):
        """事件调度（拒绝采样）：彼此隔 ≥20 桶、避开禁窗（缺口 ±40）。返回 [(start, end)]。"""
        evs, tries = [], 0
        lo, hi = 48, self.n - dur[1] - 48
        if hi <= lo:
            return evs
        while len(evs) < n_events and tries < 100000:
            tries += 1
            d = int(rng.integers(dur[0], dur[1] + 1))
            s = int(rng.integers(lo, hi))
            e = s + d
            if any(not (e + 40 < fs or s > fe + 40) for fs, fe in forbid):
                continue
            if any(not (e + 20 < s2 or s > e2 + 20) for s2, e2 in evs):
                continue
            evs.append((s, e))
        evs.sort()
        return evs

    def loads(self, i):
        """第 i 桶（场景坐标）负荷：返回 (P[3], Q[3], {kettle: W, dish_washer: W})。"""
        P = [180.0, 180.0, 180.0]
        Q = [60.0, 60.0, 60.0]
        if (i % 600) < 200:                    # 冰箱 20min 开 / 40min 关
            P[1] += 150.0
            Q[1] += 112.5
        if self.tv[i]:                         # 电视随机块
            P[2] += 120.0
            Q[2] += 39.5
        tk = self.KETTLE_W if any(s <= i < e for s, e in self.k_events) else 0.0
        td = 0.0
        for s, _e in self.d_events:
            if s <= i < s + 180:               # [42 开/18 关]×3
                td = self.DW_W if (i - s) % 60 < 42 else 0.0
                break
        P[0] += tk + td
        return P, Q, {"kettle": tk, "dish_washer": td}

    def render_6s(self, noise=3.0):
        """渲染 6s 训练序列（与周波路径同口径：apparent 三相和）。"""
        agg = np.empty(self.n)
        tk = np.empty(self.n)
        td = np.empty(self.n)
        for i in range(self.n):
            P, Q, t = self.loads(i)
            agg[i] = sum(np.hypot(P[p], Q[p]) for p in range(3))
            tk[i], td[i] = t["kettle"], t["dish_washer"]
        agg += self.rng_6.normal(0.0, noise, self.n)
        return agg, tk, td

    def iter_wave(self):
        """逐桶渲染周波包。yield (i, ts[m], wave[m,768], truth{name:[m] int8}, analytic_S)。

        时间轴：每包时间戳 ts=B0*6+1.0+j/50（20ms 网格，j=绝对周波序号）；包内 128 样本
        采样时刻 = ts + p/6400（p=0..127）→ 首桶仅 250 包（桶内偏移 1s，测部分桶语义）。
        wave 行布局 = 点×通道交错：row[p*6 + ch]，ch∈[uA,iA,uB,iB,uC,iC]。
        """
        w = 2.0 * np.pi * F0
        poff = np.arange(NPT) / FS                   # 包内样本偏移（1/6400s 网格）
        for i in range(self.n):
            j0, j1 = max(0, 300 * i - 50), 300 * i + 250
            ts = B0 * TICK + 1.0 + np.arange(j0, j1) / F0
            P, Q, targets = self.loads(i)
            m = len(ts)
            t = ts[:, None] + poff[None, :]          # (m, 128) 样本绝对时间
            wave = np.empty((m, NPT * CH), np.float32)
            w3 = wave.reshape(m, NPT, CH)
            for ph in range(3):
                u = (V0 * np.sqrt(2.0) * np.sin(w * t + PHI[ph])
                     + 0.003 * V0 * np.sqrt(2.0) * np.sin(3.0 * w * t + PHI[ph]))
                cur = ((np.sqrt(2.0) * P[ph] / V0) * np.sin(w * t + PHI[ph])
                       - (np.sqrt(2.0) * Q[ph] / V0) * np.cos(w * t + PHI[ph]))
                u = u + self.rng_w.normal(0.0, 0.5, (m, NPT))
                cur = cur + self.rng_w.normal(0.0, 0.03, (m, NPT))
                w3[:, :, ph * 2] = u
                w3[:, :, ph * 2 + 1] = cur
            truth = {k: np.full(m, int(v > 0), np.int8) for k, v in targets.items()}
            analytic = float(sum(np.hypot(P[p], Q[p]) for p in range(3)))
            yield i, ts, wave, truth, analytic


# ═══════════════ 引擎驱动 ═══════════════

class EdgeRunner:
    def __init__(self, lib, bundle_paths):
        self.lib = lib
        assert lib.nilm_edge_start() == 0
        self.models = []   # (name, mid, window, threshold)
        for bp in bundle_paths:
            man = json.loads((Path(bp).parent / "manifest.json").read_text(encoding="utf-8"))
            mid = lib.nilm_edge_add_model(str(bp).encode())
            assert mid >= 0, lib.nilm_edge_last_error()
            self.models.append((man["name"], mid, int(man["arch"]["window"]),
                                float(man["on_threshold_watts"])))

    def push(self, row, ts):
        rc = self.lib.nilm_edge_push_packet(
            row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), NPT, CH, ts)
        assert rc == 0, self.lib.nilm_edge_last_error()

    def drain(self):
        out = {}
        r = Result()
        for name, mid, _w, _t in self.models:
            lst = []
            while self.lib.nilm_edge_poll(mid, ctypes.byref(r)) == 1:
                lst.append((float(r.center_ts), float(r.aggregate_w),
                            float(r.pred_w), int(r.on)))
            out[name] = lst
        return out

    def flush_and_drain(self):
        self.lib.nilm_edge_flush()
        return self.drain()


# ═══════════════ 训练+导出（S1 用） ═══════════════

TINY_CFG = """seed: 42
device: cpu
data:
  appliance: {name}
  window_size: 64
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  max_samples_train: 20000
  max_samples_val: 10000
  max_samples_test: 2000
model:
  input_dim: 1
  d_model: 32
  nhead: 4
  num_layers: 1
  dim_feedforward: 64
  dropout: 0.0
training:
  batch_size: 128
  epochs: 10
  lr: 0.0005
  weight_decay: 0.0001
  patience: 3
  grad_clip: 1.0
  loss: mse
metrics:
  on_threshold_watts: {thr}
"""


def train_and_export(sc_train, out_dir):
    """渲染训练 6s 序列 → 逐电器 train.py + export_edge_bundle.py。返回 model.bin 路径列表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    agg, tk, td = sc_train.render_6s()
    bundles = []
    for name, tgt, thr in (("kettle", tk, 500.0), ("dish_washer", td, 200.0)):
        npz = out_dir / f"train_{name}.npz"
        np.savez(npz, aggregate=agg.astype(np.float32), target=tgt.astype(np.float32))
        cfg = out_dir / f"cfg_{name}.yaml"
        cfg.write_text(TINY_CFG.format(name=name, thr=thr), encoding="utf-8")
        run = out_dir / f"run_{name}"
        r = subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--config", str(cfg),
                            "--data-path", str(npz), "--out", str(run), "--no-log"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit(f"训练失败：{name}")
        bdir = out_dir / f"bundle_{name}"
        r = subprocess.run([sys.executable, str(ROOT / "scripts/export_edge_bundle.py"),
                            "--run-dir", str(run), "--stats-from", str(npz),
                            "--out", str(bdir), "--no-log"], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit(f"导出失败：{name}")
        bundles.append(bdir / "model.bin")
        print(f"  训练+导出完成：{name}（bundle_{name}/model.bin）")
    return bundles


# ═══════════════ 指标 ═══════════════

def segments(flags):
    """连续 True 段 [(start, end)]（end exclusive）。"""
    segs, s = [], None
    for i, v in enumerate(flags):
        if v and s is None:
            s = i
        elif not v and s is not None:
            segs.append((s, i))
            s = None
    if s is not None:
        segs.append((s, len(flags)))
    return segs


def point_f1(truth_on, pred_on):
    tp = int(np.sum(truth_on & pred_on))
    fp = int(np.sum(~truth_on & pred_on))
    fn = int(np.sum(truth_on & ~pred_on))
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return (2 * p * r / max(1e-9, p + r), p, r, tp, fp, fn)


def event_f1(truth_flags, pred_flags):
    """事件级：真值段与预测段任一桶重叠即匹配；返回 (f1, p, r, n_t, n_p, 平均起始偏移桶)。"""
    tseg, pseg = segments(truth_flags), segments(pred_flags)
    used, offs = set(), []
    for ps, pe in pseg:
        best = None
        for ti, (ts_, te) in enumerate(tseg):
            if ti in used:
                continue
            if ps < te and ts_ < pe:
                if best is None or ts_ < tseg[best][0]:
                    best = ti
        if best is not None:
            used.add(best)
            offs.append(ps - tseg[best][0])
    tp = len(used)
    p = tp / max(1, len(pseg))
    r = tp / max(1, len(tseg))
    f1 = 2 * p * r / max(1e-9, p + r)
    return (f1, p, r, len(tseg), len(pseg), float(np.mean(offs)) if offs else None)


# ═══════════════ S1：合成流 ═══════════════

def run_synth_stream(runner, sc, record_path=None, record_buckets=200):
    """推送整条场景流（含 D4 短缺口填充与长缺口死寂），返回 (results, truth_bucket, analytic, 计时)。"""
    results = {name: [] for name, *_ in runner.models}
    truth_bucket = {name: np.zeros(sc.n, bool) for name, *_ in runner.models}
    analytic = {}
    rec_rows, rec_ts, rec_truth = [], [], {name: [] for name, *_ in runner.models}
    g0 = (B0 + sc.gap_short[0]) * TICK if sc.gap_short else -1.0
    g1 = g0 + sc.gap_short[1] if sc.gap_short else -1.0
    long_gap = set(range(sc.gap_long[0], sc.gap_long[0] + sc.gap_long_buckets)) if sc.gap_long else set()
    prev_row = None
    t_engine, n_cycles = 0.0, 0
    t_wall0 = time.perf_counter()
    for i, ts, wave, truth, ana in sc.iter_wave():
        if i in long_gap:
            continue                       # 终端死寂：不推包（引擎在恢复包上 carry→重置）
        analytic[B0 + i] = ana
        for name in truth_bucket:
            truth_bucket[name][i] = bool(truth[name].max() > 0)
        t0 = time.perf_counter()
        for k in range(len(ts)):
            if g0 <= ts[k] < g1 and prev_row is not None:
                runner.push(prev_row, float(ts[k]))   # D4：丢帧槽位用上一包数据填充
                pushed = prev_row
            else:
                row = wave[k]
                runner.push(row, float(ts[k]))
                prev_row = row
                pushed = row
            n_cycles += 1
            if record_path is not None and i < record_buckets:
                rec_rows.append(pushed)
                rec_ts.append(float(ts[k]))
                for name, *_ in runner.models:
                    rec_truth[name].append(int(truth[name][k]))
        t_engine += time.perf_counter() - t0
        for name, lst in runner.drain().items():
            results[name].extend(lst)
    for name, lst in runner.flush_and_drain().items():
        results[name].extend(lst)
    wall = time.perf_counter() - t_wall0
    if record_path is not None:
        meta = {"format": "nilm-edge wave recording v1", "channels": "[uA,iA,uB,iB,uC,iC]",
                "points_per_packet": NPT, "fs": FS, "b0": B0, "note": "S1 合成录制段"}
        kw = {"wave": np.asarray(rec_rows, np.float32), "ts": np.asarray(rec_ts, np.float64),
              "meta": json.dumps(meta, ensure_ascii=False)}
        for name, *_ in runner.models:
            kw[f"truth_on_{name}"] = np.asarray(rec_truth[name], np.int8)
        np.savez(record_path, **kw)
        print(f"  录制段已写出：{record_path}（{len(rec_ts)} 包，前 {record_buckets} 桶）")
    return results, truth_bucket, analytic, {"wall_s": wall, "engine_s": t_engine,
                                             "n_cycles": n_cycles}


def evaluate_synth(results, truth_bucket, analytic, sc, models, timing):
    """S1 功能验收：返回 (verdicts dict, metrics dict)。"""
    W = models[0][2]
    n = sc.n
    R_i = sc.gap_long[0] + sc.gap_long_buckets if sc.gap_long else 0
    n_carry = min(300, sc.gap_long_buckets) if sc.gap_long else 0
    expected = (n - sc.gap_long_buckets) + n_carry - 2 * (W - 1)
    short_span = set()
    if sc.gap_short:
        short_span = set(range(B0 + sc.gap_short[0],
                               B0 + sc.gap_short[0] + int(np.ceil(sc.gap_short[1] / TICK)) + 1))
    v, m = {}, {}

    def check(key, ok, detail):
        v[key] = bool(ok)
        print(f"  [{'✓' if ok else '✗'}] {key}: {detail}")

    # 语义检查（用任一模型结果——共享功率流）
    first = results[models[0][0]]
    centers = sorted(int(round(c / TICK)) for c, *_ in first)  # 绝对桶
    print("── 语义 ──")
    check("结果条数", all(len(results[nm]) == expected for nm, *_ in models),
          f"期望 {expected}（每模型），实际 " +
          "/".join(str(len(results[nm])) for nm, *_ in models))
    check("暖机首中心", centers and centers[0] == B0 + W // 2,
          f"首中心桶 {centers[0] if centers else None}（期望 {B0 + W // 2}）")
    check("末端中心滞后", centers and centers[-1] == B0 + n - 1 - (W - 1) + W // 2,
          f"末中心桶 {centers[-1] if centers else None}（期望 {B0 + n - 1 - (W - 1) + W // 2}，"
          f"滞后 {(W - 1 - W // 2) * TICK:.0f}s）")
    if sc.gap_long:
        post = [c for c in centers if c >= B0 + R_i]
        no_warm = not any(B0 + R_i <= c < B0 + R_i + W // 2 for c in centers)
        check("长缺口重置+再暖机", no_warm and post and post[0] == B0 + R_i + W // 2,
              f"恢复桶 {B0 + R_i} 后首中心 {post[0] if post else None}（期望 {B0 + R_i + W // 2}），"
              f"暖期无结果={no_warm}")
    if sc.gap_short:
        cont = all((B0 + i) in set(centers) for i in
                   range(sc.gap_short[0], sc.gap_short[0] + int(np.ceil(sc.gap_short[1] / TICK))))
        check("短缺口 D4 连续性", cont, "缺口跨度内中心结果连续（填充语义生效）")

    print("── 特征层 ──")
    skip = set()
    if sc.gap_short:
        skip |= short_span
    if sc.gap_long:
        skip |= set(range(B0 + sc.gap_long[0], B0 + R_i))       # carry 段（陈旧值）
    errs = [abs(agg - analytic[b]) / max(1.0, analytic[b])
            for c, agg, _p, _o in first
            for b in [int(round(c / TICK))] if b in analytic and b not in skip]
    max_err = max(errs) if errs else 1.0
    check("功率对解析值", max_err <= ACC_ANALYTIC,
          f"max rel err {max_err:.2e} ≤ {ACC_ANALYTIC}（n={len(errs)}，剔除缺口/carry 段）")
    m["analytic_max_rel_err"] = max_err

    print("── 事件/能耗（每模型）──")
    for name, _mid, _w, thr in models:
        res = results[name]
        cb = np.array([int(round(c / TICK)) for c, *_ in res])
        on = np.array([bool(o) for *_x, o in res], bool)
        pred = np.array([p for _c, _a, p, _o in res])
        idx = cb - B0                                       # 场景坐标
        t_on = truth_bucket[name][idx]
        f1p, pr, rc, tp, fp, fn = point_f1(t_on, on)
        f1e, pe, re_, nt, npd, off = event_f1(t_on, on)
        kwh_pred = float(pred.sum() * TICK / 3.6e6)
        tgt = np.array([sc.loads(int(i))[2][name] for i in idx])
        kwh_truth = float(tgt.sum() * TICK / 3.6e6)
        kwh_err = (kwh_pred - kwh_truth) / max(1e-9, kwh_truth)
        check(f"{name} 点级 F1", f1p >= ACC_F1_POINT,
              f"F1 {f1p:.4f}（P {pr:.3f} / R {rc:.3f}，tp/fp/fn={tp}/{fp}/{fn}）")
        check(f"{name} 事件级 F1", f1e >= ACC_F1_EVENT,
              f"F1 {f1e:.4f}（真值 {nt} / 预测 {npd} 事件，平均起始偏移 {off} 桶）")
        check(f"{name} kWh", abs(kwh_err) <= ACC_KWH,
              f"pred {kwh_pred:.3f} vs truth {kwh_truth:.3f} kWh（rel {kwh_err * 100:+.2f}%）")
        m[name] = {"point_f1": f1p, "event_f1": f1e, "kwh_pred": kwh_pred,
                   "kwh_truth": kwh_truth, "kwh_err_rel": kwh_err, "n_results": len(res)}

    print("── 性能 ──")
    stream_sec = timing["n_cycles"] / F0
    x_rt = stream_sec / max(1e-9, timing["wall_s"])
    n_ticks = sum(len(results[nm]) for nm, *_ in models)
    ms_tick = timing["engine_s"] * 1000 / max(1, n_ticks)
    check("全速倍率", x_rt >= MIN_X_REALTIME,
          f"{x_rt:.0f}× 实时（{stream_sec / 3600:.2f}h 流 / {timing['wall_s']:.1f}s 推送），"
          f"引擎 {ms_tick:.2f} ms/tick")
    m["x_realtime"] = x_rt
    m["ms_per_tick"] = ms_tick
    return v, m


# ═══════════════ S2：录制回放 ═══════════════

def run_replay(runner, files, realtime=False):
    results = {name: [] for name, *_ in runner.models}
    truth_cycles = {name: [] for name, *_ in runner.models}
    ts_all = []
    t_engine, n_cycles = 0.0, 0
    t_wall0 = time.perf_counter()
    vr_ms = [[] for _ in range(3)]
    for f in files:
        d = np.load(f, allow_pickle=False)
        wave, ts = d["wave"], d["ts"]
        if wave.ndim != 2 or wave.shape[1] != NPT * CH:
            raise SystemExit(f"{f}: wave 形状 {wave.shape} 不符 [n, {NPT * CH}]")
        tkeys = [k[len("truth_on_"):] for k in d.files if k.startswith("truth_on_")]
        for k in range(len(ts)):
            t0 = time.perf_counter()
            runner.push(wave[k], float(ts[k]))
            t_engine += time.perf_counter() - t0
            n_cycles += 1
            ts_all.append(float(ts[k]))
            for nm in tkeys:
                truth_cycles[nm].append(int(d["truth_on_" + nm][k]))
            if k % 50 == 0:                       # 轻量子采样：每 50 包测一次相电压
                wv = wave[k].reshape(NPT, CH)
                for ph in range(3):
                    vr_ms[ph].append(float(np.sqrt(np.mean(wv[:, ph * 2] ** 2))))
            if realtime and k + 1 < len(ts):
                time.sleep(min(float(ts[k + 1] - ts[k]), 1.0))
            if k % 100 == 99:
                for name, lst in runner.drain().items():
                    results[name].extend(lst)
        for name, lst in runner.drain().items():
            results[name].extend(lst)
    for name, lst in runner.flush_and_drain().items():
        results[name].extend(lst)
    wall = time.perf_counter() - t_wall0
    quality = {
        "n_packets": n_cycles,
        "span_sec": (ts_all[-1] - ts_all[0]) if ts_all else 0.0,
        "dropouts_gt_21ms": int(sum(1 for a, b in zip(ts_all, ts_all[1:]) if b - a > 0.021)),
        "ts_monotonic": bool(all(b >= a for a, b in zip(ts_all, ts_all[1:]))),
        "vrms_per_phase": [round(float(np.mean(v)), 1) if v else None for v in vr_ms],
    }
    return results, truth_cycles, ts_all, quality, {"wall_s": wall, "engine_s": t_engine,
                                                    "n_cycles": n_cycles}


def evaluate_replay(results, truth_cycles, ts_all, quality, models, timing, files):
    v, m = {}, {"quality": quality, "files": [str(f) for f in files]}

    def check(key, ok, detail):
        v[key] = bool(ok)
        print(f"  [{'✓' if ok else '✗'}] {key}: {detail}")

    print("── 录制质量 ──")
    check("时间戳单调", quality["ts_monotonic"],
          f"{quality['n_packets']} 包 / {quality['span_sec'] / 3600:.2f}h / "
          f"疑似丢帧 {quality['dropouts_gt_21ms']} 处 / Vrms {quality['vrms_per_phase']}")
    print("── 模型输出（无真值=描述性；有真值=事件指标）──")
    for name, _mid, _w, thr in models:
        res = results[name]
        if not res:
            check(f"{name} 有结果", False, "无任何结果（检查录制长度是否 ≥ 窗口×6s）")
            continue
        on = np.array([bool(o) for *_x, o in res])
        pred = np.array([p for _c, _a, p, _o in res])
        kwh = float(pred.sum() * TICK / 3.6e6)
        info = (f"n={len(res)} | ON {on.mean() * 100:.2f}% | 事件段 {len(segments(on))} | "
                f"pred kWh {kwh:.3f}")
        if truth_cycles.get(name):
            # 周波真值 → 桶真值（≥50% 周波 ON）
            tb = {}
            for t, flag in zip(ts_all, truth_cycles[name]):
                tb.setdefault(int(np.floor(t / TICK)), []).append(flag)
            cb = np.array([int(round(c / TICK)) for c, *_ in res])
            t_on = np.array([np.mean(tb.get(int(b), [0])) >= 0.5 for b in cb], bool)
            f1p, pr, rc, tp, fp, fn = point_f1(t_on, on)
            f1e, pe, re_, nt, npd, off = event_f1(t_on, on)
            info += (f" | 点级 F1 {f1p:.4f}（P {pr:.3f}/R {rc:.3f}）| 事件级 F1 {f1e:.4f}"
                     f"（真值 {nt}/预测 {npd}）")
            m[name] = {"point_f1": f1p, "event_f1": f1e, "kwh_pred": kwh, "n_results": len(res)}
        else:
            m[name] = {"kwh_pred": kwh, "n_results": len(res), "on_frac": float(on.mean())}
        print(f"  [i] {name}: {info}")
    stream_sec = timing["n_cycles"] / F0
    x_rt = stream_sec / max(1e-9, timing["wall_s"])
    n_ticks = sum(len(results[nm]) for nm, *_ in models)
    print(f"  [i] 性能：{x_rt:.0f}× 实时（{stream_sec / 3600:.2f}h / {timing['wall_s']:.1f}s），"
          f"引擎 {timing['engine_s'] * 1000 / max(1, n_ticks):.2f} ms/tick")
    m["x_realtime"] = x_rt
    return v, m


def compare_with(report_a, report_b):
    """两份报告逐条对拍（确定性回归）。"""
    print("── 对拍（确定性回归）──")
    ok_all, total = True, 0
    for name, res_b in report_b["results"].items():
        res_a = report_a["results"].get(name)
        if res_a is None:
            print(f"  [✗] {name}: 基准报告无此模型")
            ok_all = False
            continue
        da = {round(c, 6): (p, o) for c, _ag, p, o in res_a}
        matched, bad = 0, 0
        for c, _ag, p, o in res_b:
            key = round(c, 6)
            if key in da:
                total += 1
                if da[key][0] != p or da[key][1] != int(o):
                    bad += 1
                matched += 1
        ok = bad == 0 and matched > 0
        ok_all = ok_all and ok
        print(f"  [{'✓' if ok else '✗'}] {name}: 交集 {matched} 条，不一致 {bad} 条")
    return ok_all, total


# ═══════════════ CLI ═══════════════

def main():
    ap = argparse.ArgumentParser(description="边缘流式推理分阶段验证 harness")
    ap.add_argument("--mode", choices=["synth", "replay"], default="synth")
    ap.add_argument("--hours", type=float, default=4.0, help="S1 流时长（默认 4h）")
    ap.add_argument("--out-dir", default="reports/edge_stream_test")
    ap.add_argument("--bundle", action="append", default=[],
                    help="model.bin 路径（可多次；synth 模式给出则跳过训练）")
    ap.add_argument("--record", default="", help="S1：录制前 N 桶到该 npz（S2 自证用）")
    ap.add_argument("--record-buckets", type=int, default=200)
    ap.add_argument("--wave-file", action="append", default=[], help="S2：录制 npz（可多次）")
    ap.add_argument("--realtime", action="store_true", help="S2：按时间戳 pacing 回放")
    ap.add_argument("--compare-with", default="", help="S2：与此前 report.json 逐条对拍")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from runlog import setup_run_log
    setup_run_log(out_dir / ("replay.log" if args.mode == "replay" else "synth.log"),
                  enabled=not args.no_log)

    lib = build_lib()
    if args.mode == "synth":
        print(f"══ S1 合成周波流验证（{args.hours:g}h）══")
        if args.bundle:
            bundles = [Path(b) for b in args.bundle]
            print(f"  使用既有部署包：{[str(b.parent.name) for b in bundles]}")
        else:
            print("  ① 渲染训练序列并训练 tiny 模型（6 天合成数据）…")
            sc_train = Scenario(101, 86400, 432, 24)
            bundles = train_and_export(sc_train, out_dir / "train")
        print("  ② 构建流场景（4h：kettle/dw 事件 + 20s D4 短缺口 + 31min 长缺口）…")
        n_buckets = int(args.hours * 600)
        sc = Scenario(202, n_buckets, 12, 2, gap_short=(150, 20.0), gap_long=(1000, 1860.0))
        runner = EdgeRunner(lib, bundles)
        print(f"  ③ 流式推送（{n_buckets} 桶 = {n_buckets * CPB} 包）…")
        rec = Path(args.record) if args.record else None
        if rec is not None:
            rec.parent.mkdir(parents=True, exist_ok=True)
        results, truth_bucket, analytic, timing = run_synth_stream(
            runner, sc, record_path=rec, record_buckets=args.record_buckets)
        print("  ④ 功能验收…")
        verdicts, metrics = evaluate_synth(results, truth_bucket, analytic, sc,
                                           runner.models, timing)
        runner.lib.nilm_edge_shutdown()
        report = {"mode": "synth", "verdicts": verdicts, "metrics": metrics,
                  "results": results}
        ok = all(verdicts.values())
    else:
        if not args.wave_file or not args.bundle:
            raise SystemExit("replay 模式需要 --wave-file 与 --bundle")
        print(f"══ S2 录制回放验证（{len(args.wave_file)} 个文件）══")
        runner = EdgeRunner(lib, [Path(b) for b in args.bundle])
        results, truth_cycles, ts_all, quality, timing = run_replay(
            runner, [Path(f) for f in args.wave_file], realtime=args.realtime)
        verdicts, metrics = evaluate_replay(results, truth_cycles, ts_all, quality,
                                            runner.models, timing, args.wave_file)
        runner.lib.nilm_edge_shutdown()
        report = {"mode": "replay", "verdicts": verdicts, "metrics": metrics,
                  "results": results}
        ok = all(verdicts.values())
        if args.compare_with:
            base = json.loads(Path(args.compare_with).read_text(encoding="utf-8"))
            ok_cmp, n_cmp = compare_with(base, report)
            report["compare_with"] = {"ok": ok_cmp, "matched": n_cmp,
                                      "base": args.compare_with}
            ok = ok and ok_cmp

    rp = out_dir / "report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告：{rp}")
    print("总结论：", "✅ 通过" if ok else "❌ 存在未通过项（见上）")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
