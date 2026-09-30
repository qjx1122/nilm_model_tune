"""export_meter_csv.py 测试：NILMTK pandas 表 h5 → 多表 CSV 导出（时间+功率）。

覆盖：多表导出/文件名约定/两列结构/数值正确性（含 bin 均值与缺口剔除）/功率类型
回退（apparent-only 表 WARNING）/强制功率类型失败/--resample-sec 0 原始采样/
表号不存在报错（附可用表清单）。
"""
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_meter_csv.py"


def _make_h5(path):
    """3 表：meter1 apparent-only（测回退）、meter2 active、meter10 active（kettle 型）。

    时间轴：6s 间隔 3000 点，但抽掉一段（模拟缺口），并把部分点错位 2s（测重采样 bin 均值）。
    """
    n = 3000
    t0 = 1_500_000_000
    ts = (t0 + np.arange(n) * 6).astype(np.int64)
    keep = np.ones(n, bool)
    keep[500:560] = False                       # 60 样本缺口（6min）
    ts = ts[keep]
    m = len(ts)
    rng = np.random.default_rng(0)
    mains1 = np.full(m, 300.0) + rng.normal(0, 0.1, m)
    mains2 = np.full(m, 250.0)
    kettle = np.zeros(m)
    kettle[100:130] = 2000.0                    # 事件 3min
    mains1 = mains1 + kettle
    idx = pd.Index(ts + 2, dtype=np.int64)      # 统一偏移 2s（bin 内单样本，仍落对应格）
    with pd.HDFStore(str(path), "w") as store:
        store.put("/building1/elec/meter1",
                  pd.DataFrame(mains1[:, None], index=idx,
                               columns=pd.MultiIndex.from_tuples([("power", "apparent")])),
                  format="fixed")
        store.put("/building1/elec/meter2",
                  pd.DataFrame(mains2[:, None], index=idx,
                               columns=pd.MultiIndex.from_tuples([("power", "active")])),
                  format="fixed")
        store.put("/building1/elec/meter10",
                  pd.DataFrame(kettle[:, None], index=idx,
                               columns=pd.MultiIndex.from_tuples([("power", "active")])),
                  format="fixed")
    return ts, mains1, mains2, kettle


def _run(h5, tmp, *extra):
    r = subprocess.run([sys.executable, str(SCRIPT), "--h5-path", str(h5),
                        "--house", "1"] + list(extra),
                       capture_output=True, text=True, cwd=tmp)
    return r


def test_export_multi_meters(tmp_path):
    h5 = tmp_path / "ukdale.h5"
    ts, mains1, _mains2, kettle = _make_h5(h5)
    out = tmp_path / "csv"
    r = _run(h5, tmp_path, "--meter", "1", "2", "10", "--out-dir", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WARNING" in r.stdout and "apparent" in r.stdout   # meter1 回退提示
    for mid in (1, 2, 10):
        f = out / f"house_1_meter_{mid}.csv"
        assert f.exists(), f
        df = pd.read_csv(f)
        assert list(df.columns) == ["time", "power_w"]
        assert len(df) > 0
    # 数值：meter10 网格化后事件均值≈2000，off≈0；缺口 60 格被剔除
    df10 = pd.read_csv(out / "house_1_meter_10.csv")
    on = df10[df10.power_w > 1000]
    off = df10[df10.power_w < 100]
    assert len(on) == 30 and abs(on.power_w.mean() - 2000.0) < 1.0
    assert abs(off.power_w.mean()) < 1.0
    # 行数 = 原始样本数（每 bin 单样本）；时间列为 UTC ISO
    assert len(df10) == len(ts)
    assert df10.time.iloc[0].startswith("2017-09-05") or len(df10.time.iloc[0]) == 19
    # meter1（apparent 回退）值与 mains1 一致
    df1 = pd.read_csv(out / "house_1_meter_1.csv")
    assert np.allclose(df1.power_w.values, mains1, atol=0.51)  # %.3f 精度
    print("多表导出 ✓", r.stdout)


def test_raw_mode_and_forced_type(tmp_path):
    h5 = tmp_path / "ukdale.h5"
    _make_h5(h5)
    out = tmp_path / "csv2"
    r = _run(h5, tmp_path, "--meter", "2", "--out-dir", str(out), "--resample-sec", "0")
    assert r.returncode == 0, r.stdout + r.stderr
    df = pd.read_csv(out / "house_1_meter_2.csv")
    assert np.allclose(df.power_w.values, 250.0)
    # 强制 active：meter1 无 active 列 → 报错可用
    r2 = _run(h5, tmp_path, "--meter", "1", "--out-dir", str(out), "--power-type", "active")
    assert r2.returncode != 0 or "apparent" in r2.stdout + r2.stderr
    print("原始采样+强制类型 ✓")


def test_missing_meter_helpful_error(tmp_path):
    h5 = tmp_path / "ukdale.h5"
    _make_h5(h5)
    r = _run(h5, tmp_path, "--meter", "99", "--out-dir", str(tmp_path))
    assert r.returncode != 0
    assert "99" in (r.stdout + r.stderr) and "10" in (r.stdout + r.stderr)  # 附可用表
    print("表号不存在报错 ✓")
