"""scripts/prepare_ukdale.py 端到端验证（合成 NILMTK 风格 h5，无 torch 依赖）。

运行：cd 仓库根目录 && python -m pytest tests/test_prepare_ukdale.py -q
     或直接 python tests/test_prepare_ukdale.py（会创建临时文件）。
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_ukdale.py"

# 三段烧水事件起点/长度（与 make_toy_h5 一致，供校验）
EVENTS = [(500, 40), (1200, 80), (2500, 25)]


def make_toy_h5(path):
    """按脚本契约构造 /building1/elec/meter{1,2,10}/power (N,2)。"""
    n = 3000
    t0 = 1_500_000_000  # unix 秒
    ts = t0 + np.arange(n) * 6.0
    rng = np.random.default_rng(0)

    kettle = np.zeros(n)
    for s, d in EVENTS:
        kettle[s:s + d] = 2000 + rng.normal(0, 30, d)
    mains1 = 300 + rng.normal(0, 20, n) + kettle  # 总负荷包含 kettle 事件
    mains2 = 250 + rng.normal(0, 15, n)

    with h5py.File(path, "w") as f:
        for mid, sig in [(1, mains1), (2, mains2), (10, kettle)]:
            grp = f.create_group(f"building1/elec/meter{mid}")
            grp.create_dataset("power", data=np.stack([ts, sig], axis=1))
    return n, mains1, mains2, kettle


def test_list_meters():
    with tempfile.TemporaryDirectory() as d:
        h5 = Path(d) / "ukdale_toy.h5"
        make_toy_h5(h5)
        r = subprocess.run([sys.executable, str(SCRIPT), "--h5-path", str(h5),
                            "--list-meters"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        # 表号 1/2/10 都要列出来
        assert "meter 1" in r.stdout and "meter 2" in r.stdout \
            and "meter 10" in r.stdout


def test_prepare_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        h5 = Path(d) / "ukdale_toy.h5"
        n, mains1, mains2, kettle = make_toy_h5(h5)
        out = Path(d) / "ukdale_prepared.npz"
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--h5-path", str(h5),
             "--mains-ids", "1,2", "--kettle-meter-id", "10",
             "--out", str(out)],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

        z = np.load(out)
        agg, tgt = z["aggregate"], z["target"]
        assert agg.dtype == np.float32 and tgt.dtype == np.float32
        assert agg.shape == tgt.shape == (n,)
        assert (tgt >= 0).all() and (agg >= 0).all()

        # 无事件处：target≈0，aggregate≈mains1+mains2
        i_off = 100
        assert abs(tgt[i_off]) < 50
        assert abs(agg[i_off] - (mains1[i_off] + mains2[i_off])) < 60
        # 事件中：target≈2000，aggregate（含 kettle 事件）≈mains1+mains2
        s, d = EVENTS[1]
        i_on = s + d // 2
        assert 1500 < tgt[i_on] < 2500
        assert abs(agg[i_on] - (mains1[i_on] + mains2[i_on])) < 100

        spec_path = out.with_suffix(".data_spec.json")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        assert spec["kettle_meter_id"] == 10
        assert spec["mains_meter_ids_used"] == [1, 2]
        assert spec["n_output"] == n
        assert spec["sample_period_sec"] == 6
        assert spec["unit"] == "W"


def test_missing_kettle_id_fails_helpfully():
    with tempfile.TemporaryDirectory() as d:
        h5 = Path(d) / "ukdale_toy.h5"
        make_toy_h5(h5)
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--h5-path", str(h5),
             "--kettle-meter-id", "99", "--out", str(Path(d) / "x.npz")],
            capture_output=True, text=True)
        assert r.returncode != 0
        assert "99" in r.stderr and "--list-meters" in r.stderr


if __name__ == "__main__":
    test_list_meters()
    test_prepare_end_to_end()
    test_missing_kettle_id_fails_helpfully()
    print("test_prepare_ukdale: all assertions passed")
