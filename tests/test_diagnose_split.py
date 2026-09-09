"""scripts/diagnose_split.py CLI 验证（电器标签/默认阈值表/事件统计）。无 torch 依赖。

运行：cd 仓库根目录 && python -m pytest tests/test_diagnose_split.py -q
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_split.py"


def _make_npz(path, n=4000, amp=700.0):
    rng = np.random.default_rng(0)
    target = np.zeros(n, dtype=np.float32)
    for s in (500, 1500, 2500):
        target[s:s + 30] = amp + rng.normal(0, 20, 30)
    agg = (300 + rng.normal(0, 20, n) + target).astype(np.float32)
    np.savez(path, aggregate=agg, target=target)


def _run(npz, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--npz", str(npz), *extra],
                          capture_output=True, text=True)


def _on_events_total(stdout):
    total = 0
    for line in stdout.splitlines():
        parts = line.split()
        if parts and parts[0] in ("train", "val", "test"):
            total += int(parts[3])  # n=1, days=2, on_evt=3
    return total


def test_appliance_label_and_threshold_table():
    with tempfile.TemporaryDirectory() as d:
        npz = Path(d) / "dw.npz"
        _make_npz(npz, amp=700.0)
        # 默认 kettle：表值 500W（冻结口径）
        r = _run(npz)
        assert r.returncode == 0, r.stderr
        assert "appliance=kettle" in r.stdout
        assert "on_threshold=500.0W" in r.stdout
        assert _on_events_total(r.stdout) == 3  # 700W 事件全部检出
        # dish_washer：表值 20W
        r2 = _run(npz, "--appliance", "dish_washer")
        assert r2.returncode == 0, r2.stderr
        assert "appliance=dish_washer" in r2.stdout
        assert "on_threshold=20.0W" in r2.stdout
        assert _on_events_total(r2.stdout) == 3
        # 判读提示含电器名 + 常开型提示
        assert "dish_washer 关断时" in r2.stdout
        assert "常开型" in r2.stdout
        # 显式 --on-threshold 覆盖表值
        r3 = _run(npz, "--appliance", "dish_washer", "--on-threshold", "600")
        assert "on_threshold=600.0W" in r3.stdout
        # 阈值高于事件功率 → 0 事件（阈值确实生效）
        r4 = _run(npz, "--on-threshold", "800")
        assert _on_events_total(r4.stdout) == 0
