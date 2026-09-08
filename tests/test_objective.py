"""src/objective.py 纯函数单测（无 torch 依赖）。

运行：cd 仓库根目录 && python -m pytest tests/test_objective.py -q
     或直接 python tests/test_objective.py（import 路径见下）。
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.objective import (
    DEFAULT_WEIGHTS,
    composite_score,
    gate_results,
    row_score,
)


def test_composite_zero_when_perfect():
    # 完美预测：MAE=0、F1=1、能量误差=0 → S=0
    assert composite_score(0.0, 1.0, 0.0) == 0.0


def test_composite_default_formula():
    # S = 0.4*(100/2000) + 0.4*(1-0.8) + 0.2*0.1 = 0.02+0.08+0.02 = 0.12
    s = composite_score(100.0, 0.8, 0.1)
    assert math.isclose(s, 0.12, rel_tol=1e-9)
    # F1 拖后腿比 MAE 更伤（业务口径：开没开对比曲线贴不贴更重要）
    s_low_f1 = composite_score(50.0, 0.6, 0.05)
    s_ok_f1 = composite_score(200.0, 0.9, 0.1)
    assert s_low_f1 > s_ok_f1


def test_composite_custom_weights():
    w = {"mae": 1.0, "f1": 0.0, "energy_error": 0.0}
    assert math.isclose(composite_score(200.0, 0.0, 1.0, weights=w,
                                         mae_norm_watts=2000.0), 0.1)


def test_row_score_from_history_row():
    row = {"val_mae": 100.0, "val_f1": 0.8, "val_energy_error": 0.1,
           "val_rmse": 999.0}  # 多余字段不影响
    assert math.isclose(row_score(row), 0.12, rel_tol=1e-9)
    # 自定义 objective（norm 换 1000）会改变结果
    obj = {"mae_norm_watts": 1000.0, "weights": DEFAULT_WEIGHTS}
    assert math.isclose(row_score(row, obj), 0.4 * 100 / 1000 + 0.08 + 0.02,
                        rel_tol=1e-9)


def test_gates_disabled_by_default():
    row = {"val_f1": 0.1, "val_recall": 0.1, "val_energy_error": 0.9}
    ok, reasons = gate_results(row)
    assert ok and reasons == []


def test_gates_enabled():
    good = {"val_f1": 0.8, "val_recall": 0.75, "val_energy_error": 0.1}
    ok, reasons = gate_results(good, {"enabled": True})
    assert ok and reasons == []

    bad_f1 = {"val_f1": 0.7, "val_recall": 0.75, "val_energy_error": 0.1}
    ok, reasons = gate_results(bad_f1, {"enabled": True})
    assert not ok and "f1<0.75" in ";".join(reasons)

    bad_ee = {"val_f1": 0.8, "val_recall": 0.75, "val_energy_error": -0.4}
    ok, reasons = gate_results(bad_ee, {"enabled": True})
    assert not ok and "|ee|>0.15" in ";".join(reasons)


if __name__ == "__main__":
    test_composite_zero_when_perfect()
    test_composite_default_formula()
    test_composite_custom_weights()
    test_row_score_from_history_row()
    test_gates_disabled_by_default()
    test_gates_enabled()
    print("test_objective: all assertions passed")
