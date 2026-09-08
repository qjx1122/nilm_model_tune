"""业务综合分 S 模块（纯函数，无 torch 依赖，trainer 与 tune 共用）。

Kettle 业务口径（REPORT_TEST.md 专题「Transformer-NILM 调参方案设计」）：
    S = w_mae * (MAE / mae_norm_watts) + w_f1 * (1 - F1) + w_ee * |energy_error|
默认权重：MAE 0.4（除以 2000 W 归一，kettle 量程约 0–3000 W）、F1 0.4、能量误差 0.2。
S 越小越好；先用业务硬门槛（gate）过滤，再按 S 排序。
"""

DEFAULT_WEIGHTS = {"mae": 0.4, "f1": 0.4, "energy_error": 0.2}
DEFAULT_MAE_NORM_WATTS = 2000.0
DEFAULT_GATES = {
    "enabled": False,
    "f1_min": 0.75,
    "recall_min": 0.70,
    "abs_energy_error_max": 0.15,
}


def composite_score(mae, f1, energy_error, weights=None, mae_norm_watts=DEFAULT_MAE_NORM_WATTS):
    """由三项指标算综合分 S（越小越好）。"""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    mae_term = float(mae) / float(mae_norm_watts)
    f1_term = max(0.0, 1.0 - float(f1))
    return w["mae"] * mae_term + w["f1"] * f1_term + w["energy_error"] * abs(float(energy_error))


def row_score(row, objective=None, prefix="val_"):
    """从 history 行（含 prefix+mae / +f1 / +energy_error 字段）算综合分。

    objective 可选字段：weights、mae_norm_watts（缺省用默认值）。
    """
    obj = objective or {}
    return composite_score(
        row[prefix + "mae"],
        row[prefix + "f1"],
        row[prefix + "energy_error"],
        weights=obj.get("weights"),
        mae_norm_watts=obj.get("mae_norm_watts", DEFAULT_MAE_NORM_WATTS),
    )


def gate_results(row, gates=None, prefix="val_"):
    """业务硬门槛检查 -> (ok, 失败原因列表)。gates 未启用时恒通过。"""
    g = {**DEFAULT_GATES, **(gates or {})}
    if not g.get("enabled", False):
        return True, []
    reasons = []
    if row[prefix + "f1"] < g["f1_min"]:
        reasons.append("f1<%.2f" % g["f1_min"])
    if row[prefix + "recall"] < g["recall_min"]:
        reasons.append("recall<%.2f" % g["recall_min"])
    if abs(row[prefix + "energy_error"]) > g["abs_energy_error_max"]:
        reasons.append("|ee|>%.2f" % g["abs_energy_error_max"])
    return len(reasons) == 0, reasons
