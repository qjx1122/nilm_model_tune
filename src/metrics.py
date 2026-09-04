import numpy as np
from sklearn.metrics import r2_score


def regression_metrics(y_true, y_pred, on_threshold=500.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = float(r2_score(y_true, y_pred)) if np.var(y_true) > 1e-12 else 0.0

    denom = max(abs(np.sum(y_true)), 1e-6)
    sae = float(abs(np.sum(y_true) - np.sum(y_pred)) / denom)
    energy_error = float((np.sum(y_pred) - np.sum(y_true)) / denom)

    yt = y_true >= on_threshold
    yp = y_pred >= on_threshold
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "sae": sae,
        "energy_error": energy_error,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
