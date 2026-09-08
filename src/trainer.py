import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from .metrics import regression_metrics
from .objective import composite_score


def _selection_score(vm, select_metric, objective):
    """按 select_metric 把 val 指标行折算成「选型分」（越小越好）。

    mae: 直接用 val MAE（默认，历史行为）；
    composite: 业务综合分 S（需要 objective 提供 weights / mae_norm_watts）。
    """
    if select_metric == "mae":
        return vm["mae"]
    if select_metric == "composite":
        obj = objective or {}
        return composite_score(
            vm["mae"], vm["f1"], vm["energy_error"],
            weights=obj.get("weights"),
            mae_norm_watts=obj.get("mae_norm_watts", 2000.0),
        )
    raise ValueError(f"unknown select_metric: {select_metric!r} (use 'mae' or 'composite')")


def run_epoch(model, loader, device, optimizer=None, loss_name="mse", grad_clip=1.0):
    train = optimizer is not None
    model.train(train)
    losses = []
    ys, ps = [], []
    loss_fn = nn.MSELoss() if loss_name == "mse" else nn.L1Loss()

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        if train:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        losses.append(float(loss.item()))
        ys.append(yb.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    return float(np.mean(losses)), np.concatenate(ys), np.concatenate(ps)


def fit(model, train_loader, val_loader, device, optimizer, epochs, patience,
        y_mean, y_std, checkpoint, on_threshold=500.0, loss_name="mse",
        grad_clip=1.0, select_metric="mae", objective=None):
    """训练 + 按 Validation 选型（早停 & checkpoint）。

    select_metric: "mae"（默认，历史行为）或 "composite"（业务综合分 S）；
    objective: {"weights": {...}, "mae_norm_watts": ...}，仅 composite 需要。
    选型判据只来自 Validation，与 Test 无关。
    """
    best = float("inf")
    best_epoch = 0
    bad = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, yt, yp = run_epoch(model, train_loader, device, optimizer, loss_name, grad_clip)
        val_loss, yv, pv = run_epoch(model, val_loader, device, None, loss_name, grad_clip)

        # Inverse normalization for human-readable metrics.
        yt_w = yt * y_std + y_mean
        yp_w = yp * y_std + y_mean
        yv_w = yv * y_std + y_mean
        pv_w = pv * y_std + y_mean

        tm = regression_metrics(yt_w, yp_w, on_threshold)
        vm = regression_metrics(yv_w, pv_w, on_threshold)
        score = _selection_score(vm, select_metric, objective)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "select_metric": select_metric, "val_score": score,
               **{f"train_{k}": v for k, v in tm.items()},
               **{f"val_{k}": v for k, v in vm.items()}}
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train MAE={tm['mae']:.2f} | val MAE={vm['mae']:.2f} | "
            f"train R2={tm['r2']:.4f} | val R2={vm['r2']:.4f} | "
            f"val score={score:.4f}"
        )

        if score < best:
            best = score
            best_epoch = epoch
            bad = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad += 1
            if bad >= patience:
                break

    return history, best_epoch
