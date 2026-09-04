import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from .metrics import regression_metrics


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
        grad_clip=1.0):
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
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               **{f"train_{k}": v for k, v in tm.items()},
               **{f"val_{k}": v for k, v in vm.items()}}
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train MAE={tm['mae']:.2f} | val MAE={vm['mae']:.2f} | "
            f"train R2={tm['r2']:.4f} | val R2={vm['r2']:.4f}"
        )

        if vm["mae"] < best:
            best = vm["mae"]
            best_epoch = epoch
            bad = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad += 1
            if bad >= patience:
                break

    return history, best_epoch
