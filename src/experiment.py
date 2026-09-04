import random, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import NILMTransformer
from .data import build_splits
from .metrics import regression_metrics
from .trainer import fit, run_epoch


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_experiment(aggregate, target, cfg, out_dir):
    seed_everything(cfg.get("seed", 42))
    device = select_device(cfg.get("device", "auto"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["training"]
    window = int(dcfg["window_size"])

    train_ds, val_ds, test_ds = build_splits(
        aggregate, target, window,
        dcfg["train_ratio"], dcfg["val_ratio"],
        dcfg.get("max_samples_train"), dcfg.get("max_samples_val"),
        dcfg.get("max_samples_test")
    )
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=tcfg["batch_size"], shuffle=False)

    model = NILMTransformer(**mcfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    start = time.time()
    history, best_epoch = fit(
        model, train_loader, val_loader, device, optimizer,
        tcfg["epochs"], tcfg["patience"],
        train_ds.y_mean, train_ds.y_std,
        out_dir / "best.pt",
        cfg["metrics"]["on_threshold_watts"],
        tcfg["loss"], tcfg["grad_clip"]
    )

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    _, yt, yp = run_epoch(model, test_loader, device, None, tcfg["loss"], tcfg["grad_clip"])
    yt = yt * train_ds.y_std + train_ds.y_mean
    yp = yp * train_ds.y_std + train_ds.y_mean
    test_metrics = regression_metrics(yt, yp, cfg["metrics"]["on_threshold_watts"])

    result = {
        "best_epoch": best_epoch,
        "device": str(device),
        "runtime_sec": time.time() - start,
        "test": test_metrics,
        "config": cfg,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
