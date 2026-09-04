from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, aggregate, target, window_size, indices, x_mean, x_std, y_mean, y_std):
        self.aggregate = aggregate.astype(np.float32)
        self.target = target.astype(np.float32)
        self.window = int(window_size)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.x_mean, self.x_std = float(x_mean), float(x_std)
        self.y_mean, self.y_std = float(y_mean), float(y_std)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        center = int(self.indices[i])
        half = self.window // 2
        start = center - half
        end = start + self.window
        x = (self.aggregate[start:end] - self.x_mean) / self.x_std
        y = (self.target[center] - self.y_mean) / self.y_std
        return torch.from_numpy(x[:, None]), torch.tensor(y, dtype=torch.float32)


def make_synthetic_signal(n=12000, seed=42):
    """Only for code-path smoke testing, not a public-dataset result."""
    rng = np.random.default_rng(seed)
    base = 350 + 80 * np.sin(np.arange(n) / 160.0) + rng.normal(0, 25, n)
    target = np.zeros(n, dtype=np.float32)
    for _ in range(max(1, n // 1200)):
        s = int(rng.integers(100, n - 400))
        dur = int(rng.integers(20, 120))
        amp = float(rng.choice([1200, 1800, 2200]))
        target[s:s+dur] = amp + rng.normal(0, 30, dur)
    aggregate = base + target + rng.normal(0, 20, n)
    return aggregate.astype(np.float32), target.astype(np.float32)


def build_splits(aggregate, target, window, train_ratio=0.7, val_ratio=0.15,
                 max_train=None, max_val=None, max_test=None):
    n = len(aggregate)
    a = int(n * train_ratio)
    b = int(n * (train_ratio + val_ratio))
    # Windows are restricted so no sample crosses split boundaries.
    train_centers = np.arange(window // 2, a - window // 2, dtype=np.int64)
    val_centers = np.arange(a + window // 2, b - window // 2, dtype=np.int64)
    test_centers = np.arange(b + window // 2, n - window // 2, dtype=np.int64)

    if max_train and len(train_centers) > max_train:
        train_centers = np.linspace(train_centers[0], train_centers[-1], max_train).astype(np.int64)
    if max_val and len(val_centers) > max_val:
        val_centers = np.linspace(val_centers[0], val_centers[-1], max_val).astype(np.int64)
    if max_test and len(test_centers) > max_test:
        test_centers = np.linspace(test_centers[0], test_centers[-1], max_test).astype(np.int64)

    # Fit normalization on TRAIN ONLY.
    train_x = aggregate[:a]
    train_y = target[:a]
    x_mean, x_std = float(train_x.mean()), float(train_x.std() + 1e-6)
    y_mean, y_std = float(train_y.mean()), float(train_y.std() + 1e-6)

    return (
        WindowDataset(aggregate, target, window, train_centers, x_mean, x_std, y_mean, y_std),
        WindowDataset(aggregate, target, window, val_centers, x_mean, x_std, y_mean, y_std),
        WindowDataset(aggregate, target, window, test_centers, x_mean, x_std, y_mean, y_std),
    )


def load_nilmbench_labels(labels_path, class_name="kettle"):
    """Load x_agg/y_power from NILMbench processed UK-DALE labels_and_index.npz.
    x_agg is an 11-point aggregate context. This loader is intentionally separate
    from the 16 kHz waveform loader.
    """
    z = np.load(labels_path, allow_pickle=True)
    names = [str(x).lower() for x in z["class_names"]]
    idx = names.index(class_name.lower())
    x = z["x_agg"].astype(np.float32)
    y = z["y_power"][:, idx].astype(np.float32)
    return x, y


def load_simple_npz(path, appliance_index=0):
    z = np.load(path)
    if "aggregate" not in z or "target" not in z:
        raise ValueError("NPZ must contain aggregate and target arrays")
    return z["aggregate"].astype(np.float32), z["target"].astype(np.float32)
