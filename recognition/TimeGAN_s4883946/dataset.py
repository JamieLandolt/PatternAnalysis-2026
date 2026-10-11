"""Data loading and preprocessing for LOBSTER AMZN Level-10 order book data.

Pipeline:
    raw LOBSTER csv -> fixed time grid snapshots -> per-snapshot features
    -> chronological train/val/test split -> z-score (train statistics only)
    -> overlapping windows -> torch Dataset / DataLoader

Feature vector per time step (41 values, see `FEATURE_NAMES`):
    0       log mid-price return since the previous grid step
    1..10   log1p(ask distance from mid, in ticks) for levels 1..10
    11..20  log1p(bid distance from mid, in ticks) for levels 1..10
    21..30  log1p(ask volume) for levels 1..10
    31..40  log1p(bid volume) for levels 1..10

Prices are stored as distances from the mid-price, so the microstructure rules
become sign/ordering conditions on the features instead of being baked into the
representation: a generated book is crossed when ask_dist_1 + bid_dist_1 <= 0,
and a price ladder is invalid when the distances are not increasing with level.
This lets us measure whether a model learns the rules rather than enforcing them
by construction.

Usage: python dataset.py <data_dir>   (prints a summary and a round-trip check)
"""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PREFIX = "AMZN_2012-06-21_34200000_57600000_"
LEVELS = 10
PRICE_SCALE = 1e4   # LOBSTER stores prices in units of 1/10,000 dollar
TICK = 0.01         # minimum price increment in dollars
N_FEATURES = 1 + 4 * LEVELS
FEATURE_NAMES = (
    ["mid_return"]
    + [f"ask_dist_{i}" for i in range(1, LEVELS + 1)]
    + [f"bid_dist_{i}" for i in range(1, LEVELS + 1)]
    + [f"ask_vol_{i}" for i in range(1, LEVELS + 1)]
    + [f"bid_vol_{i}" for i in range(1, LEVELS + 1)]
)


def load_lob(data_dir):
    """Load event times (seconds after midnight) and the raw order book.

    The orderbook file has 4 columns per level: ask price, ask size,
    bid price, bid size. Row i is the book state after event i.
    """
    book = np.loadtxt(f"{data_dir}/{PREFIX}orderbook_10.csv", delimiter=",")
    times = np.loadtxt(f"{data_dir}/{PREFIX}message_10.csv", delimiter=",", usecols=0)
    return times, book


def resample_to_grid(times, book, step=1.0):
    """Sample the book on a fixed time grid using the last event at or before each time.

    The raw event stream is irregular and ~90% of events do not move the mid-price,
    so a fixed grid gives the model evenly spaced steps.
    """
    grid = np.arange(times[0], times[-1], step)
    idx = np.searchsorted(times, grid, side="right") - 1  # last event <= grid time
    return grid, book[idx]


def book_to_levels(book):
    """Split a (K, 40) book array into ask/bid prices (dollars) and volumes."""
    ask_p = book[:, 0::4] / PRICE_SCALE
    ask_v = book[:, 1::4]
    bid_p = book[:, 2::4] / PRICE_SCALE
    bid_v = book[:, 3::4]
    return ask_p, ask_v, bid_p, bid_v


def build_features(book):
    """Turn (K, 40) book snapshots into a (K, 41) raw feature array.

    Returns the features and the mid-price series (needed to rebuild prices).
    """
    ask_p, ask_v, bid_p, bid_v = book_to_levels(book)
    mid = (ask_p[:, 0] + bid_p[:, 0]) / 2
    ret = np.zeros(len(mid))
    ret[1:] = np.diff(np.log(mid))   # first step has no previous snapshot
    ask_dist = np.log1p((ask_p - mid[:, None]) / TICK)
    bid_dist = np.log1p((mid[:, None] - bid_p) / TICK)
    feats = np.concatenate(
        [ret[:, None], ask_dist, bid_dist, np.log1p(ask_v), np.log1p(bid_v)], axis=1
    )
    return feats, mid


def chronological_split(n_steps, fractions=(0.6, 0.2, 0.2)):
    """Contiguous (start, end) index ranges for train/val/test in time order.

    No shuffling across time: the model only ever trains on earlier data than it
    is validated and tested on, which avoids look-ahead bias.
    """
    assert abs(sum(fractions) - 1.0) < 1e-9
    a = int(n_steps * fractions[0])
    b = int(n_steps * (fractions[0] + fractions[1]))
    return {"train": (0, a), "val": (a, b), "test": (b, n_steps)}


def make_windows(feats, seq_len, stride=1):
    """Overlapping windows of shape (n_windows, seq_len, n_features).

    Windows are built inside one split only, so none straddles a split boundary.
    """
    starts = np.arange(0, len(feats) - seq_len + 1, stride)
    return np.stack([feats[s:s + seq_len] for s in starts]).astype(np.float32)


class LOBWindowDataset(Dataset):
    """Windows of normalised order book features, shape (seq_len, n_features)."""

    def __init__(self, windows):
        self.windows = torch.from_numpy(windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return self.windows[i]


def get_dataloaders(data_dir, seq_len=24, step=1.0, stride=1, batch_size=128,
                    fractions=(0.6, 0.2, 0.2)):
    """Build train/val/test DataLoaders and return them with preprocessing metadata.

    Normalisation statistics (mean/std per feature) come from the training split
    only, so no information from validation or test leaks into training.
    Returns (loaders, info) where info holds the stats and the split ranges.
    """
    times, book = load_lob(data_dir)
    grid, snaps = resample_to_grid(times, book, step)
    feats, mid = build_features(snaps)
    splits = chronological_split(len(feats), fractions)

    tr0, tr1 = splits["train"]
    mean = feats[tr0:tr1].mean(axis=0)
    std = feats[tr0:tr1].std(axis=0) + 1e-8   # epsilon guards against constant features
    norm = (feats - mean) / std

    loaders = {}
    for name, (s0, s1) in splits.items():
        windows = make_windows(norm[s0:s1], seq_len, stride)
        loaders[name] = DataLoader(
            LOBWindowDataset(windows),
            batch_size=batch_size,
            shuffle=(name == "train"),   # shuffling windows within train is fine
            drop_last=(name == "train"),
        )
    info = {"mean": mean, "std": std, "splits": splits, "grid": grid, "mid": mid,
            "feats_raw": feats, "seq_len": seq_len, "step": step}
    return loaders, info


def denormalise(windows, info):
    """Undo z-scoring: (…, n_features) normalised -> raw features."""
    return np.asarray(windows) * info["std"] + info["mean"]


def features_to_book(raw, start_mid):
    """Rebuild prices (dollars) and volumes from raw features of one window.

    raw: (seq_len, 41) raw features; start_mid: mid-price before the first step.
    Returns ask_p, ask_v, bid_p, bid_v, each (seq_len, 10). Invalid outputs
    (crossed books, non-monotonic ladders) are kept as-is so they can be audited.
    """
    mid = start_mid * np.exp(np.cumsum(raw[:, 0]))
    ask_dist = np.expm1(raw[:, 1:1 + LEVELS]) * TICK
    bid_dist = np.expm1(raw[:, 1 + LEVELS:1 + 2 * LEVELS]) * TICK
    ask_v = np.expm1(raw[:, 1 + 2 * LEVELS:1 + 3 * LEVELS])
    bid_v = np.expm1(raw[:, 1 + 3 * LEVELS:])
    return mid[:, None] + ask_dist, ask_v, mid[:, None] - bid_dist, bid_v


if __name__ == "__main__":
    loaders, info = get_dataloaders(sys.argv[1])
    for name, loader in loaders.items():
        s0, s1 = info["splits"][name]
        print(f"{name}: {len(loader.dataset)} windows, grid steps {s0}-{s1}, "
              f"batch shape {tuple(next(iter(loader)).shape)}")

    # Round-trip check: rebuilding a real window must recover the original book.
    times, book = load_lob(sys.argv[1])
    _, snaps = resample_to_grid(times, book, info["step"])
    ask_p, ask_v, bid_p, bid_v = book_to_levels(snaps)
    s = 1000
    raw = info["feats_raw"][s:s + 24]
    a_p, a_v, b_p, b_v = features_to_book(raw, info["mid"][s - 1])
    print("round-trip max abs error: ask price", np.abs(a_p - ask_p[s:s + 24]).max(),
          "bid price", np.abs(b_p - bid_p[s:s + 24]).max(),
          "ask vol", np.abs(a_v - ask_v[s:s + 24]).max())
    print("real windows crossed books:", int((a_p[:, 0] <= b_p[:, 0]).sum()))
    print("fraction of zero mid-returns on the grid:",
          float((info["feats_raw"][1:, 0] == 0).mean()))
