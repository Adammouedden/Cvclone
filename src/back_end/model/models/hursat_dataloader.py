# hurricane_seq_dataloader.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# ----------------------------
# Dataset (sequence-level)
# ----------------------------
class HurricaneSeqDataset(Dataset):
    """
    Each sample is a fixed-length window from a single storm (SID), time-sorted.
    Returns:
      X_seq: torch.float32 [T,1,H,W] (values in [0,1])
      Y_seq: torch.float32 [T,2]     ([lat, lon] per frame)
      meta : dict { 'sid': str, 'timestamps_ns': list[int], 'len': int }
    Notes:
      - Opens NPZ files ONLY inside __getitem__ (safe for Windows spawn).
      - Precomputes windows using lightweight metadata, then closes files.
    """
    def __init__(self, shards_dir: Path | str, past_len: int = 6, stride: int = 1):
        super().__init__()
        self.shards_dir = Path(shards_dir)
        self.past_len = int(past_len)
        self.stride = int(stride)

        self._files: List[Path] = sorted(self.shards_dir.glob("*.npz"))
        if not self._files:
            raise FileNotFoundError(f"No .npz found in {self.shards_dir.resolve()}")

        # Build windows per SID without keeping files open
        by_sid: Dict[str, List[Tuple[int, int, np.datetime64]]] = defaultdict(list)
        for fi, f in enumerate(self._files):
            # Read minimal metadata then close (no mmap here)
            with np.load(f, allow_pickle=True) as d:
                t_arr = d["t"]
                sid_arr = d["sid"]
                n = len(t_arr)
                for i in range(n):
                    by_sid[str(sid_arr[i])].append((fi, i, t_arr[i]))

        self._windows: List[List[Tuple[int, int]]] = []
        self.window_sids: List[str] = []
        for sid_key, rows in by_sid.items():
            rows.sort(key=lambda r: r[2])                    # sort by time
            idxs = [(fi, i) for fi, i, _ in rows]
            for s in range(0, len(idxs) - self.past_len + 1, self.stride):
                self._windows.append(idxs[s:s + self.past_len])
                self.window_sids.append(sid_key)

        # Per-process cache of open npz handles (created after worker spawn)
        self._cache: Optional[Dict[int, Any]] = None

    def __len__(self) -> int:
        return len(self._windows)

    # Exclude open-file cache from pickling (Windows spawn safety)
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache = None

    def _get_npz(self, fi: int):
        if self._cache is None:
            self._cache = {}
        if fi not in self._cache:
            # Open lazily inside the worker; keep memory low with mmap
            self._cache[fi] = np.load(self._files[fi], allow_pickle=True, mmap_mode="r")
        return self._cache[fi]

    def __getitem__(self, widx: int):
        window = self._windows[widx]  # list[(fi, i)]
        xs, ys, ts_ns = [], [], []
        for fi, i in window:
            d = self._get_npz(fi)
            x01 = d["X"][i, 0]                 # (H,W) float32 in [0,1]
            y   = d["y"][i].astype(np.float32) # (2,) lat,lon
            t   = d["t"][i]                    # datetime64[ns]
            xs.append(torch.from_numpy(x01).unsqueeze(0).float())   # [1,H,W]
            ys.append(torch.from_numpy(y).float())                  # [2]
            ts_ns.append(int(np.datetime64(t).astype("datetime64[ns]").astype(np.int64)))

        X_seq = torch.stack(xs, dim=0)  # [T,1,H,W]
        Y_seq = torch.stack(ys, dim=0)  # [T,2]
        meta = {"sid": self.window_sids[widx], "timestamps_ns": ts_ns, "len": self.past_len}
        return X_seq, Y_seq, meta

# ----------------------------
# Collate
# ----------------------------
def seq_collate(batch):
    """
    Collate sequences of equal T.
    Returns:
      X: [B,T,1,H,W], Y: [B,T,2], meta: list[dict]
    """
    Xs, Ys, metas = zip(*batch)
    return torch.stack(Xs, dim=0), torch.stack(Ys, dim=0), list(metas)

# ----------------------------
# Split by SID (no leakage)
# ----------------------------
def _split_indices_by_sid(sids: List[str], train_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    uniq = sorted(set(sids))
    rng.shuffle(uniq)
    n_train = int(len(uniq) * train_frac)
    train_sids = set(uniq[:n_train])
    idx_train, idx_val = [], []
    for idx, sid in enumerate(sids):
        (idx_train if sid in train_sids else idx_val).append(idx)
    return idx_train, idx_val

# ----------------------------
# Public factory
# ----------------------------
def make_loaders(
    shards_dir: str | Path,
    past_len: int = 6,
    stride: int = 1,
    batch_size: int = 8,
    train_frac: float = 0.8,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Returns:
      {
        'train': DataLoader -> (X:[B,T,1,H,W], Y:[B,T,2], meta:list[dict]),
        'val'  : DataLoader -> ...
      }
    """
    ds = HurricaneSeqDataset(shards_dir, past_len=past_len, stride=stride)

    # Use dataset’s precomputed SIDs for each window (no file opens here)
    sids = ds.window_sids
    idx_train, idx_val = _split_indices_by_sid(sids, train_frac, seed)

    train_ds = Subset(ds, idx_train)
    val_ds   = Subset(ds, idx_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=seq_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=seq_collate,
        drop_last=False,
    )
    return {"train": train_loader, "val": val_loader}

# ----------------------------
# Quick smoke test (optional)
# ----------------------------
if __name__ == "__main__":
    # On Windows, first verify with single-process loading:
    loaders = make_loaders(
        shards_dir=r"src\back_end\model\data_preprocess\shards",
        past_len=6, stride=1, batch_size=4,
        train_frac=0.8, seed=42,
        num_workers=0,   # <= set to 0 for initial sanity
        pin_memory=True
    )
    X, Y, meta = next(iter(loaders["train"]))
    print("X", X.shape, X.dtype)
    print("Y", Y.shape, Y.dtype)
    print("meta[0]", meta[0])
