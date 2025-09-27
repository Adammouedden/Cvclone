# dataset.py  (replace your HurricaneSeqDataset with this safer version)
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset

class HurricaneSeqDataset(Dataset):
    """
    Returns:
      X_seq: [T,1,H,W], Y_seq: [T,2], meta: {'sid': str, 'timestamps_ns': list[int]}
    """
    def __init__(self, shards_dir: Path, past_len: int = 6, stride: int = 1):
        super().__init__()
        self.shards_dir = Path(shards_dir)
        self.past_len = past_len
        self.stride = stride

        self._files: List[Path] = sorted(self.shards_dir.glob("*.npz"))
        if not self._files:
            raise FileNotFoundError(f"No .npz found in {self.shards_dir.resolve()}")

        # Load only lightweight metadata into memory, then CLOSE file.
        by_sid: Dict[str, List[Tuple[int,int,np.datetime64]]] = defaultdict(list)
        self._lengths: List[int] = []
        for fi, f in enumerate(self._files):
            with np.load(f, allow_pickle=True) as d:   # no mmap; close immediately
                n = len(d["t"])
                self._lengths.append(n)
                t_arr = d["t"]
                sid_arr = d["sid"]
                for i in range(n):
                    by_sid[str(sid_arr[i])].append((fi, i, t_arr[i]))

        # Build fixed-length windows (fi,i) without opening files.
        self._windows: List[List[Tuple[int,int]]] = []
        self.window_sids: List[str] = []  # <- used by loader split
        for sid_key, rows in by_sid.items():
            rows.sort(key=lambda r: r[2])  # sort by time
            idxs = [(fi, i) for fi, i, _ in rows]
            for s in range(0, len(idxs) - past_len + 1, stride):
                self._windows.append(idxs[s:s+past_len])
                self.window_sids.append(sid_key)

        # Lazily-created, per-process cache of open NPZ files.
        self._cache: Optional[Dict[int, Any]] = None

    def __len__(self):
        return len(self._windows)

    # Ensure caches are not pickled on Windows spawn.
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
            # Open inside worker process; safe to cache here
            self._cache[fi] = np.load(self._files[fi], allow_pickle=True, mmap_mode="r")
        return self._cache[fi]

    def __getitem__(self, widx: int):
        window = self._windows[widx]
        xs, ys, ts = [], [], []
        for fi, i in window:
            d = self._get_npz(fi)
            x01 = d["X"][i, 0]                 # (H,W) float32
            y   = d["y"][i].astype(np.float32) # (2,)
            t   = d["t"][i]
            xs.append(torch.from_numpy(x01).unsqueeze(0).float())  # [1,H,W]
            ys.append(torch.from_numpy(y).float())                  # [2]
            ts.append(int(np.datetime64(t).astype("datetime64[ns]").astype(np.int64)))
        X_seq = torch.stack(xs, dim=0)  # [T,1,H,W]
        Y_seq = torch.stack(ys, dim=0)  # [T,2]
        meta = {"sid": self.window_sids[widx], "timestamps_ns": ts, "len": self.past_len}
        return X_seq, Y_seq, meta
