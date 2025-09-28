import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class hurricane_dataset(Dataset):
    def __init__(self, shards_dir, max_samples=None):
        """
        shards_dir: folder containing many .npz files.
        Expects keys per shard: y [N,2], t [N], sid [N]
        (X may exist in the shard, but we never read it.)
        """
        self.shards_dir = Path(shards_dir)
        self.files = sorted(self.shards_dir.rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz found under {self.shards_dir}")

        # Build a flat index: (file_idx, sample_idx_in_file)
        self.index = []
        self.lengths = []
        for fi, p in enumerate(self.files):
            with np.load(p, allow_pickle=True, mmap_mode="r") as data:
                n = len(data["y"])  # all arrays share length N
            self.lengths.append(n)
            self.index.extend([(fi, i) for i in range(n)])
        
        if max_samples is not None:
            self.index = self.index[:int(max_samples)]
        

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, i = self.index[idx]
        p = self.files[fi]
        with np.load(p, allow_pickle=True, mmap_mode="r") as data:
            # Only read what we need:
            y   = data["y"][i]    # [2] (lat, lon)
            t   = data["t"][i]    # datetime64[ns] or similar
            sid = data["sid"][i]  # bytes or str
                


        # Decode SID if bytes
        if isinstance(sid, (bytes, np.bytes_)):
            sid = sid.decode()

        # Convert outputs to tensors (time as int ns)
        y = torch.as_tensor(y, dtype=torch.float32)  # [2]
        if np.issubdtype(type(t), np.datetime64) or np.issubdtype(np.array(t).dtype, np.datetime64):
            t_ns = np.int64(np.datetime64(t).astype("datetime64[ns]").astype(np.int64))
        else:
            t_ns = np.int64(t)

        return y, torch.as_tensor(t_ns, dtype=torch.int64), sid

# Minimal usage
if __name__ == "__main__":
    ds = hurricane_dataset(r"Cvclone\src\back_end\model\data_preprocess\shards")
    print("samples:", len(ds))
    y, t_ns, sid = ds[0]
    print(y.shape, t_ns, sid)
