# shards_to_csv.py
from pathlib import Path
import numpy as np
import pandas as pd

def shards_to_csv(shards_dir, out_csv="hurricane_points.csv"):
    rows = []
    for p in sorted(Path(shards_dir).rglob("*.npz")):
        with np.load(p, allow_pickle=True, mmap_mode="r") as d:
            y   = d["y"]     # (N,2) [lat, lon]
            t   = d["t"]     # (N,) datetime64[ns] or similar
            sid = d["sid"]   # (N,) bytes/str
            N = len(y)
            for i in range(N):
                lat, lon = float(y[i,0]), float(y[i,1])
                s = sid[i].decode() if isinstance(sid[i], (bytes, np.bytes_)) else str(sid[i])
                ts = pd.to_datetime(t[i])
                rows.append((s, ts.isoformat(), int(pd.Timestamp(ts).value), lat, lon))
    df = pd.DataFrame(rows, columns=["sid","time_iso","time_ns","lat","lon"])
    df = df.dropna().sort_values(["sid","time_ns"]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df):,} rows")

if __name__ == "__main__":
    shards_to_csv(r"Cvclone\src\back_end\model\data_preprocess\shards", "hurricane_points.csv")
