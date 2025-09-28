# train_gru_csv.py
import math, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model import GRUSeq2Seq  # the fixed version from earlier

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
TIN      = 6
TOUT     = 8
STRIDE   = 4
BATCH    = 64
EPOCHS   = 50
LR       = 1e-2
DT_NORM  = 3.0

def unwrap_lon(arr):  # degrees -> degrees unwrapped
    return np.rad2deg(np.unwrap(np.deg2rad(arr.astype(np.float64)))).astype(np.float32)

def to_hours(ns):     # int ns -> float hours
    return (ns.astype(np.float64) / 3_600_000_000_000.0).astype(np.float32)

def equirect_km(lat0, lon0, lat, lon):
    R = 6371.0088
    lat0r = math.radians(float(lat0))
    x = (np.radians(lon) - math.radians(float(lon0))) * math.cos(lat0r) * R
    y = (np.radians(lat) - math.radians(float(lat0))) * R
    return x.astype(np.float32), y.astype(np.float32)

class CSVWindowedDataset(Dataset):
    def __init__(self, csv_path, tin=TIN, tout=TOUT, stride=STRIDE,
                 sid_whitelist=None, max_windows=None):
        self.tin, self.tout, self.stride = tin, tout, stride
        df = pd.read_csv(csv_path)
        if sid_whitelist is not None:
            df = df[df["sid"].isin(set(sid_whitelist))]
        df = df.dropna(subset=["sid","time_ns","lat","lon"])
        df = df.sort_values(["sid","time_ns"]).reset_index(drop=True)

        self.windows = []
        self.buf = []  # keep (lat, lon_unwrap, time_ns) per row
        for sid, g in df.groupby("sid", sort=False):
            lat = g["lat"].to_numpy(np.float32)
            lon = unwrap_lon(g["lon"].to_numpy(np.float32))
            ts  = g["time_ns"].to_numpy(np.int64)
            n = len(g); need = tin + tout
            base_offset = len(self.buf)
            for i in range(n):
                self.buf.append((lat[i], lon[i], ts[i], sid))
            for start in range(0, n - need + 1, stride):
                idxs = list(range(base_offset + start, base_offset + start + need))
                self.windows.append(idxs)

        if max_windows is not None and len(self.windows) > int(max_windows):
            self.windows = self.windows[:int(max_windows)]
        if not self.windows:
            raise RuntimeError("No windows found.")

    def __len__(self): return len(self.windows)

    def __getitem__(self, k):
        idxs = self.windows[k]; tin, tout = self.tin, self.tout
        arr = np.array([self.buf[i] for i in idxs], dtype=object)  # [(lat,lon,ts,sid),...]
        lat = arr[:,0].astype(np.float32)
        lon = arr[:,1].astype(np.float32)
        ts  = arr[:,2].astype(np.int64)

        th  = to_hours(ts)
        dth = np.diff(th, prepend=th[0])
        dth[0]  = max(dth[0], DT_NORM)
        dth[1:] = np.clip(dth[1:], 1e-3, 24.0)

        lat0, lon0 = float(lat[0]), float(lon[0])
        x, y = equirect_km(lat0, lon0, lat, lon)

        XY_scale = 1000.0
        x /= XY_scale
        y /= XY_scale


        x_in    = np.stack([x[:tin], y[:tin], (dth[:tin] / DT_NORM)], axis=-1).astype(np.float32)
        y_out   = np.stack([x[tin:], y[tin:]], axis=-1).astype(np.float32)
        dts_out = (dth[tin:] / DT_NORM).reshape(-1,1).astype(np.float32)

        return (torch.from_numpy(x_in),
                torch.from_numpy(dts_out),
                torch.from_numpy(y_out))

def get_loaders(ds, batch=BATCH, test_size=0.2, seed=42, workers=4):
    idx = np.arange(len(ds))
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
    tr = torch.utils.data.Subset(ds, tr_idx); te = torch.utils.data.Subset(ds, te_idx)
    tr_loader = DataLoader(tr, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=(DEVICE=="cuda"))
    te_loader = DataLoader(te, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=(DEVICE=="cuda"))
    return tr_loader, te_loader

def train_one_epoch(model, loader, opt, scaler):
    model.train(); mse = nn.MSELoss(); total=n=0
    for x_in, dts_out, teacher in loader:
        x_in, dts_out, teacher = x_in.to(DEVICE), dts_out.to(DEVICE), teacher.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
            pred = model(x_in, dts_out, teacher=teacher)
            loss = mse(pred, teacher)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        total += loss.item()*x_in.size(0); n += x_in.size(0)
    return total/max(n,1)

@torch.no_grad()
def eval_one_epoch(model, loader):
    model.eval(); mse = nn.MSELoss(reduction="sum"); total=n=0
    for x_in, dts_out, teacher in loader:
        x_in, dts_out, teacher = x_in.to(DEVICE), dts_out.to(DEVICE), teacher.to(DEVICE)
        pred = model(x_in, dts_out, teacher=None)
        total += mse(pred, teacher).item(); n += x_in.size(0)
    return total/max(n,1)

def main():
    ds = CSVWindowedDataset("hurricane_points.csv", tin=TIN, tout=TOUT, stride=STRIDE, max_windows=100_000)
    tr, te = get_loaders(ds)
    model = GRUSeq2Seq(in_dim=3, hid=64, num_layers=1, out_steps=TOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    best=float("inf")
    for e in range(1, EPOCHS+1):
        tr_loss = train_one_epoch(model, tr, opt, scaler)
        te_loss = eval_one_epoch(model, te)
        if te_loss < best:
            best = te_loss
            torch.save(model.state_dict(),"gru_seq2seq_best.pt")
        print(f"epoch {e:02d} | train {tr_loss:.4f} | val {te_loss:.4f} {'*' if te_loss==best else ''}")

if __name__ == "__main__":
    main()
