# resume_train.py
from __future__ import annotations
import inspect, sys, math
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# --- your project imports ---
from model import GRUSeq2Seq
from dataloader import get_dataloaders  # expects get_dataloaders(d, ...)

# Try to import a dataset builder if you have one
try:
    from dataset import hurricane_dataset as _maybe_build_dataset
except Exception:
    _maybe_build_dataset = None

# ----------------------------
# Config
# ----------------------------
class Cfg:
    ckpt_in      = "saved_gru_seq2seq_best.pt"
    out_dir      = "checkpoints_resumed"
    out_last     = "gru_seq2seq_last_from_resume.pt"
    out_best     = "gru_seq2seq_best_from_resume.pt"

    in_dim       = 3
    hid          = 64
    num_layers   = 1
    out_steps    = 8

    epochs       = 5
    lr           = 1e-3
    wd           = 0.0
    amp          = True
    grad_clip    = 1.0

    # dataloader/model data knobs
    TIN          = 6
    TOUT         = 8
    batch_size   = 64
    num_workers  = 4
    shuffle      = True
    pin_memory   = False

    # Option A: path to a serialized torch Dataset (torch.save)
    dataset_pickle: str | None = None
    # Option B: rely on dataset.hurricane_dataset(...) (we'll introspect its signature)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Utilities
# ----------------------------
def build_model(cfg: Cfg) -> nn.Module:
    return GRUSeq2Seq(
        in_dim=cfg.in_dim,
        hid=cfg.hid,
        num_layers=cfg.num_layers,
        out_steps=cfg.out_steps
    ).to(DEVICE)

def _call_with_sig(fn, alias_values: dict):
    sig = inspect.signature(fn)
    kwargs = {name: alias_values[name] for name in sig.parameters.keys() if name in alias_values}
    return fn(**kwargs)

def _build_dataset_from_cfg(cfg: Cfg):
    # Option A: load previously saved Dataset
    if cfg.dataset_pickle:
        p = Path(cfg.dataset_pickle)
        if not p.exists():
            raise FileNotFoundError(f"dataset_pickle not found: {p}")
        d = torch.load(p)
        return d

    # Option B: call dataset.hurricane_dataset(...) if available
    if _maybe_build_dataset is not None:
        alias = {
            # common names for in/out lengths
            "TIN": cfg.TIN, "tin": cfg.TIN, "input_len": cfg.TIN,
            "seq_in": cfg.TIN, "past_len": cfg.TIN, "context_len": cfg.TIN,
            "TOUT": cfg.TOUT, "tout": cfg.TOUT, "output_len": cfg.TOUT,
            "pred_len": cfg.TOUT, "horizon": cfg.TOUT, "forecast_h": cfg.TOUT,
            # sometimes builders accept batch-ish hints (ignored safely if absent)
            "batch_size": cfg.batch_size,
        }
        try:
            return _call_with_sig(_maybe_build_dataset, alias)
        except TypeError:
            # if the builder takes no args, just call it
            return _maybe_build_dataset()

    # If neither path nor builder exists, fail clearly
    raise RuntimeError(
        "No dataset provided. Set Cfg.dataset_pickle to a torch-saved Dataset, "
        "or define dataset.hurricane_dataset(...) so I can build it."
    )

def _call_get_dataloaders_from_cfg(cfg: Cfg):
    """Pass only the kwargs get_dataloaders actually accepts, including d if required."""
    d = _build_dataset_from_cfg(cfg)

    sig = inspect.signature(get_dataloaders)
    params = sig.parameters

    aliases = {
        # required dataset
        "d": d,

        # splits / batching / workers / pins
        "test_size": 0.2,
        "batch_size": cfg.batch_size, "batch": cfg.batch_size, "bs": cfg.batch_size,
        "seed": 42,
        "shuffle_train": cfg.shuffle, "shuffle": cfg.shuffle,
        "num_workers": cfg.num_workers, "workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
    }

    kwargs = {name: aliases[name] for name in params if name in aliases}

    try:
        loaders = get_dataloaders(**kwargs)
    except TypeError as e:
        # last resort: try positional (d) then fallback to no kwargs
        try:
            loaders = get_dataloaders(d)
        except Exception:
            raise

    # Normalize to (train, val)
    if isinstance(loaders, (list, tuple)) and len(loaders) >= 2:
        return loaders[0], loaders[1]
    if isinstance(loaders, dict):
        if "train" in loaders and "val" in loaders:
            return loaders["train"], loaders["val"]
        if "train_loader" in loaders and "val_loader" in loaders:
            return loaders["train_loader"], loaders["val_loader"]
    raise RuntimeError("get_dataloaders must return (train, val) or a dict with those.")

def try_load_checkpoint(model: nn.Module, optimizer=None, scheduler=None, scaler=None, path=""):
    ckpt = torch.load(path, map_location=DEVICE)
    start_epoch = 0
    best_val = math.inf

    if isinstance(ckpt, dict) and any(k in ckpt for k in ["model", "state_dict"]):
        state = ckpt.get("model", ckpt.get("state_dict"))
        model.load_state_dict(state, strict=False)

        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", best_val))
        print(f"Loaded full checkpoint from {path} (start_epoch={start_epoch}, best_val={best_val:.4f})")
    else:
        model.load_state_dict(ckpt, strict=False)
        print(f"Loaded raw state_dict from {path}")

    return start_epoch, best_val

@torch.no_grad()
def evaluate(model: nn.Module, loss_fn, val_loader):
    model.eval()
    total, n = 0.0, 0
    for xb, yb in val_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        bs = yb.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)

def train_one_epoch(model, loss_fn, opt, train_loader, scaler: GradScaler|None, grad_clip: float):
    model.train()
    total, n = 0.0, 0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast():
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        bs = yb.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_in", default=Cfg.ckpt_in)
    ap.add_argument("--out_dir", default=Cfg.out_dir)
    ap.add_argument("--out_last", default=Cfg.out_last)
    ap.add_argument("--out_best", default=Cfg.out_best)
    ap.add_argument("--epochs", type=int, default=Cfg.epochs)
    ap.add_argument("--lr", type=float, default=Cfg.lr)
    ap.add_argument("--wd", type=float, default=Cfg.wd)
    ap.add_argument("--no_amp", action="store_true", help="disable AMP")
    ap.add_argument("--dataset_pickle", default=Cfg.dataset_pickle, help="optional: torch-saved Dataset to load")
    args = ap.parse_args()

    # allow CLI override of dataset_pickle
    Cfg.dataset_pickle = args.dataset_pickle if args.dataset_pickle not in ("None", "null", "") else None

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_last = out_dir / args.out_last
    out_best = out_dir / args.out_best

    # Data (now passes d properly)
    train_loader, val_loader = _call_get_dataloaders_from_cfg(Cfg)

    # Model / optim
    model = build_model(Cfg)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    scaler = GradScaler(enabled=(not args.no_amp) and Cfg.amp and (DEVICE == "cuda"))

    # Load previous weights / states
    start_epoch, best_val = try_load_checkpoint(model, optimizer, scheduler, scaler, args.ckpt_in)

    # Train
    for epoch in range(start_epoch, start_epoch + args.epochs):
        tr = train_one_epoch(model, loss_fn, optimizer, train_loader, scaler, Cfg.grad_clip)
        va = evaluate(model, loss_fn, val_loader)
        scheduler.step(va)
        print(f"epoch {epoch+1:03d} | train {tr:.4f} | val {va:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}")

        torch.save(model.state_dict(), out_last)

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), out_best)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch + 1,
                "best_val": best_val,
                "cfg": vars(Cfg),
            }, out_dir / "full_best_from_resume.ckpt")
            print(f"  ↳ new best! saved to {out_best}")

    print(f"\nSaved LAST weights to: {out_last}")
    print(f"Saved BEST-from-resume weights to: {out_best}")

if __name__ == "__main__":
    main()
