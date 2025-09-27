from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse

SHARDS_DIR = Path(r"src\back_end\model\data_preprocess\shards")
FIRST_SHARD = Path(r"src\back_end\model\data_preprocess\shards\shard_000.npz")

BT_MIN, BT_MAX = 180.0, 320.0  # Kelvin, only needed if you want BT back

def x01_to_uint8(x01: np.ndarray) -> np.ndarray:
    """[0,1] float -> uint8 0..255"""
    x = np.clip(x01, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def x01_to_kelvin(x01: np.ndarray) -> np.ndarray:
    """[0,1] float -> Kelvin"""
    return x01 * (BT_MAX - BT_MIN) + BT_MIN

def reconstruct_image():
    with np.load(FIRST_SHARD, allow_pickle=True, mmap_mode="r") as data:
        X, y, t, sid = data["X"], data["y"], data["t"], data["sid"]

        i = 0  # pick an index
        img01 = X[i, 0]                # (256, 256) float32 in [0,1]
        img8  = x01_to_uint8(img01)    # uint8 0..255

        plt.imshow(img8, cmap="gray", origin="lower")
        plt.title(f"{sid[i]} @ {t[i]}  lat={y[i,0]:.2f}, lon={y[i,1]:.2f}")
        plt.axis("off")
        plt.show()

    return img8

def send_image_test():
    IMAGE_PATH = rf"C:\Users\adamm\Documents\HACKATHONS\Cvclone\src\back_end\model\data_preprocess\shards\preview_shard_000.png"

    return FileResponse(IMAGE_PATH, media_type="image/png")

def display_shard():
    npz_files = sorted(SHARDS_DIR.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {SHARDS_DIR.resolve()}")
        return

    for f in npz_files:
        print(f"\nLoading: {f}")
        with np.load(f, allow_pickle=True) as data:
            print("Keys:", list(data.keys()))
            X = data["X"]       # (N, 1, H, W)
            y = data["y"]       # (N, 2)
            t = data["t"]       # (N,)
            SID = data["sid"]   # (N,)
            FNAME = data["fname"]  # (N,)

            print(f"X: {X.shape} {X.dtype}")
            print(f"y: {y.shape} {y.dtype}")
            print(f"t: {t.shape} {t.dtype}")
            print(f"sid: {SID.shape} {SID.dtype}")
            print(f"fname: {FNAME.shape} {FNAME.dtype}")

if __name__ == "__main__":
    #display_shard()
    reconstruct_image()

