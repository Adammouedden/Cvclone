# cuda_test.py
import time
import torch

def main():
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version (torch):", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Small correctness check
    a = torch.randn(3, 3, device=device)
    b = torch.randn(3, 3, device=device)
    c = a @ b
    print("c shape:", c.shape, "dtype:", c.dtype, "device:", c.device)

    # Quick performance check with a big matmul
    N = 4096  # increase if you want a heavier test
    x = torch.randn(N, N, device=device)
    y = torch.randn(N, N, device=device)

    # Warmup (important for fair timing on GPU)
    for _ in range(2):
        _ = x @ y
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    z = x @ y
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    print(f"Matmul {N}x{N} time: {t1 - t0:.3f}s")

    # Move a tensor back to CPU (verifies transfer path)
    z_cpu = z[:2, :2].to("cuda")
    print("z[0:2,0:2] on GPU:\n", z_cpu)

if __name__ == "__main__":
    main()
