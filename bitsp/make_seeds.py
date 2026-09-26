import argparse
import torch
from env import random_tours

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ne", type=int, default=200)
    ap.add_argument("--n_sols", type=int, default=101)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tours = random_tours(a.ne, a.n, a.n_sols, a.seed)
    torch.save({"tours": tours, "n": a.n, "n_instances": a.ne, "seed": a.seed}, a.out)
    print(f"saved {tuple(tours.shape)} -> {a.out}")
