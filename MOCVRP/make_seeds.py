import argparse
import torch
from env import BiCVRP, random_rings

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ne", type=int, default=200)
    ap.add_argument("--n_sols", type=int, default=101)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prob = BiCVRP(a.n)
    rings = random_rings(a.ne, a.n, prob.K, a.n_sols, a.seed)
    torch.save({"tours": rings, "n": a.n, "K": prob.K, "n_instances": a.ne, "seed": a.seed}, a.out)
    print(f"saved {tuple(rings.shape)} -> {a.out}")
