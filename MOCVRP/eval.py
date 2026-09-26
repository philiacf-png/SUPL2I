import argparse, os, time
import numpy as np
import torch

from env import (BiCVRP, REF, CAP_TOL, make_weights, tcheb, dominates, hv_norm, pareto_filter_np,
                 build_ring, select_state, step_env, verify_rings, random_rings, load_rings)
from model import load_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(os.path.dirname(HERE), "testdata")
N_AUG = 8
BIG = 1.0e6


# instance augmentation: 8 distance-preserving transformations of the ring coordinates
def augment(xy):
    x, y = xy[..., :1], xy[..., 1:]
    d8 = [(x, y), (1 - x, y), (x, 1 - y), (1 - x, 1 - y),
          (y, x), (1 - y, x), (y, 1 - x), (1 - y, 1 - x)]
    return torch.cat([torch.cat([a, b], -1) for a, b in d8], 0)


# elite sharing: every preference picks the best feasible solution of the pool under its own scalarisation
def cross_assign(best_obj, best_g, weights, ref_t):
    g = tcheb(best_obj[:, None, :, :], weights[None, :, None, :], ref_t)
    return g.min(dim=2)


@torch.inference_mode()
def improve(model, prob, depot, nodes, demand, rings, weights, ref, T, aug=False, elite_sharing=True,
            sync_every=200, sync_start=200, temp_hi=1.5, temp_lo=1.0, chunk=4040,
            hv_steps=None, log_every=0):
    device = depot.device
    ne, N = nodes.shape[0], prob.size
    W = weights.shape[0]
    A = N_AUG if aug else 1
    assert chunk % W == 0, f"chunk must be a multiple of {W}"
    ref_t = torch.tensor(ref, device=device)
    xy_i, dem_i = build_ring(depot, nodes, demand, prob.K)
    xy_src = augment(xy_i) if aug else xy_i
    dem_src = dem_i.repeat(A, 1)
    G = A * ne                                      # (augmentation, instance) groups of W trajectories
    g_idx = torch.arange(G).repeat_interleave(W)
    w_idx = torch.arange(W).repeat(G)
    xy_all, dem_all = xy_src[g_idx], dem_src[g_idx]
    rec_all = BiCVRP.seq_to_rec(rings[g_idx % ne, w_idx].reshape(G * W, N).long().to(device))
    pref_all = weights[w_idx].to(device)
    T_all = G * W
    INF = float("inf")
    best_obj = torch.full((T_all, 2), BIG, device=device)
    best_g = torch.full((T_all,), INF, device=device)
    best_rec = torch.zeros(T_all, N, dtype=torch.long, device=device)
    arch_pts = [[] for _ in range(ne)]
    front_arch = np.zeros((G, W, 2))
    trace_steps = sorted({int(x) for x in (hv_steps or []) if 0 <= int(x) <= T})
    trace_idx = {t_: i for i, t_ in enumerate(trace_steps)}
    tr_obj = torch.zeros(len(trace_steps), T_all, 2)
    tr_g = torch.zeros(len(trace_steps), T_all)
    tr_arch_pts = [[[] for _ in trace_steps] for _ in range(ne)]
    if not elite_sharing:
        sync_every, temp_lo = 0, temp_hi
    n_chunks = (T_all + chunk - 1) // chunk
    t_start = time.time()

    def select_front(bo, bg):
        # end-of-run cross-assignment inside every group, then the best augmentation per preference
        bo, bg = bo.view(G, W, 2), bg.view(G, W)
        if elite_sharing:
            gmin, src = cross_assign(bo, bg, weights, ref_t)
            inj = gmin < bg
            bo = torch.where(inj[..., None], bo.gather(1, src[:, :, None].expand(G, W, 2)), bo)
            bg = torch.where(inj, gmin, bg)
        bo, bg = bo.view(A, ne, W, 2), bg.view(A, ne, W)
        a_best = bg.argmin(0)                                                # (ne, W)
        return bo.gather(0, a_best[None, :, :, None].expand(1, ne, W, 2)).squeeze(0).cpu().numpy()

    model.eval()
    for ci, c0 in enumerate(range(0, T_all, chunk)):
        c1 = min(c0 + chunk, T_all)
        b = c1 - c0
        nb = b // W
        xy, dem, rec, pref = xy_all[c0:c1], dem_all[c0:c1], rec_all[c0:c1], pref_all[c0:c1]
        last = None
        vis = torch.full((T + 1, b, 2), INF, device=device)
        st = prob.ring_state(xy, dem, rec, torch.ones_like(dem))
        feas = st["viol"] <= CAP_TOL
        vis[0] = torch.where(feas[:, None], st["obj_true"], vis[0])
        g = tcheb(st["obj"], pref, ref_t)
        best_g[c0:c1] = torch.where(feas, g, best_g[c0:c1])
        best_obj[c0:c1] = torch.where(feas[:, None], st["obj_true"], best_obj[c0:c1])
        best_rec[c0:c1] = torch.where(feas[:, None], rec, best_rec[c0:c1])
        if 0 in trace_idx:
            tr_obj[0, c0:c1], tr_g[0, c0:c1] = best_obj[c0:c1].cpu(), best_g[c0:c1].cpu()
        for t in range(T):
            temp = temp_hi + (temp_lo - temp_hi) * t / max(1, T - 1)
            _, last_new, rec_new, s_new = step_env(model, prob, xy, dem, rec, st, last, pref,
                                                  do_sample=True, samp_temp=temp)
            feas = s_new["viol"] <= CAP_TOL
            vis[t + 1] = torch.where(feas[:, None], s_new["obj_true"], vis[t + 1])
            g = tcheb(s_new["obj"], pref, ref_t)
            imp = feas & (g < best_g[c0:c1])
            best_g[c0:c1] = torch.where(imp, g, best_g[c0:c1])
            best_obj[c0:c1] = torch.where(imp[:, None], s_new["obj_true"], best_obj[c0:c1])
            best_rec[c0:c1] = torch.where(imp[:, None], rec_new, best_rec[c0:c1])
            revert = dominates(st["obj"], s_new["obj"])     # nondominance-based acceptance
            rec = torch.where(revert[:, None], rec, rec_new)
            st = select_state(revert, st, s_new)
            last = last_new
            if log_every and (t + 1) % log_every == 0:
                el = time.time() - t_start
                done = ci * T + t + 1
                print(f"[chunk {ci + 1}/{n_chunks}] step {t + 1}/{T} "
                      f"elapsed={el / 60:.1f}min eta={el / done * (n_chunks * T - done) / 60:.1f}min",
                      flush=True)
            if sync_every and t + 1 >= sync_start and (t + 1) % sync_every == 0 and t < T - 1:
                bo = best_obj[c0:c1].view(nb, W, 2)
                bg = best_g[c0:c1].view(nb, W)
                br = best_rec[c0:c1].view(nb, W, N)
                gmin, src = cross_assign(bo, bg, weights, ref_t)
                inj = gmin < bg
                best_obj[c0:c1] = torch.where(inj[..., None], bo.gather(1, src[:, :, None].expand(nb, W, 2)), bo).view(-1, 2)
                best_rec[c0:c1] = torch.where(inj[..., None], br.gather(1, src[:, :, None].expand(nb, W, N)), br).view(-1, N)
                best_g[c0:c1] = torch.where(inj, gmin, bg).view(-1)
                injf = inj.view(-1)
                if injf.any():
                    rec = torch.where(injf[:, None], best_rec[c0:c1], rec)
                    st = prob.ring_state(xy, dem, rec, st["age"])
                    last = last.clone()
                    last[injf] = 0
            ti = trace_idx.get(t + 1)
            if ti is not None:
                tr_obj[ti, c0:c1], tr_g[ti, c0:c1] = best_obj[c0:c1].cpu(), best_g[c0:c1].cpu()
        vis_np = vis.permute(1, 0, 2).cpu().numpy()
        for j in range(nb):
            gi = c0 // W + j
            pts = vis_np[j * W:(j + 1) * W].reshape(-1, 2)
            # archive: non-dominated set of every feasible visited point of the instance
            arch_pts[gi % ne].append(pareto_filter_np(pts[np.isfinite(pts).all(1)]))
            for ti, sp in enumerate(trace_steps):
                p = vis_np[j * W:(j + 1) * W, :int(sp) + 1].reshape(-1, 2)
                tr_arch_pts[gi % ne][ti].append(pareto_filter_np(p[np.isfinite(p).all(1)]))
            pts_t = vis[:, j * W:(j + 1) * W, :].reshape(-1, 2)
            pts_t = torch.where(torch.isfinite(pts_t), pts_t, torch.full_like(pts_t, BIG))
            gp = tcheb(pts_t[None, :, :], weights[:, None, :], ref_t)          # (W, |pts|)
            front_arch[gi] = pts_t[gp.argmin(dim=1)].cpu().numpy()
    ok = best_g < INF
    verify_rings(prob, xy_all, dem_all, best_rec, best_obj, ok, chunk)
    front_pts = select_front(best_obj, best_g)
    fa = front_arch.reshape(A, ne, W, 2)
    fag = tcheb(torch.from_numpy(fa).to(device), weights[None, None], ref_t).cpu().numpy()
    a_best = fag.argmin(0)                                                       # (ne, W)
    front_arch_pts = fa[a_best, np.arange(ne)[:, None], np.arange(W)[None, :]]
    arch = [pareto_filter_np(np.concatenate(arch_pts[i], 0)) for i in range(ne)]
    out = {"front_pts": front_pts, "front_arch_pts": front_arch_pts,
           "front_hv": np.array([hv_norm(front_pts[i], ref) for i in range(ne)]),
           "front_arch_hv": np.array([hv_norm(front_arch_pts[i], ref) for i in range(ne)]),
           "arch_hv": np.array([hv_norm(arch[i], ref) for i in range(ne)]),
           "arch_nds": np.array([len(arch[i]) for i in range(ne)], dtype=float),
           "feasible_frac": float(ok.float().mean().item()), "infer_time_s": time.time() - t_start}
    if trace_steps:
        out["trace_steps"] = np.array(trace_steps)
        out["trace_front_hv"] = np.zeros((len(trace_steps), ne))
        out["trace_arch_hv"] = np.zeros((len(trace_steps), ne))
        for ti in range(len(trace_steps)):
            fp = select_front(tr_obj[ti].to(device), tr_g[ti].to(device))
            for i in range(ne):
                out["trace_front_hv"][ti, i] = hv_norm(fp[i], ref)
                out["trace_arch_hv"][ti, i] = hv_norm(np.concatenate(tr_arch_pts[i][ti], 0), ref)
    return out


def parse_steps(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            rng, _, stp = part.partition(":")
            a, b = rng.split("-")
            out.extend(range(int(a), int(b) + 1, int(stp) if stp else 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=100, help="number of customers")
    ap.add_argument("--ne", type=int, default=200, help="number of test instances")
    ap.add_argument("--T", type=int, default=1000, help="improvement steps per trajectory")
    ap.add_argument("--n_sols", type=int, default=101, help="number of preference vectors")
    ap.add_argument("--elite_sharing", type=int, default=1, help="1 = elite sharing across trajectories")
    ap.add_argument("--sync_every", type=int, default=200, help="steps between elite-sharing rounds")
    ap.add_argument("--aug", type=int, default=0, help="1 = solve under 8 coordinate transformations")
    ap.add_argument("--chunk", type=int, default=0, help="trajectories per GPU batch (0 = 40 x n_sols)")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--test_data", default="", help="instance file; default ../testdata/test_bicvrp_n{n}.pt")
    ap.add_argument("--rings", default="", help="start-ring file; default: random rings (seed 2024)")
    ap.add_argument("--ref", default="", help="reference point 'a,b'; default from the size table")
    ap.add_argument("--hv_steps", default="", help="record HV at these steps, e.g. '0,10-100:10,200-5000:100'")
    ap.add_argument("--save", default="", help="write per-instance results to this .npz file")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = args.n
    ref = tuple(float(x) for x in args.ref.split(",")) if args.ref else REF[n]
    prob = BiCVRP(n)
    weights = make_weights(args.n_sols, device=device)
    tfile = args.test_data or os.path.join(TESTDATA, f"test_bicvrp_n{n}.pt")
    te = torch.load(tfile, map_location=device, weights_only=False)
    depot, nodes, demand = (te[k].float().to(device)[:args.ne] for k in ("depot", "nodes", "demand"))
    assert nodes.shape[0] >= args.ne, f"{tfile} holds {nodes.shape[0]} instances < --ne {args.ne}"
    rings = load_rings(args.rings)[:args.ne] if args.rings else random_rings(args.ne, n, prob.K, args.n_sols, args.seed)
    assert rings.shape[1] >= args.n_sols and rings.shape[2] == prob.size, "start-ring file does not match n_sols / n"
    rings = rings[:, :args.n_sols]
    model, ck = load_checkpoint(args.ckpt, device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    chunk = args.chunk or 40 * args.n_sols
    res = improve(model, prob, depot, nodes, demand, rings, weights, ref, args.T, aug=bool(args.aug),
                  elite_sharing=bool(args.elite_sharing), sync_every=args.sync_every,
                  chunk=chunk, hv_steps=parse_steps(args.hv_steps) if args.hv_steps else None,
                  log_every=args.log_every)
    tag = f"aug{N_AUG}" if args.aug else "no-aug"
    sh = f"sharing every {args.sync_every}" if args.elite_sharing else "no sharing"
    print(f"[eval] {os.path.basename(args.ckpt)} n={n} ne={args.ne} T={args.T} {tag} {sh} ref={ref} | "
          f"front HV {res['front_hv'].mean():.4f} | archive-extracted front HV {res['front_arch_hv'].mean():.4f} | "
          f"archive HV {res['arch_hv'].mean():.4f} | |ND| {res['arch_nds'].mean():.1f} | "
          f"feasible {res['feasible_frac']:.3f} | {res['infer_time_s']:.0f}s")
    if "trace_steps" in res:
        print("  step  front_HV  archive_HV")
        for i, s in enumerate(res["trace_steps"]):
            print(f"  {int(s):5d}  {res['trace_front_hv'][i].mean():.6f}  {res['trace_arch_hv'][i].mean():.6f}")
    if args.save:
        meta = dict(n=n, ne=args.ne, T=args.T, aug=int(args.aug), elite_sharing=int(args.elite_sharing),
                    sync_every=args.sync_every, seed=args.seed, ref=np.asarray(ref), ckpt=os.path.basename(args.ckpt))
        np.savez(args.save, **res, **meta)
        print(f"[eval] saved -> {args.save}")


if __name__ == "__main__":
    main()
