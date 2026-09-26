import argparse, os, time
import numpy as np
import torch

from env import (TriTSP, REF, make_weights, tcheb, dominates, hv_norm, pareto_filter_np,
                 nd_filter, do_move, age_tick, random_tours, load_tours)
from model import load_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(os.path.dirname(HERE), "testdata")
N_AUG = 8


# instance augmentation: 8 distance-preserving transformations of the first objective's coordinates
def augment(inst):
    x, y, rest = inst[..., [0]], inst[..., [1]], inst[..., 2:]
    d8 = [(x, y), (1 - x, y), (x, 1 - y), (1 - x, 1 - y),
          (y, x), (1 - y, x), (y, 1 - x), (1 - y, 1 - x)]
    return torch.cat([torch.cat([a, b, rest], -1) for a, b in d8], 0)


# elite sharing: every preference picks the best solution of the pool under its own scalarisation
def cross_assign(best_obj, best_g, weights, ref_t):
    g = tcheb(best_obj[:, None, :, :], weights[None, :, None, :], ref_t)
    return g.min(dim=2)


@torch.inference_mode()
def improve(model, prob, instances, tours, weights, ref, T, aug=False, elite_sharing=True,
            sync_every=200, sync_start=200, temp_hi=1.5, temp_lo=1.0, chunk=4200,
            hv_steps=None, log_every=0):
    device = instances.device
    ne, n = instances.shape[0], prob.size
    W, K = weights.shape
    A = N_AUG if aug else 1
    P = A * W                                       # trajectories per instance
    assert chunk % P == 0, f"chunk must be a multiple of {P}"
    ref_t = torch.tensor(ref, device=device)
    coords_src = augment(instances) if aug else instances
    k_idx = torch.arange(ne).repeat_interleave(P)
    a_idx = torch.arange(A).repeat_interleave(W).repeat(ne)
    w_idx = torch.arange(W).repeat(ne * A)
    coords_all = coords_src[a_idx * ne + k_idx]
    rec_all = TriTSP.seq_to_rec(tours[k_idx, w_idx].reshape(-1, n).to(device))
    pref_all = weights[w_idx].to(device)
    T_all = ne * P
    best_obj = torch.zeros(T_all, K, device=device)
    best_g = torch.full((T_all,), float("inf"), device=device)
    arch_hv = np.zeros(ne)
    arch_nds = np.zeros(ne)
    front_arch = np.zeros((ne, W, K))
    trace_steps = sorted({int(x) for x in (hv_steps or []) if 0 <= int(x) <= T})
    trace_idx = {t_: i for i, t_ in enumerate(trace_steps)}
    tr_front = np.zeros((len(trace_steps), ne))
    tr_arch = np.zeros((len(trace_steps), ne))
    if not elite_sharing:
        sync_every, temp_lo = 0, temp_hi
    n_chunks = (T_all + chunk - 1) // chunk
    t_start = time.time()

    def record(ti, c0, nb):
        bo = best_obj[c0:c0 + nb * P].view(nb, P, K)
        bg = best_g[c0:c0 + nb * P].view(nb, P)
        sel = select_front(bo, bg)
        for j in range(nb):
            tr_front[ti, c0 // P + j] = hv_norm(sel[j], ref)

    def select_front(bo, bg):
        nb = bo.shape[0]
        if aug:
            bg_a = bg.view(nb, A, W)
            best_a = bg_a.argmin(1)                                          # (nb, W)
            idx = (best_a * W + torch.arange(W, device=bo.device)[None, :])
            return bo.gather(1, idx[:, :, None].expand(nb, W, K)).cpu().numpy()
        if elite_sharing:
            gmin, src = cross_assign(bo, bg, weights, ref_t)
            inj = gmin < bg
            bo = torch.where(inj[..., None], bo.gather(1, src[:, :, None].expand(nb, W, K)), bo)
        return bo.cpu().numpy()

    model.eval()
    for ci, c0 in enumerate(range(0, T_all, chunk)):
        c1 = min(c0 + chunk, T_all)
        b = c1 - c0
        nb = b // P
        coords, rec, pref = coords_all[c0:c1], rec_all[c0:c1], pref_all[c0:c1]
        last = None
        age = torch.ones(b, n, device=device)
        vis = torch.empty(T + 1, b, K, device=device)
        obj = prob.get_costs(coords, rec)
        vis[0] = obj
        g = tcheb(obj, pref, ref_t)
        best_g[c0:c1] = g
        best_obj[c0:c1] = obj
        best_rec = rec.clone()
        o_rec = obj
        if 0 in trace_idx:
            record(trace_idx[0], c0, nb)
        for t in range(T):
            temp = temp_hi + (temp_lo - temp_hi) * t / max(1, T - 1)
            pair, _ = model(coords, rec, last, pref, age, do_sample=True, samp_temp=temp)
            rec_new = do_move(prob, rec, pair)
            obj = prob.get_costs(coords, rec_new)
            vis[t + 1] = obj
            g = tcheb(obj, pref, ref_t)
            revert = dominates(o_rec, obj)                   # nondominance-based acceptance
            rec = torch.where(revert[:, None], rec, rec_new)
            o_rec = torch.where(revert[:, None], o_rec, obj)
            last = pair
            age = torch.where(revert[:, None], age, age_tick(age, pair))
            imp = g < best_g[c0:c1]
            best_g[c0:c1] = torch.where(imp, g, best_g[c0:c1])
            best_obj[c0:c1] = torch.where(imp[:, None], obj, best_obj[c0:c1])
            best_rec = torch.where(imp[:, None], rec_new, best_rec)
            if log_every and (t + 1) % log_every == 0:
                el = time.time() - t_start
                done = ci * T + t + 1
                print(f"[chunk {ci + 1}/{n_chunks}] step {t + 1}/{T} "
                      f"elapsed={el / 60:.1f}min eta={el / done * (n_chunks * T - done) / 60:.1f}min",
                      flush=True)
            if sync_every and t + 1 >= sync_start and (t + 1) % sync_every == 0 and t < T - 1:
                bo = best_obj[c0:c1].view(nb, P, K)
                bg = best_g[c0:c1].view(nb, P)
                br = best_rec.view(nb, P, n)
                gmin_w, src_w = cross_assign(bo, bg, weights, ref_t)         # (nb, W)
                gmin = gmin_w[:, None, :].expand(nb, A, W).reshape(nb, P)
                src = src_w[:, None, :].expand(nb, A, W).reshape(nb, P)
                inj = gmin < bg
                bo_new = torch.where(inj[..., None], bo.gather(1, src[..., None].expand(nb, P, K)), bo)
                br_new = torch.where(inj[..., None], br.gather(1, src[..., None].expand(nb, P, n)), br)
                best_g[c0:c1] = torch.where(inj, gmin, bg).view(-1)
                best_obj[c0:c1] = bo_new.view(-1, K)
                best_rec = br_new.view(-1, n)
                injf = inj.view(-1)
                rec = torch.where(injf[:, None], best_rec, rec)
                o_rec = torch.where(injf[:, None], best_obj[c0:c1], o_rec)
                last = last.clone()
                last[injf] = 0
            ti = trace_idx.get(t + 1)
            if ti is not None:
                record(ti, c0, nb)
        # archive: non-dominated set of every visited point of the instance
        for j in range(nb):
            pts = vis[:, j * P:(j + 1) * P, :].reshape(-1, K)
            if trace_steps:
                pts_np = vis[:, j * P:(j + 1) * P, :].permute(1, 0, 2).cpu().numpy()
                run, prev = None, 0
                for ti, sp in enumerate(trace_steps):
                    new_pts = pts_np[:, prev:int(sp) + 1, :].reshape(-1, K)
                    run = new_pts if run is None else np.concatenate([run, new_pts], 0)
                    run = pareto_filter_np(run)
                    tr_arch[ti, c0 // P + j] = hv_norm(run, ref)
                    prev = int(sp) + 1
            nd = nd_filter(pts).cpu().numpy()
            arch_hv[c0 // P + j] = hv_norm(nd, ref)
            arch_nds[c0 // P + j] = len(nd)
            gp = tcheb(pts[None, :, :], weights[:, None, :], ref_t)          # (W, |pts|)
            front_arch[c0 // P + j] = pts[gp.argmin(dim=1)].cpu().numpy()
    front_pts = select_front(best_obj.view(ne, P, K), best_g.view(ne, P))
    out = {"front_pts": front_pts, "front_arch_pts": front_arch,
           "front_hv": np.array([hv_norm(front_pts[i], ref) for i in range(ne)]),
           "front_arch_hv": np.array([hv_norm(front_arch[i], ref) for i in range(ne)]),
           "arch_hv": arch_hv, "arch_nds": arch_nds, "infer_time_s": time.time() - t_start}
    if trace_steps:
        out["trace_steps"] = np.array(trace_steps)
        out["trace_front_hv"] = tr_front
        out["trace_arch_hv"] = tr_arch
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
    ap.add_argument("--n", type=int, default=100, help="problem size")
    ap.add_argument("--ne", type=int, default=200, help="number of test instances")
    ap.add_argument("--T", type=int, default=1000, help="improvement steps per trajectory")
    ap.add_argument("--n_sols", type=int, default=105, help="number of preference vectors")
    ap.add_argument("--elite_sharing", type=int, default=1, help="1 = elite sharing across trajectories")
    ap.add_argument("--sync_every", type=int, default=200, help="steps between elite-sharing rounds")
    ap.add_argument("--aug", type=int, default=0, help="1 = solve under 8 coordinate transformations")
    ap.add_argument("--chunk", type=int, default=0,
                    help="trajectories per GPU batch (0 = 40 instances without, 5 with augmentation)")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--test_data", default="", help="instance file; default ../testdata/test_tritsp_n{n}.pt")
    ap.add_argument("--tours", default="", help="start-tour file; default: random tours (seed 2024)")
    ap.add_argument("--ref", default="", help="reference point 'a,b,c'; default from the size table")
    ap.add_argument("--hv_steps", default="", help="record HV at these steps, e.g. '0,10-100:10,200-5000:100'")
    ap.add_argument("--save", default="", help="write per-instance results to this .npz file")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = args.n
    ref = tuple(float(x) for x in args.ref.split(",")) if args.ref else REF[n]
    prob = TriTSP(n)
    weights = make_weights(args.n_sols, device=device)
    tfile = args.test_data or os.path.join(TESTDATA, f"test_tritsp_n{n}.pt")
    inst = torch.load(tfile, map_location=device, weights_only=False).float()[:args.ne]
    assert inst.shape[0] >= args.ne, f"{tfile} holds {inst.shape[0]} instances < --ne {args.ne}"
    tours = load_tours(args.tours)[:args.ne] if args.tours else random_tours(args.ne, n, args.n_sols, args.seed)
    assert tours.shape[1] >= args.n_sols, "start-tour file has fewer slots than --n_sols"
    tours = tours[:, :args.n_sols]
    model, ck = load_checkpoint(args.ckpt, device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    P = (N_AUG if args.aug else 1) * args.n_sols
    chunk = args.chunk or (5 if args.aug else 40) * P
    res = improve(model, prob, inst, tours, weights, ref, args.T, aug=bool(args.aug),
                  elite_sharing=bool(args.elite_sharing), sync_every=args.sync_every,
                  chunk=chunk, hv_steps=parse_steps(args.hv_steps) if args.hv_steps else None,
                  log_every=args.log_every)
    tag = f"aug{N_AUG}" if args.aug else "no-aug"
    sh = f"sharing every {args.sync_every}" if args.elite_sharing else "no sharing"
    print(f"[eval] {os.path.basename(args.ckpt)} n={n} ne={args.ne} T={args.T} {tag} {sh} ref={ref} | "
          f"front HV {res['front_hv'].mean():.4f} | archive-extracted front HV {res['front_arch_hv'].mean():.4f} | "
          f"archive HV {res['arch_hv'].mean():.4f} | |ND| {res['arch_nds'].mean():.1f} | {res['infer_time_s']:.0f}s")
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
