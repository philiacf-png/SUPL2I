import argparse, os, time
import numpy as np
import torch

from env import (BiCVRP, REF, CAP_TOL, ref_point, make_weights, tcheb, dominates, hv_norm, pareto_filter_np,
                 make_instances, build_ring, select_state, age_tick, step_env, verify_rings, random_rings)
from model import SUPL2I

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints")

N_STEP = 5
K_EPOCHS = 3
GAMMA = 0.999
EPS_CLIP = 0.1
ENT_COEF = 0.001
REWARD_SCALE = 100.0
MAX_GRAD_NORM = 0.04
CURRICULUM = 0.5
EVAL_TEMP = 1.5


def parse_sizes(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    assert out and sorted(set(out)) == out, spec
    return out


def batch_size(N):
    return max(8, min(32, round(1920 / N)))


@torch.no_grad()
def evaluate(model, prob, depot, nodes, demand, rings, weights, ref, T_eval, samp_temp=EVAL_TEMP, chunk=2020):
    device = depot.device
    Ne, N = nodes.shape[0], prob.size
    W = weights.shape[0]
    ref_t = torch.tensor(ref, device=device)
    xy_i, dem_i = build_ring(depot, nodes, demand, prob.K)
    inst_idx = torch.arange(Ne).repeat_interleave(W)
    xy_all, dem_all = xy_i[inst_idx], dem_i[inst_idx]
    rec_all = BiCVRP.seq_to_rec(rings.reshape(Ne * W, N).long().to(device))
    pref_all = weights.repeat(Ne, 1).to(device)
    T = Ne * W
    INF = float("inf")
    best_obj = torch.full((T, 2), INF, device=device)
    best_g = torch.full((T,), INF, device=device)
    best_rec = torch.zeros(T, N, dtype=torch.long, device=device)
    arch_hv = np.zeros(Ne)
    model.eval()
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        b = c1 - c0
        xy, dem, rec, pref = xy_all[c0:c1], dem_all[c0:c1], rec_all[c0:c1], pref_all[c0:c1]
        last = None
        vis = torch.full((T_eval + 1, b, 2), INF, device=device)
        st = prob.ring_state(xy, dem, rec, torch.ones_like(dem))
        feas = st["viol"] <= CAP_TOL
        vis[0] = torch.where(feas[:, None], st["obj_true"], vis[0])
        g = tcheb(st["obj"], pref, ref_t)
        best_g[c0:c1] = torch.where(feas, g, best_g[c0:c1])
        best_obj[c0:c1] = torch.where(feas[:, None], st["obj_true"], best_obj[c0:c1])
        best_rec[c0:c1] = torch.where(feas[:, None], rec, best_rec[c0:c1])
        for t in range(T_eval):
            _, last_new, rec_new, s_new = step_env(model, prob, xy, dem, rec, st, last, pref,
                                                  do_sample=True, samp_temp=samp_temp)
            feas = s_new["viol"] <= CAP_TOL
            vis[t + 1] = torch.where(feas[:, None], s_new["obj_true"], vis[t + 1])
            g = tcheb(s_new["obj"], pref, ref_t)
            imp = feas & (g < best_g[c0:c1])
            best_g[c0:c1] = torch.where(imp, g, best_g[c0:c1])
            best_obj[c0:c1] = torch.where(imp[:, None], s_new["obj_true"], best_obj[c0:c1])
            best_rec[c0:c1] = torch.where(imp[:, None], rec_new, best_rec[c0:c1])
            revert = dominates(st["obj"], s_new["obj"])
            rec = torch.where(revert[:, None], rec, rec_new)
            st = select_state(revert, st, s_new)
            last = last_new
        vis_np = vis.permute(1, 0, 2).cpu().numpy()
        for j in range(b // W):
            pts = vis_np[j * W:(j + 1) * W].reshape(-1, 2)
            arch_hv[c0 // W + j] = hv_norm(pareto_filter_np(pts), ref)
    ok = best_g < INF
    verify_rings(prob, xy_all, dem_all, best_rec, best_obj, ok, chunk)
    bo = best_obj.cpu().numpy().reshape(Ne, W, 2)
    front_hv = np.array([hv_norm(bo[i], ref) for i in range(Ne)])
    return float(front_hv.mean()), float(arch_hv.mean()), float(ok.float().mean().item())


def train(args):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    K = 2
    iters = args.epochs * args.batches
    sizes = parse_sizes(args.sizes)
    esizes = parse_sizes(args.eval_sizes)
    probs = {m: BiCVRP(m) for m in sorted(set(sizes + esizes))}
    ref_ts = {m: torch.tensor(ref_point(m), device=device) for m in sizes}
    os.makedirs(CKPT, exist_ok=True)
    weights = make_weights(args.n_sols, device=device)
    Wm = args.n_sols
    ne = args.eval_subset
    # validation set: random instances and start rings drawn with a fixed seed
    # (independent of the training stream and of the test instances)
    ev_inst, ev_rings = {}, {}
    for m in esizes:
        g_val = torch.Generator().manual_seed(args.val_seed + m)
        ev_inst[m] = tuple(x.to(device) for x in make_instances(ne, m, gen=g_val))
        ev_rings[m] = random_rings(ne, m, probs[m].K, Wm, seed=args.val_seed + m)

    model = SUPL2I(n_obj=K).to(device)
    model_args = {"n_obj": K, "view_dim": BiCVRP.VIEW_DIM, "n_ops": BiCVRP.N_OPS, "embedding_dim": 128,
                  "hidden_dim": 64, "n_heads": 4, "n_layers": 3, "v_range": 6.0, "rope_nref": 80, "rope_glob": 4,
                  "sizes": args.sizes, "eval_sizes": args.eval_sizes, "epochs": args.epochs, "batches": args.batches}
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay) if args.lr_decay < 1.0 else None
    best_hv, best_path, start_it = -1.0, None, 1
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["actor"])
        opt.load_state_dict(ck["opt"])
        if sched is not None and ck.get("sched") is not None:
            sched.load_state_dict(ck["sched"])
        best_hv, best_path, start_it = ck["best_hv"], ck.get("best_path"), ck["it"] + 1
        r = ck["rng"]
        torch.set_rng_state(r["torch"].cpu())
        if torch.cuda.is_available() and r["cuda"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in r["cuda"]])
        np.random.set_state(r["numpy"])
        g_state = r["gen"].cpu()
        if g_state.numel() == gen.get_state().numel():
            gen.set_state(g_state)
        else:
            gen.manual_seed(args.seed)
            print("[resume] the checkpoint was written on a different device; the sampling stream restarts from --seed")
        print(f"[resume] {args.resume}: continuing from iteration {ck['it']} (epoch {ck['it'] // args.batches})")
    print(f"[init] SUPL2I | parameters {sum(p.numel() for p in model.parameters())} | sizes {args.sizes} | "
          f"{args.epochs} epochs x {args.batches} iterations", flush=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(HERE, "runs", args.tag))
    except ImportError:
        pass

    # curriculum: warm-start depth ramps from 0 to CURRICULUM * T_train (sigmoid)
    if iters > 1:
        x = np.linspace(0.0, 1.0, iters)
        y = 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))
        y = (y - y.min()) / (y.max() - y.min())
        warm_sched = (y * CURRICULUM * args.T_train).astype(int)
    else:
        warm_sched = np.zeros(1, dtype=int)
    last_path = os.path.join(CKPT, f"supl2i-{args.tag}-last.pt")

    def snapshot(it_now):
        return {"actor": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict() if sched is not None else None,
                "it": it_now, "epoch": it_now // args.batches, "best_hv": best_hv, "best_path": best_path,
                "rng": {"torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        "numpy": np.random.get_state(), "gen": gen.get_state()},
                "args": model_args}

    def save_atomic(d, path):
        torch.save(d, path + ".tmp")
        os.replace(path + ".tmp", path)

    for it in range(start_it, iters + 1):
        model.train()
        epoch, batch = (it - 1) // args.batches + 1, (it - 1) % args.batches + 1
        n = sizes[int(torch.randint(len(sizes), (1,), generator=gen, device=device).item())]
        prob, ref_t = probs[n], ref_ts[n]
        N = prob.size
        bs_inst = batch_size(N)
        B = bs_inst * Wm
        u = torch.rand(B, 1, generator=gen, device=device)
        pref = torch.cat([u, 1.0 - u], dim=1)
        depot, nodes, dm = make_instances(bs_inst, n, device=device, gen=gen)
        xy_i, dem_i = build_ring(depot, nodes, dm, prob.K)
        rec = BiCVRP.seq_to_rec(torch.argsort(torch.rand(B, N, device=device, generator=gen), dim=1))
        xy = xy_i.repeat_interleave(Wm, dim=0)
        dem = dem_i.repeat_interleave(Wm, dim=0)
        group = torch.arange(bs_inst, device=device).repeat_interleave(Wm)

        # curriculum warm-start: the current policy pre-improves the start rings
        warm = int(warm_sched[it - 1])
        warm_last = None
        state = prob.ring_state(xy, dem, rec, torch.ones_like(dem))
        if warm > 0:
            model.eval()
            with torch.no_grad():
                for _ in range(warm):
                    _, warm_last, rec_new, s_new = step_env(model, prob, xy, dem, rec, state, warm_last, pref,
                                                            do_sample=True)
                    revert = dominates(state["obj"], s_new["obj"])
                    rec = torch.where(revert[:, None], rec, rec_new)
                    state = select_state(revert, state, s_new)
            model.train()

        last = warm_last
        ep_reward = ep_entropy = ep_viol = ep_live = ep_rev = 0.0
        t = n_win = 0
        g_best = tcheb(state["obj"], pref, ref_t)
        while t < args.T_train:
            buf_state, buf_pref, buf_act, buf_logp, buf_last, buf_r = [], [], [], [], [], []
            for _ in range(N_STEP):
                if t >= args.T_train:
                    break
                with torch.no_grad():
                    act, logp, ent = model(state, last, pref, do_sample=True, require_entropy=True)
                buf_state.append(state); buf_pref.append(pref); buf_act.append(act)
                buf_logp.append(logp.detach()); buf_last.append(last)
                g_old = tcheb(state["obj"], pref, ref_t)
                p_old = state["pre"].gather(1, act[:, 1:2])
                rec_new = prob.apply_move(rec, act)
                s_new = prob.ring_state(xy, dem, rec_new, age_tick(state["age"], act))
                g_new = tcheb(s_new["obj"], pref, ref_t)
                revert = dominates(state["obj"], s_new["obj"])   # nondominance-based acceptance
                ep_rev += float(revert.float().mean().item())
                rec = torch.where(revert[:, None], rec, rec_new)
                state = select_state(revert, state, s_new)
                g_new = torch.where(revert, g_old, g_new)
                buf_r.append((g_best - g_new).clamp(min=0) * REWARD_SCALE)   # ratchet reward
                g_best = torch.minimum(g_best, g_new)
                last = torch.cat([act, p_old], dim=1)
                ep_reward += buf_r[-1].mean().item()
                ep_entropy += ent.mean().item()
                ep_viol += state["viol"].mean().item()
                t += 1

            R = torch.zeros(B, device=device)
            returns = []
            for r in reversed(buf_r):
                R = r + GAMMA * R
                returns.append(R.clone())
            returns = returns[::-1]
            live_list = []
            for Rt in returns:
                gm = torch.zeros(bs_inst, device=device).index_add_(0, group, (Rt > 0).float())
                live_list.append((gm > 0)[group])
            ep_live += float(torch.stack(live_list).float().mean().item())
            n_win += 1
            # advantage: per-instance group baseline
            adv_list = []
            for Rt in returns:
                sums = torch.zeros(bs_inst, device=device).index_add_(0, group, Rt)
                cnt = torch.zeros(bs_inst, device=device).index_add_(0, group, torch.ones_like(Rt))
                adv_list.append(Rt - (sums / cnt.clamp(min=1))[group])
            adv = torch.stack(adv_list, 0)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            L = len(returns)
            for _ in range(K_EPOCHS):
                opt.zero_grad()
                for li in range(L):
                    _, lp, e = model(buf_state[li], buf_last[li], buf_pref[li], fixed_action=buf_act[li],
                                     require_entropy=True)
                    ratio = torch.exp((lp - buf_logp[li]).clamp(-20.0, 20.0))
                    surr1 = ratio * adv[li]
                    surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * adv[li]
                    loss = (-torch.min(surr1, surr2).mean() - ENT_COEF * e.mean()) / L
                    loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                if torch.isfinite(gn):
                    opt.step()
                else:
                    print(f"[guard] non-finite gradients at iteration {it}: update skipped", flush=True)

        if sched is not None and it >= args.lr_decay_start and it % args.lr_decay_every == 0:
            sched.step()
        cur_ent = ep_entropy / max(1, args.T_train)
        if writer is not None:
            writer.add_scalar("train/lr", opt.param_groups[0]["lr"], it)
            writer.add_scalar("train/reward_sum", ep_reward, it)
            writer.add_scalar("train/entropy", cur_ent, it)
            writer.add_scalar("train/viol_mean", ep_viol / max(1, args.T_train), it)
            writer.add_scalar("train/live_frac", ep_live / max(1, n_win), it)
        print(f"Epoch {epoch:3d}/{args.epochs} | iteration {batch:2d}/{args.batches} (global {it:4d}) | n={n:3d} | "
              f"reward {ep_reward:.4f} | entropy {cur_ent:.3f} | viol {ep_viol / max(1, args.T_train):.4f} | "
              f"live {ep_live / max(1, n_win):.3f} | revert {ep_rev / max(1, args.T_train):.2f} | "
              f"{time.time() - t0:.0f}s", flush=True)

        if it % args.eval_every == 0 or it == iters:
            save_atomic(snapshot(it), last_path)
            fhs, ahs, msg = [], [], []
            for m in esizes:
                fm, am, ff = evaluate(model, probs[m], *ev_inst[m], ev_rings[m], weights, REF[m], args.T_eval,
                                      chunk=ne * Wm)
                fhs.append(fm); ahs.append(am)
                msg.append(f"n{m} {fm:.4f}/{am:.4f} (feasible {ff:.2f})")
                if writer is not None:
                    writer.add_scalar(f"eval/front_hv_n{m}", fm, it)
                    writer.add_scalar(f"eval/archive_hv_n{m}", am, it)
                    writer.add_scalar(f"eval/feasible_frac_n{m}", ff, it)
            fh, ah = sum(fhs) / len(fhs), sum(ahs) / len(ahs)
            print(f"   [eval] epoch {epoch} (iteration {it}) | " + " | ".join(msg)
                  + f" | mean front/archive HV {fh:.4f}/{ah:.4f}", flush=True)
            if ah > best_hv:
                best_hv = ah
                new_best = os.path.join(CKPT, f"supl2i-{args.tag}-epoch-{epoch}.pt")
                if best_path and best_path != new_best and os.path.exists(best_path):
                    os.remove(best_path)
                best_path = new_best
                save_atomic(snapshot(it), best_path)
                print(f"   [best] mean archive HV {ah:.4f} -> {os.path.basename(best_path)}", flush=True)
        save_atomic(snapshot(it), last_path)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batches", type=int, default=10, help="iterations per epoch")
    ap.add_argument("--sizes", default="20-100", help="training sizes, e.g. '20-100' or '50,100'")
    ap.add_argument("--eval_sizes", default="20,50,100")
    ap.add_argument("--eval_every", type=int, default=100, help="iterations between evaluations")
    ap.add_argument("--eval_subset", type=int, default=25, help="evaluation instances per size")
    ap.add_argument("--T_train", type=int, default=200, help="improvement steps per training episode")
    ap.add_argument("--T_eval", type=int, default=500, help="improvement steps in the evaluation walk")
    ap.add_argument("--n_sols", type=int, default=101, help="preference vectors per instance")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_decay", type=float, default=0.98)
    ap.add_argument("--lr_decay_every", type=int, default=10)
    ap.add_argument("--lr_decay_start", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="bicvrp")
    ap.add_argument("--val_seed", type=int, default=7, help="seed of the random validation instances")
    ap.add_argument("--resume", default="", help="checkpoint to continue from")
    train(ap.parse_args())
