import argparse, os, time
import numpy as np
import torch

from env import BiTSP, REF, ref_point, make_weights, tcheb, dominates, hv_norm, pareto_filter_np, \
    do_move, age_tick, random_tours
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


def batch_size(n):
    return max(16, min(48, round(1600 / n)))


@torch.no_grad()
def evaluate(model, prob, instances, tours, weights, ref, T_eval, samp_temp=EVAL_TEMP, chunk=4040):
    device = instances.device
    Ne, n = instances.shape[0], prob.size
    W, K = weights.shape
    ref_t = torch.tensor(ref, device=device)
    inst_idx = torch.arange(Ne).repeat_interleave(W)
    coords_all = instances[inst_idx]
    rec_all = BiTSP.seq_to_rec(tours.reshape(Ne * W, n).to(device))
    pref_all = weights.repeat(Ne, 1).to(device)
    T = Ne * W
    best_obj = torch.zeros(T, K, device=device)
    best_g = torch.full((T,), float("inf"), device=device)
    arch_hv = np.zeros(Ne)
    model.eval()
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        b = c1 - c0
        coords, rec, pref = coords_all[c0:c1], rec_all[c0:c1], pref_all[c0:c1]
        last = None
        age = torch.ones(b, n, device=device)
        vis = torch.empty(T_eval + 1, b, K, device=device)
        obj = prob.get_costs(coords, rec)
        vis[0] = obj
        g = tcheb(obj, pref, ref_t)
        best_g[c0:c1] = g
        best_obj[c0:c1] = obj
        for t in range(T_eval):
            o_cur = prob.get_costs(coords, rec)
            pair, _ = model(coords, rec, last, pref, age, do_sample=True, samp_temp=samp_temp)
            rec_new = do_move(prob, rec, pair)
            obj = prob.get_costs(coords, rec_new)
            vis[t + 1] = obj
            g = tcheb(obj, pref, ref_t)
            revert = dominates(o_cur, obj)
            rec = torch.where(revert[:, None], rec, rec_new)
            last = pair
            age = torch.where(revert[:, None], age, age_tick(age, pair))
            imp = g < best_g[c0:c1]
            best_g[c0:c1] = torch.where(imp, g, best_g[c0:c1])
            best_obj[c0:c1] = torch.where(imp[:, None], obj, best_obj[c0:c1])
        vis_np = vis.permute(1, 0, 2).cpu().numpy()
        for j in range(b // W):
            pts = vis_np[j * W:(j + 1) * W].reshape(-1, K)
            arch_hv[c0 // W + j] = hv_norm(pareto_filter_np(pts), ref)
    best_obj = best_obj.cpu().numpy().reshape(Ne, W, K)
    front_hv = np.array([hv_norm(best_obj[i], ref) for i in range(Ne)])
    return float(front_hv.mean()), float(arch_hv.mean())


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
    probs = {m: BiTSP(m) for m in sorted(set(sizes + esizes))}
    ref_ts = {m: torch.tensor(ref_point(m), device=device) for m in sizes}
    os.makedirs(CKPT, exist_ok=True)
    weights = make_weights(args.n_sols, device=device)
    Wm = args.n_sols
    ne = args.eval_subset
    # validation set: random instances and start tours drawn with a fixed seed
    # (independent of the training stream and of the test instances)
    ev_inst, ev_tours = {}, {}
    for m in esizes:
        g_val = torch.Generator().manual_seed(args.val_seed + m)
        ev_inst[m] = torch.rand(ne, m, 2 * K, generator=g_val).to(device)
        ev_tours[m] = random_tours(ne, m, Wm, seed=args.val_seed + m)

    model = SUPL2I(n_obj=K).to(device)
    model_args = {"n_obj": K, "embedding_dim": 128, "hidden_dim": 64, "n_heads": 4, "n_layers": 3,
                  "v_range": 6.0, "rope_nref": 50, "rope_glob": 4, "sizes": args.sizes,
                  "eval_sizes": args.eval_sizes, "epochs": args.epochs, "batches": args.batches}
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
        bs_inst = batch_size(n)
        B = bs_inst * Wm
        u = torch.rand(B, 1, generator=gen, device=device)
        pref = torch.cat([u, 1.0 - u], dim=1)
        inst = torch.rand(bs_inst, n, 2 * K, generator=gen, device=device)
        rec = BiTSP.seq_to_rec(torch.argsort(torch.rand(B, n, device=device), dim=1))
        coords = inst.repeat_interleave(Wm, dim=0)
        group = torch.arange(bs_inst, device=device).repeat_interleave(Wm)

        # curriculum warm-start: the current policy pre-improves the start tours
        warm = int(warm_sched[it - 1])
        warm_last = None
        age = torch.ones(B, n, device=device)
        if warm > 0:
            model.eval()
            with torch.no_grad():
                for _ in range(warm):
                    pair, _ = model(coords, rec, warm_last, pref, age, do_sample=True)
                    o_old = prob.get_costs(coords, rec)
                    rec_new = do_move(prob, rec, pair)
                    o_new = prob.get_costs(coords, rec_new)
                    revert = dominates(o_old, o_new)
                    rec = torch.where(revert[:, None], rec, rec_new)
                    age = torch.where(revert[:, None], age, age_tick(age, pair))
                    warm_last = pair
            model.train()

        last = warm_last
        ep_reward = ep_entropy = ep_live = ep_rev = 0.0
        t = n_win = 0
        g_best = tcheb(prob.get_costs(coords, rec), pref, ref_t)
        while t < args.T_train:
            buf_state, buf_pref, buf_act, buf_logp, buf_last, buf_age, buf_r = [], [], [], [], [], [], []
            for _ in range(N_STEP):
                if t >= args.T_train:
                    break
                with torch.no_grad():
                    pair, logp, ent = model(coords, rec, last, pref, age, do_sample=True, require_entropy=True)
                buf_state.append(rec); buf_pref.append(pref); buf_act.append(pair)
                buf_logp.append(logp.detach()); buf_last.append(last); buf_age.append(age)
                o_old = prob.get_costs(coords, rec)
                g_old = tcheb(o_old, pref, ref_t)
                rec_new = do_move(prob, rec, pair)
                obj = prob.get_costs(coords, rec_new)
                g_new = tcheb(obj, pref, ref_t)
                revert = dominates(o_old, obj)                    # nondominance-based acceptance
                ep_rev += float(revert.float().mean().item())
                rec = torch.where(revert[:, None], rec, rec_new)
                age = torch.where(revert[:, None], age, age_tick(age, pair))
                g_new = torch.where(revert, g_old, g_new)
                buf_r.append((g_best - g_new).clamp(min=0) * REWARD_SCALE)   # ratchet reward
                g_best = torch.minimum(g_best, g_new)
                last = pair
                ep_reward += buf_r[-1].mean().item()
                ep_entropy += ent.mean().item()
                t += 1

            R = torch.zeros(B, device=device)
            returns = []
            for r in reversed(buf_r):
                R = r + GAMMA * R
                returns.append(R.clone())
            returns = returns[::-1]
            # advantage: per-instance group baseline, normalised over the live groups
            live_list = []
            for Rt in returns:
                gm = torch.zeros(bs_inst, device=device).index_add_(0, group, (Rt > 0).float())
                live_list.append((gm > 0)[group])
            ep_live += float(torch.stack(live_list).float().mean().item())
            n_win += 1
            adv_list = []
            for Rt in returns:
                sums = torch.zeros(bs_inst, device=device).index_add_(0, group, Rt)
                cnt = torch.zeros(bs_inst, device=device).index_add_(0, group, torch.ones_like(Rt))
                adv_list.append(Rt - (sums / cnt.clamp(min=1))[group])
            adv = torch.stack(adv_list, 0)
            lv = torch.stack(live_list, 0)
            m = adv[lv].mean() if lv.any() else adv.mean()
            s = adv[lv].std() if lv.sum() > 1 else adv.std()
            adv = (adv - m) / (s + 1e-8)

            L = len(returns)
            for _ in range(K_EPOCHS):
                opt.zero_grad()
                for li in range(L):
                    _, lp, e = model(coords, buf_state[li], buf_last[li], buf_pref[li], buf_age[li],
                                     fixed_action=buf_act[li], require_entropy=True)
                    ratio = torch.exp(lp - buf_logp[li])
                    surr1 = ratio * adv[li]
                    surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * adv[li]
                    lvi = live_list[li]
                    pg = -(torch.min(surr1, surr2) * lvi).sum() / lvi.sum().clamp(min=1)
                    loss = (pg - ENT_COEF * e.mean()) / L
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()

        if sched is not None and it >= args.lr_decay_start and it % args.lr_decay_every == 0:
            sched.step()
        cur_ent = ep_entropy / max(1, args.T_train)
        if writer is not None:
            writer.add_scalar("train/lr", opt.param_groups[0]["lr"], it)
            writer.add_scalar("train/reward_sum", ep_reward, it)
            writer.add_scalar("train/entropy", cur_ent, it)
            writer.add_scalar("train/live_frac", ep_live / max(1, n_win), it)
        print(f"Epoch {epoch:3d}/{args.epochs} | iteration {batch:2d}/{args.batches} (global {it:4d}) | n={n:3d} | "
              f"reward {ep_reward:.4f} | entropy {cur_ent:.3f} | live {ep_live / max(1, n_win):.3f} | "
              f"revert {ep_rev / max(1, args.T_train):.2f} | {time.time() - t0:.0f}s", flush=True)

        if it % args.eval_every == 0 or it == iters:
            save_atomic(snapshot(it), last_path)
            fhs, ahs, msg = [], [], []
            for m in esizes:
                fm, am = evaluate(model, probs[m], ev_inst[m], ev_tours[m], weights, REF[m], args.T_eval)
                fhs.append(fm); ahs.append(am)
                msg.append(f"n{m} {fm:.4f}/{am:.4f}")
                if writer is not None:
                    writer.add_scalar(f"eval/front_hv_n{m}", fm, it)
                    writer.add_scalar(f"eval/archive_hv_n{m}", am, it)
            fh, ah = sum(fhs) / len(fhs), sum(ahs) / len(ahs)
            print(f"   [eval] epoch {epoch} (iteration {it}) | " + " | ".join(msg)
                  + f" | mean front/archive HV {fh:.4f}/{ah:.4f}", flush=True)
            if fh > best_hv:
                best_hv = fh
                new_best = os.path.join(CKPT, f"supl2i-{args.tag}-epoch-{epoch}.pt")
                if best_path and best_path != new_best and os.path.exists(best_path):
                    os.remove(best_path)
                best_path = new_best
                save_atomic(snapshot(it), best_path)
                print(f"   [best] mean front HV {fh:.4f} -> {os.path.basename(best_path)}", flush=True)
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
    ap.add_argument("--tag", default="bitsp")
    ap.add_argument("--val_seed", type=int, default=7, help="seed of the random validation instances")
    ap.add_argument("--resume", default="", help="checkpoint to continue from")
    train(ap.parse_args())
