import math
import numpy as np
import torch

try:
    import hvwfg
except ImportError:
    hvwfg = None

# Hypervolume reference points per problem size (total distance, makespan)
REF = {20: (30.0, 4.0), 50: (45.0, 4.0), 100: (80.0, 4.0)}
CAPACITY = 1.0
CAP_TOL = 1e-6
TWO_PI = 6.283185307179586


def ref_point(n, table=REF):
    if n in table:
        return table[n]
    ks = sorted(table)
    assert ks[0] < n < ks[-1], (n, ks)
    lo = max(k for k in ks if k < n)
    hi = min(k for k in ks if k > n)
    w = (n - lo) / (hi - lo)
    return tuple(a + w * (b - a) for a, b in zip(table[lo], table[hi]))


def make_weights(n_sols=101, device="cpu"):
    w = torch.zeros(n_sols, 2, device=device)
    idx = torch.arange(n_sols, device=device, dtype=torch.float32)
    w[:, 0] = 1.0 - idx / (n_sols - 1)
    w[:, 1] = idx / (n_sols - 1)
    return w / w.sum(dim=1, keepdim=True)


def tcheb(obj, w, ref, rho=0.05):
    """Augmented Tchebycheff scalarisation on reference-normalised objectives:
    g = max_i w_i f_i/ref_i + rho * sum_i w_i f_i/ref_i."""
    wo = w * (obj / ref)
    return wo.max(dim=-1).values + rho * wo.sum(dim=-1)


# nondominance relation used by the acceptance rule
def dominates(a, b):
    return ((a <= b).all(dim=-1)) & ((a < b).any(dim=-1))


def pareto_filter_np(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    s = pts[order]
    keep = np.zeros(len(s), dtype=bool)
    run = np.inf
    for i in range(len(s)):
        if s[i, 1] < run:
            keep[i] = True
            run = s[i, 1]
    return s[keep]


def _hv2d_np(nd, ref):
    y_prev = np.concatenate(([ref[1]], nd[:-1, 1]))
    return float(np.sum((ref[0] - nd[:, 0]) * (y_prev - nd[:, 1])))


def hv_norm(points, ref):
    ref = np.asarray(ref, dtype=np.float64)
    nd = pareto_filter_np(points)
    nd = nd[(nd < ref).all(axis=1)]
    if len(nd) == 0:
        return 0.0
    if hvwfg is not None:
        hv = hvwfg.wfg(nd.astype(np.float64), ref.astype(np.float64))
    else:
        hv = _hv2d_np(nd.astype(np.float64), ref)
    return float(hv / np.prod(ref))


def n_depots(n):
    if 20 <= n < 40:
        r = 0.5
    elif 40 <= n < 70:
        r = 0.4
    elif 70 <= n <= 100:
        r = 0.2
    else:
        raise NotImplementedError(f"no depot-copy count for n={n}")
    return math.ceil(n * (1 + r)) - n


def demand_scaler(n):
    if 20 <= n < 40:
        return 30.0
    if 40 <= n < 70:
        return 40.0
    if 70 <= n <= 100:
        return 50.0
    raise NotImplementedError(f"no demand scaler for n={n}")


def make_instances(bs, n, device="cpu", gen=None):
    depot = torch.rand(bs, 1, 2, device=device, generator=gen)
    nodes = torch.rand(bs, n, 2, device=device, generator=gen)
    dem = torch.randint(1, 10, (bs, n), device=device, generator=gen).float() / demand_scaler(n)
    return depot, nodes, dem


# the solution ring: the n customers followed by K co-located copies of the depot
def build_ring(depot, nodes, demand, K):
    bs = nodes.shape[0]
    xy = torch.cat([nodes, depot.expand(bs, K, 2)], dim=1)
    dem = torch.cat([demand, demand.new_zeros(bs, K)], dim=1)
    return xy, dem


def select_state(revert, s_old, s_new):
    out = {}
    for k in s_old:
        if k == "mask":
            out[k] = s_old[k]
            continue
        m = revert.view(-1, *([1] * (s_old[k].dim() - 1)))
        out[k] = torch.where(m, s_old[k], s_new[k])
    return out


class BiCVRP:
    N_OPS = 3            # 0 = 2-opt, 1 = relocate, 2 = exchange
    VIEW_DIM = 10

    def __init__(self, n, K=None):
        self.n = n
        self.K = n_depots(n) if K is None else K
        self.size = self.n + self.K
        self._mask = None

    @staticmethod
    def seq_to_rec(seq):
        rec = torch.zeros_like(seq)
        rec.scatter_(1, seq, seq.roll(-1, dims=1))
        return rec

    # one walk along the ring: positions, route context, objectives and capacity violation
    def ring_state(self, xy, dem, rec, age):
        bs, N = rec.shape
        dev = rec.device
        n, K = self.n, self.K
        vt = torch.zeros(bs, N, device=dev)
        rid = torch.zeros(bs, N, dtype=torch.long, device=dev)
        rlen = torch.zeros(bs, K + 1, device=dev)
        start = torch.full((bs, 1), n, dtype=torch.long, device=dev)
        cur = start
        r = torch.zeros(bs, 1, dtype=torch.long, device=dev)
        cxy = xy.gather(1, cur.unsqueeze(-1).expand(bs, 1, 2))
        pload = torch.zeros(bs, N, device=dev)
        acc = torch.zeros(bs, 1, device=dev)
        dcum = torch.zeros(bs, N, device=dev)
        drun = torch.zeros(bs, 1, device=dev)
        for t in range(N):
            nxt = rec.gather(1, cur)
            nxy = xy.gather(1, nxt.unsqueeze(-1).expand(bs, 1, 2))
            rlen.scatter_add_(1, r, (nxy - cxy).norm(p=2, dim=-1))
            vt.scatter_(1, nxt, vt.new_full((bs, 1), float(t + 1)))
            r = r + (nxt >= n).long()
            rid.scatter_(1, nxt, r)
            acc = torch.where(nxt >= n, torch.zeros_like(acc), acc + dem.gather(1, nxt))
            pload.scatter_(1, nxt, acc)
            step_len = (nxy - cxy).norm(p=2, dim=-1)
            drun = torch.where(nxt >= n, torch.zeros_like(drun), drun + step_len)
            dcum.scatter_(1, nxt, drun)
            cur, cxy = nxt, nxy
        vt.scatter_(1, start, torch.zeros(bs, 1, device=dev))
        rid.scatter_(1, start, torch.zeros(bs, 1, dtype=torch.long, device=dev))
        ar = torch.arange(N, device=dev).expand(bs, N)
        pre = torch.zeros_like(rec).scatter_(1, rec, ar)
        rload = torch.zeros(bs, K + 1, device=dev).scatter_add_(1, rid, dem)
        f1 = rlen.sum(1)
        f2 = rlen.max(1).values
        over_r = torch.where(rload > CAPACITY + CAP_TOL, rload - CAPACITY, torch.zeros_like(rload))
        viol = over_r.sum(1)
        obj_true = torch.stack([f1, f2], dim=-1)
        # capacity penalty: both objectives are scaled by (1 + viol / f2) while the ring is infeasible
        ok = (viol > 0) & (f2 > 0)
        pf1 = torch.where(ok, f1 + viol * f1 / f2.clamp(min=1e-12), f1)
        pf2 = torch.where(ok, f2 + viol, f2)
        obj = torch.stack([pf1, pf2], dim=-1)
        node_load = rload.gather(1, rid)
        node_over = over_r.gather(1, rid)
        node_len = rlen.gather(1, rid)
        rel = node_len / (f2[:, None] + 1e-9)
        is_dep = (torch.arange(N, device=dev) >= n).float().expand(bs, N)
        # node view: coordinates, depot flag, demand, route load, route overload, prefix load,
        # route length, route length / makespan, recency
        view = torch.cat([xy, torch.stack([is_dep, dem, node_load, node_over, pload], -1),
                          torch.stack([node_len, rel], -1), age.unsqueeze(-1)], -1)
        # phase of every node within its own route (route-local SC-RoPE frame)
        rphase = TWO_PI * dcum / node_len.clamp(min=1e-9)
        return {"vt": vt, "pre": pre, "view": view, "obj": obj, "obj_true": obj_true, "viol": viol,
                "age": age, "rphase": rphase, "mask": self.op_mask(dev)}

    def get_costs(self, xy, dem, rec):
        s = self.ring_state(xy, dem, rec, torch.ones_like(dem))
        return s["obj"], s["obj_true"], s["viol"]

    def two_opt(self, solution, first, second):
        rec = solution.clone()
        argsort = solution.argsort()
        pre_first = argsort.gather(1, first)
        pre_first = torch.where(pre_first != second, pre_first, first)
        rec.scatter_(1, pre_first, second)
        post_second = solution.gather(1, second)
        post_second = torch.where(post_second != first, post_second, second)
        rec.scatter_(1, first, post_second)
        cur = first
        for _ in range(self.size):
            cur_next = solution.gather(1, cur)
            rec.scatter_(1, cur_next, torch.where(cur != second, cur, rec.gather(1, cur_next)))
            cur = torch.where(cur != second, cur_next, cur)
        return rec

    def relocate(self, solution, first, second):
        rec = solution.clone()
        ar = torch.arange(self.size, device=solution.device).expand_as(solution)
        pre = torch.zeros_like(solution).scatter_(1, solution, ar)
        pre_first = pre.gather(1, first)
        next_first = solution.gather(1, first)
        next_second = solution.gather(1, second)
        rec.scatter_(1, pre_first, next_first)
        rec.scatter_(1, second, first)
        rec.scatter_(1, first, next_second)
        return rec

    def exchange(self, solution, first, second):
        ns = solution.gather(1, second)
        swap = ns == first
        i = torch.where(swap, second, first)
        j = torch.where(swap, first, second)
        rec = solution.clone()
        ar = torch.arange(self.size, device=solution.device).expand_as(solution)
        pre = torch.zeros_like(solution).scatter_(1, solution, ar)
        pre_i = pre.gather(1, i)
        next_i = solution.gather(1, i)
        pre_j = pre.gather(1, j)
        next_j = solution.gather(1, j)
        adj = next_i == j
        rec.scatter_(1, pre_j, i)
        rec.scatter_(1, pre_i, j)
        rec.scatter_(1, j, torch.where(adj, i, next_i))
        rec.scatter_(1, i, next_j)
        return rec

    def apply_move(self, solution, act):
        i, j, op = act[:, 1:2], act[:, 2:3], act[:, 0:1]
        out = self.two_opt(solution, i, j)
        out = torch.where(op == 1, self.relocate(solution, i, j), out)
        out = torch.where(op == 2, self.exchange(solution, i, j), out)
        return out

    def op_mask(self, device):
        if self._mask is None or self._mask.device != device:
            N, n = self.size, self.n
            eye = torch.eye(N, dtype=torch.bool, device=device)
            dep = torch.arange(N, device=device) >= n
            dd = dep[:, None] & dep[None, :]
            m = eye[:, :, None].repeat(1, 1, self.N_OPS)
            m[:, :, 0] |= dd
            m[:, :, 2] |= dd
            self._mask = m.unsqueeze(0)
        return self._mask


def age_tick(age, act, decay=0.02):
    a = (age + decay).clamp(max=1.0)
    return a.scatter(1, act[:, 1:2], 0.0).scatter(1, act[:, 2:3], 0.0)


def step_env(model, prob, xy, dem, rec, state, last, pref, **samp):
    act, _ = model(state, last, pref, **samp)
    p_old = state["pre"].gather(1, act[:, 1:2])
    rec_new = prob.apply_move(rec, act)
    s_new = prob.ring_state(xy, dem, rec_new, age_tick(state["age"], act))
    return act, torch.cat([act, p_old], dim=1), rec_new, s_new


def random_rings(n_instances, n, K, n_sols=101, seed=2024):
    g = torch.Generator().manual_seed(seed)
    return torch.argsort(torch.rand(n_instances, n_sols, n + K, generator=g), dim=-1)


def load_rings(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["tours"] if isinstance(d, dict) else d


# every reported solution is re-walked: it must be feasible and reproduce its objectives
def verify_rings(prob, xy_all, dem_all, rec_all, obj_all, ok, chunk):
    for c0 in range(0, len(ok), chunk):
        sl = slice(c0, min(c0 + chunk, len(ok)))
        m = ok[sl]
        if not m.any():
            continue
        chk = prob.ring_state(xy_all[sl][m], dem_all[sl][m], rec_all[sl][m], torch.ones_like(dem_all[sl][m]))
        assert (chk["viol"] <= CAP_TOL).all(), "infeasible solution in the reported set"
        assert torch.allclose(chk["obj_true"], obj_all[sl][m], atol=1e-3), "objective mismatch on re-evaluation"
