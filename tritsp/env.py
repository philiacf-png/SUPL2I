import numpy as np
import torch

try:
    import hvwfg
except ImportError:
    hvwfg = None

# Hypervolume reference points per problem size
REF = {20: (20.0, 20.0, 20.0), 50: (35.0, 35.0, 35.0), 100: (65.0, 65.0, 65.0)}


def ref_point(n, table=REF):
    if n in table:
        return table[n]
    ks = sorted(table)
    assert ks[0] < n < ks[-1], (n, ks)
    lo = max(k for k in ks if k < n)
    hi = min(k for k in ks if k > n)
    w = (n - lo) / (hi - lo)
    return tuple(a + w * (b - a) for a, b in zip(table[lo], table[hi]))


def make_weights(n_sols=105, device="cpu", n_obj=3):
    # Das-Dennis simplex lattice; n_sols must be C(H + n_obj - 1, n_obj - 1) for some H
    def lattice(H, K):
        if K == 1:
            return [[H]]
        return [[i] + rest for i in range(H + 1) for rest in lattice(H - i, K - 1)]

    H = 1
    while len(lattice(H, n_obj)) < n_sols:
        H += 1
    rows = lattice(H, n_obj)
    assert len(rows) == n_sols, f"n_sols={n_sols} is not a simplex-lattice size for {n_obj} objectives"
    return torch.tensor(rows, dtype=torch.float32, device=device) / H


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
    if pts.shape[1] == 2:
        s = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        keep = np.zeros(len(s), dtype=bool)
        run = np.inf
        for i in range(len(s)):
            if s[i, 1] < run:
                keep[i] = True
                run = s[i, 1]
        return s[keep]
    s = np.unique(pts, axis=0)
    nd = np.empty((0, s.shape[1]))
    for c0 in range(0, len(s), 2048):
        blk = s[c0:c0 + 2048]
        if len(nd):
            dom = ((nd[None] <= blk[:, None]).all(-1) & (nd[None] < blk[:, None]).any(-1)).any(1)
            blk = blk[~dom]
        if len(blk):
            le = (blk[None] <= blk[:, None]).all(-1)
            lt = (blk[None] < blk[:, None]).any(-1)
            blk = blk[~(le & lt).any(1)]
            nd = np.concatenate([nd, blk], 0)
    return nd


def nd_filter(pts, block=2048, nd_slab=8192):
    s = torch.unique(pts, dim=0)
    nd = s[:0]
    for c0 in range(0, len(s), block):
        blk = s[c0:c0 + block]
        if len(nd):
            keep = torch.ones(len(blk), dtype=torch.bool, device=s.device)
            for d0 in range(0, len(nd), nd_slab):
                sl = nd[d0:d0 + nd_slab]
                dom = ((sl[None] <= blk[:, None]).all(-1) & (sl[None] < blk[:, None]).any(-1)).any(1)
                keep &= ~dom
            blk = blk[keep]
        if len(blk):
            le = (blk[None] <= blk[:, None]).all(-1)
            lt = (blk[None] < blk[:, None]).any(-1)
            blk = blk[~(le & lt).any(1)]
            nd = torch.cat([nd, blk], 0)
    return nd


def _hv_np(nd, ref):
    if nd.shape[1] == 2:
        s = nd[np.argsort(nd[:, 0])]
        y_prev = np.concatenate(([ref[1]], s[:-1, 1]))
        return float(np.sum((ref[0] - s[:, 0]) * (y_prev - s[:, 1])))
    s = nd[np.argsort(nd[:, -1])]
    hv = 0.0
    for i in range(len(s)):
        z_hi = s[i + 1, -1] if i + 1 < len(s) else ref[-1]
        if z_hi > s[i, -1]:
            hv += (z_hi - s[i, -1]) * _hv_np(pareto_filter_np(s[:i + 1, :-1]), ref[:-1])
    return hv


def hv_norm(points, ref):
    ref = np.asarray(ref, dtype=np.float64)
    nd = pareto_filter_np(points)
    nd = nd[(nd < ref).all(axis=1)]
    if len(nd) == 0:
        return 0.0
    if hvwfg is not None:
        hv = hvwfg.wfg(nd.astype(np.float64), ref.astype(np.float64))
    else:
        hv = _hv_np(nd.astype(np.float64), ref)
    return float(hv / np.prod(ref))


class TriTSP:
    def __init__(self, size, n_obj=3):
        self.size = size
        self.n_obj = int(n_obj)

    @staticmethod
    def seq_to_rec(seq):
        rec = torch.zeros_like(seq)
        rec.scatter_(1, seq, seq.roll(-1, dims=1))
        return rec

    def get_costs(self, coords, rec):
        bs, n = rec.size()
        idx = rec.long().unsqueeze(-1).expand(bs, n, 2)
        lengths = []
        for k in range(self.n_obj):
            ck = coords[:, :, 2 * k:2 * k + 2]
            lengths.append((ck.gather(1, idx) - ck).norm(p=2, dim=2).sum(1))
        return torch.stack(lengths, dim=-1)

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

    def get_swap_mask(self, visited_time):
        bs, gs = visited_time.size()
        eye = torch.eye(gs, device=visited_time.device).view(1, gs, gs)
        return eye.expand(bs, gs, gs).bool()


def do_move(prob, rec, pair):
    return prob.two_opt(rec, pair[:, :1], pair[:, 1:])


def age_tick(age, pair, decay=0.02):
    a = (age + decay).clamp(max=1.0)
    return a.scatter(1, pair[:, :1], 0.0).scatter(1, pair[:, 1:2], 0.0)


def random_tours(n_instances, n, n_sols=105, seed=2024):
    g = torch.Generator().manual_seed(seed)
    return torch.argsort(torch.rand(n_instances, n_sols, n, generator=g), dim=-1)


def load_tours(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["tours"] if isinstance(d, dict) else d
