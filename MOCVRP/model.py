import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from layers import RoPESingleLayer, MultiHeadCompat

TWO_PI = 6.283185307179586


class SUPL2I(nn.Module):
    def __init__(self, n_obj=2, view_dim=10, n_ops=3, embedding_dim=128, hidden_dim=64, n_heads=4,
                 n_layers=3, v_range=6.0, rope_nref=80, rope_glob=4):
        super().__init__()
        self.K = n_obj
        self.n_ops = n_ops
        self.range = v_range
        self.n_heads_decoder = n_heads
        # per-objective node embedding of the ring view (one input projection per objective stream)
        self.embeds = nn.ModuleList([nn.Sequential(nn.Linear(view_dim, embedding_dim), nn.ReLU(inplace=True),
                                                   nn.Linear(embedding_dim, embedding_dim))
                                     for _ in range(n_obj)])
        self.layers = nn.ModuleList([RoPESingleLayer(n_heads, embedding_dim, hidden_dim)
                                     for _ in range(n_layers)])
        kd = embedding_dim // n_heads
        self.register_buffer("rope_harm", torch.arange(1, kd // 2 + 1).float())
        # decoder: per-objective compatibility maps fused by a preference-conditioned MLP,
        # one score per operator
        self.compat = MultiHeadCompat(n_heads, embedding_dim, embedding_dim)
        self.ffa1 = nn.Linear(self.K * n_heads, 32)
        self.ffa2 = nn.Linear(32, 32)
        self.ffa3 = nn.Linear(32, n_ops)
        self.ada = nn.Linear(n_obj, 4 * 32)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        # SC-RoPE spectrum: `rope_glob` global harmonics followed by wavelength-aligned
        # harmonics anchored at the reference ring length `rope_nref`
        self.rope_nref = int(rope_nref)
        self.rope_glob = int(rope_glob)
        self._wave_cache = {}
        # preference modulation of the node embeddings (encoder entry)
        self.pref_film = nn.Linear(n_obj, 2 * embedding_dim)
        nn.init.zeros_(self.pref_film.weight)
        nn.init.zeros_(self.pref_film.bias)

    def harmonics(self, n, device):
        # SC-RoPE harmonics k_j of 2*pi/n: the first `rope_glob` planes use k_j = j (global
        # harmonics); the remaining planes keep a fixed wavelength across sizes,
        # k_j = round(t_j * n / rope_nref) with targets t_j log-spaced in [2, M]
        key = (n, str(device))
        if key not in self._wave_cache:
            M = self.rope_harm.numel()
            R, G = self.rope_nref, self.rope_glob
            assert n // 2 >= G, (n, G)
            H = M - G
            targets = [2.0 * (M / 2.0) ** (m / (H - 1)) for m in range(H)]
            ks_hi = [max(1, min(math.floor(t * n / R + 0.5), n // 2)) for t in targets]
            ks = [float(j) for j in range(1, G + 1)] + [float(k) for k in ks_hi]
            self._wave_cache[key] = torch.tensor(ks, dtype=torch.float32, device=device)
        return self._wave_cache[key]

    def score(self, state, pref):
        view, vt, rphase = state["view"], state["vt"], state["rphase"]
        bs, N = vt.size()
        # SC-RoPE angles from the ring position (global frame, period exactly N)
        ang = (TWO_PI / N * vt.float().unsqueeze(-1)) * self.harmonics(N, vt.device)
        emb = torch.cat([ang, ang], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        # route-local frame for the makespan stream: the angle is the node's phase within its route
        ang2 = rphase.unsqueeze(-1) * self.rope_harm
        emb2 = torch.cat([ang2, ang2], dim=-1)
        cos2, sin2 = emb2.cos(), emb2.sin()
        # objective-wise representation: one stream per objective, shared trunk
        streams = [embed(view) for embed in self.embeds]
        # preference injection (encoder): FiLM on the node embeddings;
        # stream k sees the preference rolled by k so that swapping objectives swaps streams
        streams = [(1 + gb[0][:, None, :]) * s + gb[1][:, None, :]
                   for k, s in enumerate(streams)
                   for gb in [self.pref_film(pref.roll(-k, dims=-1)).chunk(2, dim=-1)]]
        for layer in self.layers:
            streams = [layer(s, cos2, sin2) if k == 1 else layer(s, cos, sin) for k, s in enumerate(streams)]
        # decoder: per-objective compatibility maps fused by a preference-conditioned MLP
        H = torch.cat([self.compat(s).permute(1, 2, 3, 0) for s in streams], dim=-1)
        s1, sh1, s2, sh2 = self.ada(pref).chunk(4, dim=-1)
        h = F.relu((1 + s1[:, None, None, :]) * self.ffa1(H) + sh1[:, None, None, :])
        h = F.relu((1 + s2[:, None, None, :]) * self.ffa2(h) + sh2[:, None, None, :])
        return self.ffa3(h)

    def forward(self, state, last_action, pref, do_sample=True, fixed_action=None,
                require_entropy=False, samp_temp=1.0):
        vt, pre = state["vt"], state["pre"]
        bs, gs = vt.size()
        O = self.n_ops
        scores = torch.tanh(self.score(state, pref)) * self.range
        # illegal moves: diagonal, depot-depot 2-opt/exchange, relocate onto the own predecessor,
        # and the reverse of the last move
        rel = torch.zeros(bs, gs, gs, dtype=torch.bool, device=vt.device)
        rel.scatter_(2, pre.unsqueeze(-1), True)
        zero = torch.zeros_like(rel)
        dyn = torch.stack([zero, rel, zero], dim=-1)
        scores = scores.masked_fill(state["mask"] | dyn, -1e20)
        if last_action is not None:
            ar = torch.arange(bs, device=vt.device)
            op_l, a, b, aux = last_action[:, 0], last_action[:, 1], last_action[:, 2], last_action[:, 3]
            scores[ar, a, b, op_l] = -1e20
            r2 = torch.where(op_l == 1, a, b)
            c2 = torch.where(op_l == 1, aux, a)
            scores[ar, r2, c2, op_l] = -1e20
        im = scores.reshape(bs, -1)
        logp = F.log_softmax(im, dim=-1)
        probs = F.softmax(im, dim=-1)

        def decode(idx):
            op = idx % O
            ij = idx // O
            return torch.stack([op, ij // gs, ij % gs], dim=-1)

        if fixed_action is not None:
            act = fixed_action
            pair_index = ((act[:, 1] * gs + act[:, 2]) * O + act[:, 0]).view(-1, 1)
        else:
            if do_sample:
                sl = im if samp_temp == 1.0 else im / samp_temp
                pair_index = F.softmax(sl, dim=-1).multinomial(1)
            else:
                pair_index = probs.max(-1)[1].view(-1, 1)
            act = decode(pair_index.squeeze(-1))
        sel_logp = logp.gather(1, pair_index).squeeze(-1)
        if require_entropy:
            return act, sel_logp, Categorical(probs, validate_args=False).entropy()
        return act, sel_logp


def build_model(args, device):
    return SUPL2I(n_obj=args.get("n_obj", 2), view_dim=args.get("view_dim", 10), n_ops=args.get("n_ops", 3),
                  embedding_dim=args.get("embedding_dim", 128), hidden_dim=args.get("hidden_dim", 64),
                  n_heads=args.get("n_heads", 4), n_layers=args.get("n_layers", 3),
                  v_range=args.get("v_range", 6.0), rope_nref=args.get("rope_nref", 80),
                  rope_glob=args.get("rope_glob", 4)).to(device)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["args"], device)
    model.load_state_dict(ck["actor"])
    model.eval()
    return model, ck
