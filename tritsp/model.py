import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from layers import RoPESingleLayer, MultiHeadCompat

TWO_PI = 6.283185307179586


class SUPL2I(nn.Module):
    def __init__(self, n_obj=3, embedding_dim=128, hidden_dim=64, n_heads=4, n_layers=3,
                 v_range=6.0, rope_nref=50, rope_glob=4):
        super().__init__()
        self.K = n_obj
        self.range = v_range
        self.n_heads_decoder = n_heads
        # per-objective node embedding: (x, y) of the objective's coordinates + recency
        self.node_embed = nn.Sequential(nn.Linear(3, embedding_dim), nn.ReLU(inplace=True),
                                        nn.Linear(embedding_dim, embedding_dim))
        self.layers = nn.ModuleList([RoPESingleLayer(n_heads, embedding_dim, hidden_dim)
                                     for _ in range(n_layers)])
        kd = embedding_dim // n_heads
        self.register_buffer("rope_harm", torch.arange(1, kd // 2 + 1).float())
        # cross-objective compatibility: preference-gated maps between every ordered pair of
        # streams (queries from one objective, keys from another); the gate starts at zero
        self.cc_gate = nn.Sequential(nn.Linear(n_obj, 8), nn.ReLU(inplace=True), nn.Linear(8, 1))
        nn.init.zeros_(self.cc_gate[2].weight)
        nn.init.zeros_(self.cc_gate[2].bias)
        # decoder: per-objective (and cross-objective) compatibility maps fused by a
        # preference-conditioned MLP
        self.compat = MultiHeadCompat(n_heads, embedding_dim, embedding_dim)
        self.ffa1 = nn.Linear(self.K * n_heads + self.K * (self.K - 1) * n_heads, 32)
        self.ffa2 = nn.Linear(32, 32)
        self.ffa3 = nn.Linear(32, 1)
        self.ada = nn.Linear(n_obj, 4 * 32)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        # SC-RoPE spectrum: `rope_glob` global harmonics followed by wavelength-aligned
        # harmonics anchored at the reference size `rope_nref`
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

    @staticmethod
    def swap_mask(visited_time):
        bs, gs = visited_time.size()
        eye = torch.eye(gs, device=visited_time.device).view(1, gs, gs)
        return eye.expand(bs, gs, gs).bool()

    @staticmethod
    def visited_time(solution):
        bs, n = solution.size()
        vt = torch.zeros(bs, n, device=solution.device)
        ar = torch.arange(bs, device=solution.device)
        pre = torch.zeros(bs, dtype=torch.long, device=solution.device)
        for i in range(n):
            nxt = solution[ar, pre]
            vt[ar, nxt] = i + 1
            pre = nxt
        return vt

    def score(self, coords, pref, visited_time, age):
        bs, n, _ = coords.size()
        # SC-RoPE angles of every node from its position in the current tour
        ang = (TWO_PI / n * visited_time.float().unsqueeze(-1)) * self.harmonics(n, coords.device)
        emb = torch.cat([ang, ang], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        # objective-wise representation: one stream per objective, shared weights
        streams = [self.node_embed(torch.cat([coords[..., 2 * k:2 * k + 2], age.unsqueeze(-1)], dim=-1))
                   for k in range(self.K)]
        # preference injection (encoder): FiLM on the node embeddings;
        # stream k sees the preference rolled by k so that permuting objectives permutes streams
        streams = [(1 + gb[0][:, None, :]) * s + gb[1][:, None, :]
                   for k, s in enumerate(streams)
                   for gb in [self.pref_film(pref.roll(-k, dims=-1)).chunk(2, dim=-1)]]
        for layer in self.layers:
            streams = [layer(s, cos, sin) for s in streams]
        # decoder: per-objective compatibility maps, plus preference-gated cross-objective maps
        # for every ordered pair of streams, fused by a preference-conditioned MLP
        maps = [self.compat(s).permute(1, 2, 3, 0) for s in streams]
        gate = torch.tanh(self.cc_gate(pref)).view(-1, 1, 1, 1)
        for si in range(self.K):
            for sj in range(self.K):
                if si != sj:
                    maps.append(gate * self.compat(streams[si], streams[sj]).permute(1, 2, 3, 0))
        H = torch.cat(maps, dim=-1)
        s1, sh1, s2, sh2 = self.ada(pref).chunk(4, dim=-1)
        h = F.relu((1 + s1[:, None, None, :]) * self.ffa1(H) + sh1[:, None, None, :])
        h = F.relu((1 + s2[:, None, None, :]) * self.ffa2(h) + sh2[:, None, None, :])
        return self.ffa3(h).squeeze(-1)

    def forward(self, coords, solution, last_action, pref, age, do_sample=True,
                fixed_action=None, require_entropy=False, samp_temp=1.0):
        bs, gs, _ = coords.size()
        visited_time = self.visited_time(solution)
        compatibility = torch.tanh(self.score(coords, pref, visited_time, age)) * self.range
        compatibility[self.swap_mask(visited_time)] = -1e20
        if last_action is not None:
            ar = torch.arange(bs)
            compatibility[ar, last_action[:, 0], last_action[:, 1]] = -1e20
            compatibility[ar, last_action[:, 1], last_action[:, 0]] = -1e20
        im = compatibility.view(bs, -1)
        logp = F.log_softmax(im, dim=-1)
        probs = F.softmax(im, dim=-1)
        if fixed_action is not None:
            pair = fixed_action
            pair_index = (pair[:, 0] * gs + pair[:, 1]).view(-1, 1)
        else:
            if do_sample:
                sl = im if samp_temp == 1.0 else im / samp_temp
                pair_index = F.softmax(sl, dim=-1).multinomial(1)
            else:
                pair_index = probs.max(-1)[1].view(-1, 1)
            pair = torch.cat((pair_index // gs, pair_index % gs), -1)
        sel_logp = logp.gather(1, pair_index).squeeze(-1)
        if require_entropy:
            return pair, sel_logp, Categorical(probs, validate_args=False).entropy()
        return pair, sel_logp


def build_model(args, device):
    return SUPL2I(n_obj=args.get("n_obj", 3),
                  embedding_dim=args.get("embedding_dim", 128), hidden_dim=args.get("hidden_dim", 64),
                  n_heads=args.get("n_heads", 4), n_layers=args.get("n_layers", 3),
                  v_range=args.get("v_range", 6.0), rope_nref=args.get("rope_nref", 50),
                  rope_glob=args.get("rope_glob", 4)).to(device)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["args"], device)
    model.load_state_dict(ck["actor"])
    model.eval()
    return model, ck
