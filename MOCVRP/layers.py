import math
import torch
from torch import nn


class Normalization(nn.Module):
    def forward(self, x):
        return (x - x.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(x.var((1, 2)).view(-1, 1, 1) + 1e-5)


# SC-RoPE: rotate each pair of channels of the queries/keys by the node's tour-position angle
def apply_rope(x, cos, sin):
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    kd = x.shape[-1]
    x1, x2 = x[..., :kd // 2], x[..., kd // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class RoPESingleAttn(nn.Module):
    def __init__(self, n_heads, embed_dim):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.kd = embed_dim // n_heads
        self.norm = 1.0 / math.sqrt(self.kd)
        self.Wq = nn.Parameter(torch.Tensor(n_heads, embed_dim, self.kd))
        self.Wk = nn.Parameter(torch.Tensor(n_heads, embed_dim, self.kd))
        self.Wv = nn.Parameter(torch.Tensor(n_heads, embed_dim, self.kd))
        self.Wo = nn.Parameter(torch.Tensor(n_heads, self.kd, embed_dim))
        for p in self.parameters():
            nn.init.uniform_(p, -1.0 / math.sqrt(p.size(-1)), 1.0 / math.sqrt(p.size(-1)))

    def forward(self, x, cos, sin):
        bs, n, _ = x.size()
        xf = x.reshape(-1, self.embed_dim)
        shp = (self.n_heads, bs, n, self.kd)
        Q = torch.matmul(xf, self.Wq).view(shp)
        K = torch.matmul(xf, self.Wk).view(shp)
        V = torch.matmul(xf, self.Wv).view(shp)
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)
        att = torch.softmax(self.norm * torch.matmul(Q, K.transpose(2, 3)), dim=-1)
        h = torch.matmul(att, V)
        return torch.mm(h.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.kd),
                        self.Wo.view(-1, self.embed_dim)).view(bs, n, self.embed_dim)


class RoPESingleLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, ff_hidden):
        super().__init__()
        self.MHA = RoPESingleAttn(n_heads, embed_dim)
        self.Norm_a = Normalization()
        self.FF = nn.Sequential(nn.Linear(embed_dim, ff_hidden), nn.ReLU(inplace=True),
                                nn.Linear(ff_hidden, embed_dim))
        self.Norm_f = Normalization()

    def forward(self, x, cos, sin):
        x = self.Norm_a(x + self.MHA(x, cos, sin))
        x = self.Norm_f(x + self.FF(x))
        return x


# multi-head compatibility: score_h(i, j) = <Q_h x_i, K_h x_j> / sqrt(kd) for every node pair
class MultiHeadCompat(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim):
        super().__init__()
        key_dim = embed_dim // n_heads
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.norm_factor = 1 / math.sqrt(key_dim)
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        for p in self.parameters():
            nn.init.uniform_(p, -1.0 / math.sqrt(p.size(-1)), 1.0 / math.sqrt(p.size(-1)))

    def forward(self, h):
        bs, n, input_dim = h.size()
        hflat = h.contiguous().view(-1, input_dim)
        Q = torch.matmul(hflat, self.W_query).view(self.n_heads, bs, n, -1)
        K = torch.matmul(hflat, self.W_key).view(self.n_heads, bs, n, -1)
        return self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
