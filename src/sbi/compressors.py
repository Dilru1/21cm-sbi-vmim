"""Stage-1 networks: MLP on summary vectors (Path A), 3D CNN on field cubes (Path B),
and the Gaussian-mixture head q(theta | t) used by the VMIM objective.

Both compressors return (t, aux): t is the exported summary; aux is a parameter
regression head. With direct=True (MSE objective) t itself is the regression
output and the network returns (t, t).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPCompressor(nn.Module):
    def __init__(self, n_input, t_dim, n_params, dense_layers, dropout=0.0, direct=False):
        super().__init__()
        if direct and t_dim != n_params:
            raise ValueError("direct regression needs t_dim == n_params")
        self.direct = direct
        layers, prev = [], n_input
        for h in dense_layers:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.LeakyReLU(0.1), nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.fc_summary = nn.Linear(prev, t_dim)
        self.fc_aux = nn.Linear(prev, n_params)

    def forward(self, x):
        f = self.body(x)
        t = self.fc_summary(f)
        return (t, t) if self.direct else (t, self.fc_aux(f))


# ---------------------------------------------------------------- 3D CNN
class SEBlock3d(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(hidden, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1, 1).expand_as(x)


class ResidualBlock3d(nn.Module):
    def __init__(self, ci, co, stride=1, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv3d(ci, co, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(min(8, co // 4), co)
        self.conv2 = nn.Conv3d(co, co, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(min(8, co // 4), co)
        self.se = SEBlock3d(co, reduction)
        self.shortcut = nn.Sequential()
        if stride != 1 or ci != co:
            self.shortcut = nn.Sequential(nn.Conv3d(ci, co, 1, stride=stride, bias=False),
                                          nn.GroupNorm(min(8, co // 4), co))

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.se(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        return F.leaky_relu(out, 0.1)


class CNNCompressor(nn.Module):
    """Stem (32 ch) + 5 residual SE blocks (32, 64, 128, 256, 512; strides 1, 1, 2, 2, 2).
    Readout: per-channel spatial mean and std of blocks 2-5, then MLP heads."""
    CHANNELS = [32, 32, 64, 128, 256, 512]
    STRIDES = [1, 1, 2, 2, 2]
    READOUT = [2, 3, 4, 5]

    def __init__(self, in_channels, t_dim, n_params, direct=False):
        super().__init__()
        if direct and t_dim != n_params:
            raise ValueError("direct regression needs t_dim == n_params")
        self.direct = direct
        ch = self.CHANNELS
        self.stem = nn.Sequential(nn.Conv3d(in_channels, ch[0], 3, padding=1, bias=False),
                                  nn.GroupNorm(4, ch[0]), nn.LeakyReLU(0.1))
        self.blocks = []
        for i, s in enumerate(self.STRIDES):
            blk = ResidualBlock3d(ch[i], ch[i + 1], stride=s)
            setattr(self, f"layer{i + 1}", blk)
            self.blocks.append(blk)
        n_feat = 2 * sum(ch[s] for s in self.READOUT)
        self.fc_summary = nn.Sequential(
            nn.Linear(n_feat, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1), nn.Dropout(0.05),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.LeakyReLU(0.1), nn.Dropout(0.05),
            nn.Linear(128, t_dim))
        self.fc_aux = None if direct else nn.Sequential(
            nn.Linear(n_feat, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(128, n_params))

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        f = torch.cat([torch.cat([o.flatten(2).mean(2), o.flatten(2).std(2)], 1)
                       for o in (outs[s - 1] for s in self.READOUT)], 1)
        t = self.fc_summary(f)
        return (t, t) if self.direct else (t, self.fc_aux(f))


# ---------------------------------------------------------------- VMIM head
class GMMHead(nn.Module):
    """q(theta | t): K-component full-covariance Gaussian mixture on [0,1]-normalised theta.
    The Cholesky diagonal is softplus(.) + sigma_floor."""

    def __init__(self, t_dim, n_params, n_mix, hidden, sigma_floor):
        super().__init__()
        self.n_params, self.n_mix = n_params, n_mix
        self.register_buffer("sigma_floor", torch.full((n_params,), float(sigma_floor)))
        self.net = nn.Sequential(nn.Linear(t_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.logits = nn.Linear(hidden, n_mix)
        self.means = nn.Linear(hidden, n_mix * n_params)
        self.raw_tril = nn.Linear(hidden, n_mix * (n_params * (n_params + 1) // 2))

    def _params(self, t):
        h = self.net(t)
        B, K, P = t.shape[0], self.n_mix, self.n_params
        # keep this evaluation order (logits, means, tril): it fixes the order in which
        # gradients are summed into h, and therefore the exact floating-point result
        logits, means, tril = self.logits(h), self.means(h).view(B, K, P), self.raw_tril(h)
        L = torch.zeros(B, K, P, P, device=t.device, dtype=t.dtype)
        idx = torch.tril_indices(P, P, 0, device=t.device)
        L[:, :, idx[0], idx[1]] = tril.view(B, K, -1)
        di = torch.arange(P, device=t.device)
        L[:, :, di, di] = F.softplus(L[:, :, di, di]) + self.sigma_floor.to(t.dtype)
        return logits, means, L

    def log_prob(self, t, theta):
        logits, means, L = self._params(t)
        lp = torch.distributions.MultivariateNormal(loc=means, scale_tril=L).log_prob(theta[:, None, :])
        return torch.logsumexp(torch.log_softmax(logits, -1) + lp, -1)

    @torch.no_grad()
    def conditional_sigma(self, t):
        """Per-parameter std of the mixture (training diagnostic)."""
        logits, means, L = self._params(t)
        w = torch.softmax(logits, -1).unsqueeze(-1)
        mu = (w * means).sum(1)
        second = (w * ((L ** 2).sum(-1) + means ** 2)).sum(1)
        return (second - mu ** 2).clamp_min(1e-12).sqrt()
