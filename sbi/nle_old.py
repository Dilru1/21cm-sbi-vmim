"""Stage-2 NLE density models: q(t | theta), pluggable families.

The NLE models the CONDITIONAL LIKELIHOOD of the compressed summary t given
parameters theta (t is the variable, theta is the context) -- note this is
the mirror image of the VMIM head in heads.py, which models q(theta | t).

Families (yaml: nle.model):
    gmm   -- mixture density network, full-covariance Cholesky (the original)
    made  -- single-layer masked autoregressive Gaussian (MADE); the simplest
             autoregressive NLL model, == MAF with one layer
    maf   -- masked affine autoregressive flow (stacked MADE + permutations)
    nsf   -- neural spline flow (autoregressive rational-quadratic), the most
             expressive NLL family here; best default for non-Gaussian q(t|theta)

Every model exposes:
    log_prob(theta, t)  ->  log q(t | theta)   [both standardized-t space]
and is saved/loaded via save_nle()/load_nle() together with a
model_config.json, so stage-3 MCMC can rebuild any family without knowing
which one was trained.

Directory layout (one subdir per family, so arms can be compared):
    {arm}/nle/gmm/   nle_model.pt, model_config.json, t_mean.npy, t_std.npy, loss_history.npy
    {arm}/nle/maf/   ...
    {arm}/nle/nsf/   ...
The compressor artifacts (learned_compressor*.pt, rf_r2_history.npy, ...)
stay in {arm}/nle/ itself, shared by all NLE families -- they are upstream
of this stage and identical across families.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GMM (mixture density network), full covariance via Cholesky
# ---------------------------------------------------------------------------
class GMMConditional(nn.Module):
    """theta -> {weights, means, scale_tril} of a K-component MVN mixture over t.

    param_style:
      'softplus' : diag = softplus(raw) + diag_floor
      'semelin'  : diag = exp(raw) + diag_floor
    !! VERIFY the 'semelin' branch against your previous nle.py GMM before
    !! comparing new-vs-old GMM runs -- the diagonal parameterization must
    !! match exactly or the comparison is confounded.
    """
    def __init__(self, t_dim, n_params, n_mix=4, hidden=64,
                 param_style="softplus", diag_floor=1e-4):
        super().__init__()
        self.t_dim, self.n_params, self.n_mix = t_dim, n_params, n_mix
        self.param_style = param_style
        self.diag_floor = float(diag_floor)
        self.net = nn.Sequential(nn.Linear(n_params, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        n_off = t_dim * (t_dim + 1) // 2
        self.w = nn.Linear(hidden, n_mix)
        self.mu = nn.Linear(hidden, n_mix * t_dim)
        self.tr = nn.Linear(hidden, n_mix * n_off)

    def _mixture(self, theta):
        h = self.net(theta)
        B = theta.shape[0]
        logits = self.w(h)
        mu = self.mu(h).view(B, self.n_mix, self.t_dim)
        L = torch.zeros(B, self.n_mix, self.t_dim, self.t_dim,
                        device=theta.device, dtype=theta.dtype)
        idx = torch.tril_indices(self.t_dim, self.t_dim, 0, device=theta.device)
        L[:, :, idx[0], idx[1]] = self.tr(h).view(B, self.n_mix, -1)
        di = torch.arange(self.t_dim, device=theta.device)
        if self.param_style == "softplus":
            L[:, :, di, di] = F.softplus(L[:, :, di, di]) + self.diag_floor
        else:  # 'semelin'
            L[:, :, di, di] = torch.exp(L[:, :, di, di]) + self.diag_floor
        return logits, mu, L

    def log_prob(self, theta, t):
        logits, mu, L = self._mixture(theta)
        lp = torch.distributions.MultivariateNormal(loc=mu, scale_tril=L) \
                  .log_prob(t[:, None, :])
        return torch.logsumexp(torch.log_softmax(logits, -1) + lp, dim=-1)


# ---------------------------------------------------------------------------
# nflows-based families (MADE / MAF / NSF), context = theta, variable = t
# ---------------------------------------------------------------------------
def _build_flow(kind, t_dim, n_params, hidden=64, n_layers=5,
                num_bins=8, tail_bound=5.0):
    from nflows import transforms, distributions, flows
    from nflows.transforms.autoregressive import (
        MaskedAffineAutoregressiveTransform,
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    )

    tl = []
    if kind == "made":
        # single autoregressive Gaussian layer == conditional MADE
        tl.append(MaskedAffineAutoregressiveTransform(
            features=t_dim, hidden_features=hidden, context_features=n_params,
            num_blocks=2, use_residual_blocks=True, activation=F.relu))
    elif kind == "maf":
        for _ in range(n_layers):
            tl.append(MaskedAffineAutoregressiveTransform(
                features=t_dim, hidden_features=hidden, context_features=n_params,
                num_blocks=2, use_residual_blocks=True, activation=F.relu))
            tl.append(transforms.RandomPermutation(features=t_dim))
    elif kind == "nsf":
        for _ in range(n_layers):
            tl.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=t_dim, hidden_features=hidden, context_features=n_params,
                num_blocks=2, num_bins=num_bins, tails="linear",
                tail_bound=tail_bound, use_residual_blocks=True))
            tl.append(transforms.RandomPermutation(features=t_dim))
    else:
        raise ValueError(kind)
    return flows.Flow(transform=transforms.CompositeTransform(tl),
                      distribution=distributions.StandardNormal(shape=[t_dim]))


class FlowConditional(nn.Module):
    """Wraps an nflows Flow into the shared log_prob(theta, t) interface."""
    def __init__(self, kind, t_dim, n_params, hidden=64, n_layers=5,
                 num_bins=8, tail_bound=5.0):
        super().__init__()
        self.kind = kind
        self.flow = _build_flow(kind, t_dim, n_params, hidden, n_layers,
                                num_bins, tail_bound)

    def log_prob(self, theta, t):
        return self.flow.log_prob(inputs=t, context=theta)


# ---------------------------------------------------------------------------
# factory + persistence
# ---------------------------------------------------------------------------
def build_nle(nc, t_dim, n_params):
    """nc = cfg['nle'] block. Keys (all optional except model for non-gmm):
        model:       gmm | made | maf | nsf     (default gmm)
        n_mix:       4        (gmm)
        hidden:      64
        n_layers:    5        (maf/nsf)
        num_bins:    8        (nsf)
        tail_bound:  5.0      (nsf; t is standardized, so 5 sigma)
        param_style: softplus | semelin   (gmm)
        diag_floor:  1e-4                 (gmm)
    """
    kind = nc.get("model", "gmm").lower()
    if kind == "gmm":
        return kind, GMMConditional(t_dim, n_params,
                                    n_mix=int(nc.get("n_mix", 4)),
                                    hidden=int(nc.get("hidden", 64)),
                                    param_style=nc.get("param_style", "softplus"),
                                    diag_floor=float(nc.get("diag_floor", 1e-4)))
    return kind, FlowConditional(kind, t_dim, n_params,
                                 hidden=int(nc.get("hidden", 64)),
                                 n_layers=int(nc.get("n_layers", 5)),
                                 num_bins=int(nc.get("num_bins", 8)),
                                 tail_bound=float(nc.get("tail_bound", 5.0)))


def nle_subdir(nle_root, nc):
    """{arm}/nle/<scope>/<model-kind>/ where scope is 'standard_t' or 'raw_t'.

    One subdir per (t-scaling, NLE family) combination, so a raw-vs-
    standardized ablation across several families all live side by side and
    never overwrite each other.
    """
    scope = "standard_t" if bool(nc.get("standardize", True)) else "raw_t"
    d = os.path.join(str(nle_root), scope, nc.get("model", "gmm").lower())
    # ensemble replicas: .../<model>/seed_<init_seed>/  (opt-in; when
    # nle.seed_subdir is false the path is byte-identical to the old layout,
    # so existing runs and stage-3 chains keep working untouched)
    if nc.get("seed_subdir", False):
        init_seed = int(nc.get("init_seed", nc.get("seed", 42)))
        d = os.path.join(d, f"seed_{init_seed}")
    os.makedirs(d, exist_ok=True)
    return d


def save_nle(model, kind, subdir, nc, t_dim, n_params, t_mean, t_std):
    torch.save(model.state_dict(), os.path.join(subdir, "nle_model.pt"))
    np.save(os.path.join(subdir, "t_mean.npy"), t_mean)
    np.save(os.path.join(subdir, "t_std.npy"), t_std)
    cfg = {"model": kind, "t_dim": t_dim, "n_params": n_params,
           "n_mix": int(nc.get("n_mix", 4)), "hidden": int(nc.get("hidden", 64)),
           "n_layers": int(nc.get("n_layers", 5)), "num_bins": int(nc.get("num_bins", 8)),
           "tail_bound": float(nc.get("tail_bound", 5.0)),
           "param_style": nc.get("param_style", "softplus"),
           "diag_floor": float(nc.get("diag_floor", 1e-4))}
    with open(os.path.join(subdir, "model_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def load_nle(subdir, device="cpu"):
    """Rebuild any trained NLE family from its subdir. For stage-3 MCMC:
        model, t_mean, t_std = load_nle('{arm}/nle/nsf', device)
        loglike = model.log_prob(theta_batch, (t_obs - t_mean) / t_std)
    """
    with open(os.path.join(subdir, "model_config.json")) as f:
        cfg = json.load(f)
    _, model = build_nle(cfg, cfg["t_dim"], cfg["n_params"])
    model.load_state_dict(torch.load(os.path.join(subdir, "nle_model.pt"),
                                     map_location=device))
    model.to(device).eval()
    t_mean = np.load(os.path.join(subdir, "t_mean.npy"))
    t_std = np.load(os.path.join(subdir, "t_std.npy"))
    return model, t_mean, t_std