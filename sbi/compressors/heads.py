"""VMIM posterior heads, shared by both compressor arms.

CHANGES vs previous version
---------------------------
1. sigma_floor (default 1e-2, was 1e-4): floor on the GMM scale_tril diagonal,
   in NORMALIZED theta units. Caps the nats extractable from the strongly
   constrained parameters (fX, Mmin) so the residual gradient shifts to the
   weak one (rHS) early, instead of after an ~80-epoch plateau.
2. ThetaPreprocessor: dequantization jitter + affine margin for gridded /
   endpoint parameters now live INSIDE the head's log_prob, train-time only
   (jitter keys off self.training). The dataset exports PRISTINE theta, so
   NLE/MCMC/SBC/GV coordinates are untouched and comparable across arms.
3. conditional_sigma(t): per-parameter posterior std of the mixture, the
   cheap collapse diagnostic (sigma(theta_i|t) -> prior sigma == collapse).
4. BUGFIX: _L previously ignored self.sigma_floor entirely and applied a
   hardcoded 1e-2 regardless of what was passed in -- harmless only because
   every run so far happened to set sigma_floor: 1.0e-2 in yaml. Fixed, and
   sigma_floor is now a per-parameter buffer (scalar still works, broadcasts
   to all params) so e.g. fX/Mmin can be capped harder than tau/rHS without
   over-regularizing the whole vector.
5. build_head_from_cfg: generalized dequant from theta2-only to any gridded
   parameter via a generic `dequant: {idx: {half: ...}}` yaml block, kept
   backward compatible with the legacy theta2_dequant/theta2_grid_half/
   theta2_margin keys.
6. Jitter RNG isolation: the dequantization noise is now drawn from a dedicated
   torch.Generator (seeded by jitter_seed, tied to init_seed by default) rather
   than the global stream. Previously torch.rand_like consumed the global RNG,
   so enabling jitter re-phased every subsequent dropout mask -- making the
   jitter-vs-no-jitter arms two different random trajectories instead of a
   controlled A/B, and producing a seed-dependent "on/off" appearance of the
   jitter effect (most visible at the sharp rHS phase transition). With the
   private generator both arms consume the global RNG identically and differ
   ONLY by the injected theta noise. Pair with set_all_seeds(deterministic=True)
   in stage1 to also remove cuDNN benchmark nondeterminism.

The head is discarded after training -- only t is exported -- so everything
here is internal to compressor training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows import transforms, distributions, flows


# ---------------------------------------------------------------------------
# theta pre-processing (train-time dequantization + endpoint margin)
# ---------------------------------------------------------------------------
class ThetaPreprocessor(nn.Module):
    """Applied to theta inside log_prob, BEFORE any logit transform.

    jitter : {dim: half_spacing}  uniform dequantization noise, TRAIN ONLY.
             For rHS on a 0.1 grid -> {2: 0.05}.
    margin : {dim: (a, b)}        affine squeeze theta -> a + b*theta, applied
             in train AND eval so the head always sees one coordinate system.
             (0.07, 0.86) maps the jittered range [-0.05, 1.05] into
             (0.027, 0.973), safely away from the logit clamp.

    The affine Jacobian is a constant (-log b per dim); it shifts the NLL by a
    constant and is irrelevant to gradients, so it is not added.
    """
    def __init__(self, jitter=None, margin=None, jitter_seed=0):
        super().__init__()
        self.jitter = dict(jitter or {})
        self.margin = dict(margin or {})
        # Dedicated RNG for the dequantization jitter, kept OFF the global torch
        # stream so that toggling jitter no longer re-phases dropout (or any
        # other global-RNG consumer). This makes jitter-vs-no-jitter a clean
        # A/B: both arms draw identically from the global generator and differ
        # ONLY by the theta noise added here. One generator is built lazily per
        # device (keyed by str(device)) and seeded ONCE; it then advances
        # deterministically across steps for a given jitter_seed.
        # NOTE: the generator state is not checkpointed, so a mid-run resume
        # will not reproduce the exact jitter stream -- fine for start-to-end
        # training runs, which is how these arms are launched.
        self.jitter_seed = int(jitter_seed)
        self._gens = {}

    def set_jitter_seed(self, seed):
        """Update the jitter seed and invalidate cached generators."""
        self.jitter_seed = int(seed)
        self._gens = {}

    def _generator(self, device):
        key = str(device)
        g = self._gens.get(key)
        if g is None:
            g = torch.Generator(device=device)
            g.manual_seed(self.jitter_seed)
            self._gens[key] = g
        return g

    def forward(self, theta, training):
        if not self.jitter and not self.margin:
            return theta
        theta = theta.clone()
        if training and self.jitter:
            g = self._generator(theta.device)
            for d, half in self.jitter.items():
                col = theta[:, d]
                u = torch.rand(col.shape, generator=g,
                               device=col.device, dtype=col.dtype)
                theta[:, d] = col + (u * 2.0 - 1.0) * half
        for d, (a, b) in self.margin.items():
            theta[:, d] = a + b * theta[:, d]
        return theta


def _logit(theta, eps=1e-4):
    """[0,1] -> R with its log|dy/dtheta| (change of variables for log_prob)."""
    theta = theta.clamp(eps, 1.0 - eps)
    y = torch.log(theta) - torch.log1p(-theta)
    log_jac = (-torch.log(theta) - torch.log1p(-theta)).sum(-1)
    return y, log_jac


class FlowContextWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, context):
        return self.net(torch.cat([inputs, context], dim=-1))


# ---------------------------------------------------------------------------
# GMM heads
# ---------------------------------------------------------------------------
class _Base(nn.Module):
    def __init__(self, t_dim, n_params, n_mix, hidden, sigma_floor=1e-2, pre=None,
                 jitter_seed=None):
        super().__init__()
        self.t_dim, self.n_params, self.n_mix = t_dim, n_params, n_mix
        # sigma_floor: scalar (same floor for every parameter) OR a
        # length-n_params sequence (per-parameter floor -- e.g. cap fX/Mmin
        # harder than tau/rHS so their nats can't crowd a delicate param's
        # gradient out of the loss). Registered as a buffer so .to(device)
        # moves it automatically along with the rest of the module.
        if isinstance(sigma_floor, (list, tuple)):
            assert len(sigma_floor) == n_params, \
                f"sigma_floor list must have length n_params={n_params}, got {len(sigma_floor)}"
            floor_t = torch.tensor(sigma_floor, dtype=torch.float32)
        else:
            floor_t = torch.full((n_params,), float(sigma_floor), dtype=torch.float32)
        self.register_buffer("sigma_floor", floor_t)
        self.pre = pre if pre is not None else ThetaPreprocessor()
        # Thread the jitter seed explicitly through the head too, so it is the
        # single source of truth even when a pre-built preprocessor is passed
        # in. None => leave whatever seed the preprocessor already carries.
        if jitter_seed is not None:
            self.pre.set_jitter_seed(jitter_seed)
        self.net = nn.Sequential(nn.Linear(t_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.logits = nn.Linear(hidden, n_mix)
        self.means = nn.Linear(hidden, n_mix * n_params)
        self.raw_tril = nn.Linear(hidden, n_mix * (n_params * (n_params + 1) // 2))

    def forward(self, t):
        h = self.net(t)
        return self.logits(h), self.means(h).view(-1, self.n_mix, self.n_params), self.raw_tril(h)

    def _L(self, tril, B, device, dtype=None):
        dtype = tril.dtype if dtype is None else dtype
        L = torch.zeros(B, self.n_mix, self.n_params, self.n_params, device=device, dtype=dtype)
        idx = torch.tril_indices(self.n_params, self.n_params, 0, device=device)
        L[:, :, idx[0], idx[1]] = tril.view(B, self.n_mix, -1)
        di = torch.arange(self.n_params, device=device)
        floor = self.sigma_floor.to(device=device, dtype=dtype)   # (n_params,) -- broadcasts
        L[:, :, di, di] = F.softplus(L[:, :, di, di]) + floor
        return L

    @torch.no_grad()
    def conditional_sigma(self, t):
        """Per-parameter std of the mixture q(theta|t): sqrt(diag of the mixture
        covariance). Shape (B, n_params). The collapse diagnostic."""
        logits, means, tril = self.forward(t)
        w = torch.softmax(logits, dim=-1)                              # (B, K)
        L = self._L(tril, t.shape[0], t.device, t.dtype)               # (B, K, P, P)
        comp_var = (L ** 2).sum(-1)                                    # diag of L L^T  (B, K, P)
        mu = (w.unsqueeze(-1) * means).sum(1)                          # (B, P)
        second = (w.unsqueeze(-1) * (comp_var + means ** 2)).sum(1)    # E[theta^2]
        return (second - mu ** 2).clamp_min(1e-12).sqrt()


class VMIMGMMPosteriorHeadOLD(_Base):
    """GMM over (pre-processed) theta, no logit."""
    def log_prob(self, t, theta):
        theta = self.pre(theta, self.training)
        logits, means, tril = self.forward(t)
        L = self._L(tril, t.shape[0], t.device, t.dtype)
        lp = torch.distributions.MultivariateNormal(loc=means, scale_tril=L).log_prob(theta[:, None, :])
        return torch.logsumexp(torch.log_softmax(logits, dim=-1) + lp, dim=-1)


class VMIMGMMPosteriorHeadNEW(_Base):
    """GMM over logit(pre-processed theta)."""
    def log_prob(self, t, theta):
        theta = self.pre(theta, self.training)
        y, log_jac = _logit(theta)
        logits, means, tril = self.forward(t)
        L = self._L(tril, t.shape[0], t.device, t.dtype)
        lp = torch.distributions.MultivariateNormal(loc=means, scale_tril=L).log_prob(y[:, None, :])
        return torch.logsumexp(torch.log_softmax(logits, dim=-1) + lp, dim=-1) + log_jac


# ---------------------------------------------------------------------------
# Flow heads (kept for completeness; GMM-2 is the current best q)
# ---------------------------------------------------------------------------
class VMIMNFPosteriorHead(nn.Module):
    def __init__(self, t_dim, n_params, hidden=128, num_layers=4, num_bins=8,
                 tail_bound=4.0, logit=True, pre=None):
        super().__init__()
        from nflows.utils import create_alternating_binary_mask
        self.t_dim, self.n_params, self.logit = t_dim, n_params, logit
        self.pre = pre if pre is not None else ThetaPreprocessor()

        base_dist = distributions.StandardNormal(shape=[n_params])
        transform_list = []
        for i in range(num_layers):
            mask = create_alternating_binary_mask(features=n_params, even=(i % 2 == 0))

            def make_transform_net(in_features, out_features):
                net = nn.Sequential(
                    nn.Linear(in_features + t_dim, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden), nn.Tanh(),
                    nn.Linear(hidden, out_features),
                )
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)
                return FlowContextWrapper(net)

            transform_list.append(
                transforms.PiecewiseRationalQuadraticCouplingTransform(
                    mask=mask, transform_net_create_fn=make_transform_net,
                    num_bins=num_bins, tails="linear", tail_bound=tail_bound,
                    apply_unconditional_transform=False))
            transform_list.append(transforms.RandomPermutation(features=n_params))

        self.flow = flows.Flow(transform=transforms.CompositeTransform(transform_list),
                               distribution=base_dist)

    def log_prob(self, t, theta):
        theta = self.pre(theta, self.training)
        if self.logit:
            y, log_jac = _logit(theta)
            return self.flow.log_prob(inputs=y, context=t) + log_jac
        return self.flow.log_prob(inputs=theta, context=t)


class VMIMMAFPosteriorHead(nn.Module):
    def __init__(self, t_dim, n_params, hidden=128, num_layers=4, num_blocks=2,
                 tail_bound=None, logit=True, pre=None):
        super().__init__()
        from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
        self.t_dim, self.n_params, self.logit = t_dim, n_params, logit
        self.pre = pre if pre is not None else ThetaPreprocessor()

        base_dist = distributions.StandardNormal(shape=[n_params])
        transform_list = []
        for _ in range(num_layers):
            transform_list.append(
                MaskedAffineAutoregressiveTransform(
                    features=n_params, hidden_features=hidden, context_features=t_dim,
                    num_blocks=num_blocks, use_residual_blocks=True,
                    random_mask=False, activation=F.relu))
            transform_list.append(transforms.RandomPermutation(features=n_params))

        self.flow = flows.Flow(transform=transforms.CompositeTransform(transform_list),
                               distribution=base_dist)

    def log_prob(self, t, theta):
        theta = self.pre(theta, self.training)
        if self.logit:
            y, log_jac = _logit(theta)
            return self.flow.log_prob(inputs=y, context=t) + log_jac
        return self.flow.log_prob(inputs=theta, context=t)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def build_head(kind, t_dim, n_params, n_mix, hidden,
               sigma_floor=1e-2, jitter=None, margin=None, jitter_seed=0):
    """jitter/margin are {param_index: value} dicts, e.g.
       jitter={2: 0.05}, margin={2: (0.07, 0.86)} for the gridded rHS.
       sigma_floor may be a scalar (all params) or a length-n_params list.
       jitter_seed seeds the dedicated (global-stream-isolated) jitter RNG;
       tie it to the compressor's init_seed so ensemble members still differ."""
    pre = ThetaPreprocessor(jitter=jitter, margin=margin, jitter_seed=jitter_seed)
    if kind == "gmm_old":
        return VMIMGMMPosteriorHeadOLD(t_dim, n_params, n_mix, hidden,
                                       sigma_floor=sigma_floor, pre=pre,
                                       jitter_seed=jitter_seed)
    if kind == "gmm_new":
        return VMIMGMMPosteriorHeadNEW(t_dim, n_params, n_mix, hidden,
                                       sigma_floor=sigma_floor, pre=pre,
                                       jitter_seed=jitter_seed)
    if kind == "nf_spline":
        return VMIMNFPosteriorHead(t_dim=t_dim, n_params=n_params, hidden=hidden,
                                   num_layers=max(2, n_mix), num_bins=8, pre=pre)
    if kind == "nf_maf":
        return VMIMMAFPosteriorHead(t_dim=t_dim, n_params=n_params, hidden=hidden,
                                    num_layers=max(2, n_mix), num_blocks=2, pre=pre)
    raise ValueError(kind)


def build_head_from_cfg(c, cfg):
    """Convenience: read the new knobs straight from the compressor block.

    yaml keys (all optional):
      sigma_floor:       1.0e-2                  # scalar (all params) OR
      sigma_floor:       [0.02, 0.008, 0.008, 0.02]   # per-param [fX,tau,rHS,Mmin]
                                                  # -- cap the EASY params harder
                                                  # so their nats can't crowd out
                                                  # a delicate one (e.g. tau).

      # legacy, index-2-only (still works):
      theta2_dequant:    true
      theta2_grid_half:  0.1
      theta2_margin:     false

      # generic, any gridded param (run grid_check.py to get real per-param
      # half-spacings -- don't guess):
      dequant:
        0: {half: 0.0394}    # fX,   13 grid points
        1: {half: 0.0443}    # tau,  12 grid points
        2: {half: 0.1000}    # rHS,   6 grid points (same as theta2_grid_half)
        3: {half: 0.0813}    # Mmin,  7 grid points
    """
    jitter, margin = {}, {}

    # legacy path (index 2 only)
    if c.get("theta2_dequant", False):
        jitter[2] = float(c.get("theta2_grid_half", 0.05))
    if c.get("theta2_margin", False):
        h = jitter.get(2, 0.0)
        b = 0.96 / (1.0 + 2.0 * h)
        a = 0.02 + h * b
        margin[2] = (a, b)

    # generic path (any index) -- entries here override the legacy path
    # above if both happen to set the same index.
    for idx, spec in c.get("dequant", {}).items():
        idx = int(idx)
        h = float(spec.get("half", 0.0))
        if h > 0:
            jitter[idx] = h
        if spec.get("margin", False):
            b = 0.96 / (1.0 + 2.0 * h)
            a = 0.02 + h * b
            margin[idx] = (a, b)

    raw_floor = c.get("sigma_floor", 1e-2)
    sigma_floor = list(raw_floor) if isinstance(raw_floor, (list, tuple)) else float(raw_floor)

    # Seed for the isolated jitter RNG. Priority: explicit compressor.jitter_seed
    # > init_seed > legacy seed > 0. Tying it to init_seed by default means the
    # jitter realization varies per ensemble member (like the other init-seeded
    # randomness) while staying reproducible for a fixed member.
    jitter_seed = int(c.get("jitter_seed", c.get("init_seed", c.get("seed", 0))))

    return build_head(c["vmim_head"], cfg["t_dim"], cfg["n_params"],
                      c["n_mix_head"], c["hidden_head"],
                      sigma_floor=sigma_floor,
                      jitter=jitter or None, margin=margin or None,
                      jitter_seed=jitter_seed)