"""Shared SBC MCMC sampler (stretch move) + wedge prior. Identical for all arms."""
import math
import numpy as np
import torch


@torch.no_grad()
def batch_loglik(model, theta, target_t, device, bs, use_prior, ti, mi):
    theta = np.asarray(theta, np.float32); target_t = np.asarray(target_t, np.float32)
    out = np.empty(theta.shape[0], np.float32)
    for s in range(0, theta.shape[0], bs):
        e = min(s + bs, theta.shape[0])
        tt = torch.from_numpy(np.repeat(target_t[None, :], e - s, axis=0)).to(device)
        out[s:e] = model.log_prob(torch.from_numpy(theta[s:e]).to(device), tt).cpu().numpy()
    mask = np.any(theta < 0.0, axis=-1) | np.any(theta > 1.0, axis=-1)
    if use_prior and theta.shape[1] > max(ti, mi):
        mask |= (theta[:, mi] > (1.98 - 1.88 * theta[:, ti])) | (theta[:, mi] < (0.90 - 1.88 * theta[:, ti]))
    return np.where(mask, -np.inf, out)


def sample_prior(rng, n, n_params, use_prior, ti, mi):
    got = []
    while sum(len(x) for x in got) < n:
        x = rng.random((max(1000, 5 * n), n_params), dtype=np.float32)
        ok = np.all((x >= 0.0) & (x <= 1.0), axis=1)
        if use_prior and n_params > max(ti, mi):
            ok &= x[:, mi] <= (1.98 - 1.88 * x[:, ti])
            ok &= x[:, mi] >= (0.90 - 1.88 * x[:, ti])
        got.append(x[ok])
    return np.concatenate(got, 0)[:n]


def run_chain(model, target_t, n_params, device, mc, seed):
    rng, pool = np.random.default_rng(seed), mc["walkers"] // 2
    steps, walkers, burnin = mc["steps"], mc["walkers"], mc["burnin"]
    up, ti, mi, bs = mc["use_original_prior"], mc["wedge_tau_idx"], mc["wedge_mmin_idx"], mc["loglik_batch_size"]
    chain = np.zeros((steps, walkers, n_params), np.float32)
    logp = np.zeros((steps, walkers), np.float32)
    chain[0] = sample_prior(rng, walkers, n_params, up, ti, mi)
    logp[0] = batch_loglik(model, chain[0], target_t, device, bs, up, ti, mi)
    sa = math.sqrt(2.0); acc = prop = 0
    for step in range(steps - 1):
        if step % 50 == 0:
            fin = logp[step][np.isfinite(logp[step])]
            print(f"  step {step}/{steps-1} best {np.max(fin) if len(fin) else -np.inf:.3f}", flush=True)
        npos, nlogp = chain[step].copy(), logp[step].copy()
        for sub in range(2):
            lo, hi = pool * sub, pool * (sub + 1)
            partner = rng.integers(pool, size=pool) + pool * ((sub + 1) % 2)
            z = (rng.random(pool) * (sa - 1.0 / sa) + 1.0 / sa) ** 2
            proposal = chain[step, partner] + z[:, None] * (chain[step, lo:hi] - chain[step, partner])
            pl = batch_loglik(model, proposal, target_t, device, bs, up, ti, mi)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                lar = np.nan_to_num((n_params - 1) * np.log(z) + pl - logp[step, lo:hi],
                                    nan=-np.inf, posinf=np.inf, neginf=-np.inf)
            a = np.log(rng.random(pool)) < lar
            acc += int(a.sum()); prop += pool
            npos[lo:hi] = np.where(a[:, None], proposal, chain[step, lo:hi])
            nlogp[lo:hi] = np.where(a, pl, logp[step, lo:hi])
        chain[step + 1], logp[step + 1] = npos, nlogp
    pc, pl = chain[burnin:].reshape(-1, n_params), logp[burnin:].reshape(-1)
    keep = np.isfinite(pl)
    if mc["filter_dlogp"] is not None and np.any(keep):
        keep &= pl > np.max(pl[keep]) - mc["filter_dlogp"]
    return pc[keep].astype(np.float32), pl[keep].astype(np.float32), float(acc / max(prop, 1))
