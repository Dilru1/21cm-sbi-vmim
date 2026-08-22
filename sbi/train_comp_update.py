"""Compressor training engine (patched again -- see CHANGES 6).

CHANGES vs previous version
---------------------------
1. Two-time-scale optimizer: head LR = lr * head_lr_mult (default 5) so q
   tracks t closely and the compressor can't silently drop information.
2. Plateau-safe LR schedule: ReduceLROnPlateau patience raised (default 15),
   factor 0.5, min_lr = lr/100. The pre-transition NLL plateau must be
   SURVIVED, not punished.
3. Early stopping keyed to the RF-probe mean R2 (probe_patience, counted in
   probe events), NOT to the loss. Loss-patience is disabled during VMIM;
   a plateaued loss is exactly the state that precedes the rHS transition.
4. best_val reset at every objective change (MSE and NLL units are not
   comparable), scheduler state rebuilt at the switch.
5. Per-parameter conditional sigma from the GMM head logged at every probe
   -> nle/sigma_history.npy, printed. sigma(theta_i|t) drifting to the prior
   sigma (~0.29 for uniform [0,1]) == collapse; dropping == wakeup.
6. BUGFIX (this version): the probe early-stop counter used to be armed by
   `not aux_only`, which is True for the ENTIRE hard-switch/anneal run --
   including the pure-MSE warmup window where VMIM hasn't trained a single
   step yet. A run could (and did) get "early stopped" during warmup, before
   VMIM ever switched on, because MSE-phase R2 is noisy and can look stale.
   Fix: the counter is now armed ONLY once aux_coef == 0.0 (pure VMIM, the
   final "VMIM" tag -- i.e. past both warmup AND any anneal window), and is
   reset to 0 at every phase boundary (same place best_val already resets).
   rf_r2_history.npy now also logs the phase/aux_coef so post-hoc plots can
   tell which points came from warmup vs anneal vs pure VMIM.

yaml knobs read from the compressor block (all optional, defaults shown):
    head_lr_mult:     5.0
    plateau_patience: 15
    probe_patience:   6        # probe events, counted ONLY once pure VMIM
    grad_clip:        1.0
"""

import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from packaging import version


# -- AMP ---------------------------------------------------------------------
def make_amp_tools(device):
    use_amp = device.type == "cuda"
    tv = version.parse(torch.__version__.split("+")[0])
    if use_amp and tv >= version.parse("2.0"):
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        def ctx():
            return torch.amp.autocast("cuda", enabled=True)
    elif use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        def ctx():
            return torch.cuda.amp.autocast(enabled=True)
    else:
        scaler = None

        def ctx():
            return nullcontext()

    noamp = (
        (lambda: torch.autocast(device.type, enabled=False)) if use_amp else (lambda: nullcontext())
    )
    return use_amp, scaler, ctx, noamp


# -- training-mode schedule ----------------------------------------------------
def _aux_weight(epoch, warmup, anneal, anneal_epochs, lam_aux):
    if not anneal:
        if epoch < warmup:
            return lam_aux, False, "AUX"
        return 0.0, True, "VMIM"
    if epoch < warmup:
        return lam_aux, True, "AUX+VMIM"
    if epoch < warmup + anneal_epochs:
        frac = 1.0 - (epoch - warmup) / max(1, anneal_epochs)
        return lam_aux * frac, True, f"ANNEAL{frac:.2f}"
    return 0.0, True, "VMIM"


# -- RF-R2 probe ---------------------------------------------------------------
@torch.no_grad()
def probe_r2(compressor, loader, device, n_params=4, max_batches=400):
    """max_batches raised: val loader batch=8 -> 400 batches ~ 3200 samples.
    120-test-point R2 for a weak parameter is too noisy to select checkpoints."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    compressor.eval()
    raw = compressor.module if isinstance(compressor, nn.DataParallel) else compressor
    Ts, THs = [], []
    for b, (data, theta, *_) in enumerate(loader):
        if b >= max_batches:
            break
        out = raw(data.to(device))
        t = out[0] if isinstance(out, tuple) else out
        Ts.append(t.cpu().numpy())
        THs.append(theta[:, :n_params].numpy())

    T, TH = np.concatenate(Ts), np.concatenate(THs)
    Xtr, Xte, ytr, yte = train_test_split(T, TH, test_size=0.3, random_state=0)
    rf_r2 = []
    for p in range(n_params):
        rf = RandomForestRegressor(100, min_samples_leaf=2, n_jobs=-1, random_state=0)
        rf.fit(Xtr, ytr[:, p])
        rf_r2.append(float(r2_score(yte[:, p], rf.predict(Xte))))
    print("[RF R2]  " + " | ".join(f"t{p}:{rf_r2[p]:.3f}" for p in range(n_params)), flush=True)
    return rf_r2


@torch.no_grad()
def probe_sigma(compressor, head, loader, device, n_params=4, max_batches=40):
    """Mean per-parameter conditional sigma of q(theta|t) over val batches.
    Only available for GMM heads (conditional_sigma method)."""
    if not hasattr(head, "conditional_sigma"):
        return None
    compressor.eval()
    head.eval()
    raw = compressor.module if isinstance(compressor, nn.DataParallel) else compressor
    sig = []
    for b, (data, _, *_) in enumerate(loader):
        if b >= max_batches:
            break
        out = raw(data.to(device))
        t = (out[0] if isinstance(out, tuple) else out).float()
        sig.append(head.conditional_sigma(t).cpu().numpy())
    s = np.concatenate(sig).mean(0)
    print(
        "[q sigma] "
        + " | ".join(f"t{p}:{s[p]:.3f}" for p in range(len(s)))
        + "   (prior sigma ~0.29; floor = head.sigma_floor)",
        flush=True,
    )
    return s.tolist()


# -- main ----------------------------------------------------------------------
def run_training(compressor, head, train_loader, val_loader, device, nle_dir, c, n_params, run):
    raw = compressor.module if isinstance(compressor, nn.DataParallel) else compressor

    aux_only = bool(c.get("aux_only", True))
    anneal = bool(c.get("aux_anneal", False))
    warmup = int(c.get("warmup_epochs", 0))
    anneal_eps = int(c.get("aux_anneal_epochs", warmup))
    lam_aux = float(c.get("lam_aux", 1.0))
    lr = float(c.get("lr", 1e-3))
    head_mult = float(c.get("head_lr_mult", 5.0))
    epochs = int(c.get("epochs", 60))
    probe_every = int(c.get("probe_every", 10))
    probe_pat = int(c.get("probe_patience", 6))
    plateau_pat = int(c.get("plateau_patience", 15))
    max_norm = float(c.get("grad_clip", 1.0))

    if aux_only:
        print("[MODE] aux_only: pure MSE regression, no VMIM.", flush=True)
    elif warmup > 0 and not anneal:
        print(f"[MODE] hard switch: MSE for {warmup} epochs, then pure VMIM.", flush=True)
    elif anneal:
        print(
            f"[MODE] anneal: MSE+VMIM, MSE ramps down over epochs {warmup}-{warmup + anneal_eps}.",
            flush=True,
        )
    else:
        print("[MODE] pure VMIM from epoch 0.", flush=True)

    criterion = nn.MSELoss()

    groups = [{"params": list(compressor.parameters()), "lr": lr}]
    if head is not None and not aux_only:
        groups.append({"params": list(head.parameters()), "lr": lr * head_mult})
    optimizer = optim.Adam(groups)
    all_params = [p for g in groups for p in g["params"]]

    def fresh_scheduler():
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=plateau_pat, min_lr=lr * 0.01
        )

    scheduler = fresh_scheduler()

    use_amp, scaler, autocast_ctx, noamp_ctx = make_amp_tools(device)

    best_val, best_probe = float("inf"), -1e9
    probes_since_best = 0
    rf_log, sigma_log, loss_log = [], [], []
    prev_phase = None
    # BUGFIX 6: the early-stop counter only ever counts once we're in the
    # FINAL, pure-VMIM regime (aux_coef == 0.0). MSE-dominated warmup/anneal
    # probes are still logged (for plotting) but can never trigger a stop.
    was_pure_vmim = False

    for epoch in range(epochs):
        if aux_only:
            aux_coef, vmim_on, tag = lam_aux, False, "AUX_ONLY"
        else:
            aux_coef, vmim_on, tag = _aux_weight(epoch, warmup, anneal, anneal_eps, lam_aux)

        phase = 0 if (aux_coef > 0.0 or aux_only) else 1
        if prev_phase is not None and phase != prev_phase:
            print(
                f"[PHASE] objective changed at epoch {epoch + 1}: "
                "resetting best_val, LR scheduler, and probe-stop counter.",
                flush=True,
            )
            best_val = float("inf")
            scheduler = fresh_scheduler()
        prev_phase = phase

        is_pure_vmim = (not aux_only) and (aux_coef == 0.0) and vmim_on
        if is_pure_vmim and not was_pure_vmim:
            # just entered the pure-VMIM regime this epoch: arm the counter fresh
            print(
                f"[PHASE] pure VMIM begins at epoch {epoch + 1}: "
                "arming probe-based early stopping from here.",
                flush=True,
            )
            probes_since_best = 0
            best_probe = -1e9
        was_pure_vmim = is_pure_vmim

        # -- train ------------------------------------------------------------
        compressor.train()
        if head is not None:
            head.train()
        train_sum, n_train = 0.0, 0
        for data, target, *_ in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                out = compressor(data)
                t = (out[0] if isinstance(out, tuple) else out).float()
                loss = 0.0
                if aux_coef > 0.0 or aux_only:
                    pred = out[1].float() if isinstance(out, tuple) else t
                    loss = loss + aux_coef * criterion(pred[:, :n_params], target[:, :n_params])
                if vmim_on and head is not None:
                    with noamp_ctx():
                        loss = loss + (
                            -head.log_prob(t.float(), target[:, :n_params].float()).mean()
                        )
            if not torch.is_tensor(loss) or not torch.isfinite(loss):
                continue
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm)
                optimizer.step()
            train_sum += loss.item()
            n_train += 1
        avg_train = train_sum / max(n_train, 1)

        # -- val ----------------------------------------------------------------
        compressor.eval()
        if head is not None:
            head.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for data, target, *_ in val_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True).float()
                with autocast_ctx():
                    out = compressor(data)
                    t = (out[0] if isinstance(out, tuple) else out).float()
                    if aux_coef > 0.0 or aux_only:
                        pred = out[1].float() if isinstance(out, tuple) else t
                        l = criterion(pred[:, :n_params], target[:, :n_params])
                    else:
                        if head is not None:
                            with noamp_ctx():
                                l = -head.log_prob(t.float(), target[:, :n_params].float()).mean()
                        else:
                            l = criterion(t[:, :n_params], target[:, :n_params])
                if torch.isfinite(l):
                    val_sum += l.item()
                    n_val += 1
        avg_val = val_sum / max(n_val, 1)
        scheduler.step(avg_val)
        cur_lr = optimizer.param_groups[0]["lr"]

        loss_log.append([epoch + 1, avg_train, avg_val, phase])
        np.save(os.path.join(nle_dir, "loss_history.npy"), np.array(loss_log))
        print(
            f"Epoch {epoch + 1:03d} [{tag}] LR:{cur_lr:.2e} | "
            f"train {avg_train:.6f} | val {avg_val:.6f}",
            flush=True,
        )

        run.log({"epoch": epoch + 1, "val_loss": avg_val})  # for wandb dashbord

        # -- probes: RF-R2 + per-parameter q sigma -----------------------------
        if (epoch + 1) % probe_every == 0:
            r2 = probe_r2(compressor, val_loader, device, n_params)
            # phase / aux_coef logged alongside R2 so you can tell warmup vs
            # anneal vs pure-VMIM points apart when plotting rf_r2_history.npy
            rf_log.append([epoch + 1, phase, aux_coef] + r2)
            np.save(os.path.join(nle_dir, "rf_r2_history.npy"), np.array(rf_log))

            # --- Log RF-R2 to W&B ---
            probe_dict = {
                "epoch": epoch + 1,
                "probe/mean_r2": float(np.mean(r2)),
            }
            # Log individual parameter R^2 scores (theta_0 .. theta_3)
            for p_idx, score in enumerate(r2):
                probe_dict[f"probe/r2_theta_{p_idx}"] = float(score)

            if head is not None and not aux_only:
                s = probe_sigma(compressor, head, val_loader, device, n_params)
                if s is not None:
                    sigma_log.append([epoch + 1] + s)
                    np.save(os.path.join(nle_dir, "sigma_history.npy"), np.array(sigma_log))

                    # --- Log sigma to W&B ---
                    for p_idx, val in enumerate(s):
                        probe_dict[f"probe/sigma_theta_{p_idx}"] = float(val)

            # Send all probe metrics to W&B at once
            run.log(probe_dict)

            mean_r2 = float(np.mean(r2))
            if mean_r2 > best_probe + 1e-4:
                best_probe = mean_r2
                probes_since_best = 0
                # torch.save(raw.state_dict(),
                #           os.path.join(nle_dir, "learned_compressor_bestprobe.pt"))
                if head is not None:
                    torch.save(head.state_dict(), os.path.join(nle_dir, "best_vmim_head.pt"))
                print(
                    f"  saved best-by-probe (mean R2 {mean_r2:.3f}, "
                    f"{'pure-VMIM' if is_pure_vmim else tag}).",
                    flush=True,
                )
            elif is_pure_vmim:
                # BUGFIX 6: only counts staleness -- and can only trigger a
                # stop -- once we're actually in the pure-VMIM regime.
                probes_since_best += 1
                if probes_since_best >= probe_pat:
                    print(
                        f"Early stopping at epoch {epoch + 1}: mean probe R2 "
                        f"stale for {probe_pat} probes IN PURE-VMIM PHASE.",
                        flush=True,
                    )
                    break
            # else: aux_only or still in warmup/anneal -- probe logged, but
            # staleness is never counted and can never trigger a stop here.

        if avg_val < best_val:
            best_val = avg_val
            torch.save(raw.state_dict(), os.path.join(nle_dir, "learned_compressor.pt"))
            print("  saved best-by-loss.", flush=True)

    np.save(os.path.join(nle_dir, "loss_history.npy"), np.array(loss_log))
    torch.save(raw.state_dict(), os.path.join(nle_dir, "learned_compressor.pt"))

    run.finish()  # finish dashboard
    return best_val
