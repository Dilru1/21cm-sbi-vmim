"""Stage-1 compressor training.

objective: vmim   loss = -log q(theta | t)          (GMM head, trained jointly at lr * head_lr_mult)
           mse    loss = lam_mse * |aux - theta|^2   (direct regression: aux == t)

Adam, ReduceLROnPlateau on the validation loss (factor 0.5, min lr/100), gradient clipping
at 1, fixed number of epochs, final-epoch weights kept. AMP on GPU for the compressor
forward pass; the loss is always fp32. Every `probe_every` epochs a random-forest R^2 of
theta from t and the head's conditional sigma are logged (diagnostics only).
"""
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_fn(c, head, t, aux, theta):
    if c["objective"] == "vmim":
        return -head.log_prob(t.float(), theta.float()).mean()
    return float(c["lam_mse"]) * F.mse_loss(aux, theta)


@torch.no_grad()
def probe_r2(compressor, loader, device, n_params, max_batches=400):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    compressor.eval()
    T, TH = [], []
    for b, (x, theta, *_) in enumerate(loader):
        if b >= max_batches:
            break
        T.append(compressor(x.to(device))[0].cpu().numpy())
        TH.append(theta[:, :n_params].numpy())
    Xtr, Xte, ytr, yte = train_test_split(np.concatenate(T), np.concatenate(TH),
                                          test_size=0.3, random_state=0)
    r2 = []
    for p in range(n_params):
        rf = RandomForestRegressor(100, min_samples_leaf=2, n_jobs=-1, random_state=0)
        rf.fit(Xtr, ytr[:, p])
        r2.append(float(r2_score(yte[:, p], rf.predict(Xte))))
    print("[RF R2]  " + " | ".join(f"t{p}:{v:.3f}" for p, v in enumerate(r2)), flush=True)
    return r2


@torch.no_grad()
def probe_sigma(compressor, head, loader, device, max_batches=40):
    compressor.eval()
    head.eval()
    s = []
    for b, (x, *_) in enumerate(loader):
        if b >= max_batches:
            break
        s.append(head.conditional_sigma(compressor(x.to(device))[0].float()).cpu().numpy())
    s = np.concatenate(s).mean(0)
    print("[q sigma] " + " | ".join(f"t{p}:{v:.3f}" for p, v in enumerate(s)), flush=True)
    return s.tolist()


def train(compressor, head, train_loader, val_loader, device, out_dir, c, n_params):
    lr = float(c["lr"])
    groups = [{"params": list(compressor.parameters()), "lr": lr}]
    if head is not None:
        groups.append({"params": list(head.parameters()), "lr": lr * float(c["head_lr_mult"])})
    opt = torch.optim.Adam(groups)
    all_params = [p for g in groups for p in g["params"]]
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=int(c["plateau_patience"]), min_lr=lr * 0.01)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    amp = (lambda: torch.amp.autocast("cuda")) if use_amp else nullcontext
    fp32 = (lambda: torch.autocast("cuda", enabled=False)) if use_amp else nullcontext

    def step_loss(x, theta):
        with amp():
            t, aux = compressor(x)
            t, aux = t.float(), aux.float()
            with fp32():
                return loss_fn(c, head, t, aux[:, :n_params], theta[:, :n_params].float())

    loss_log, rf_log, sigma_log = [], [], []
    for epoch in range(int(c["epochs"])):
        compressor.train()
        if head is not None:
            head.train()
        tr_sum, n_tr = 0.0, 0
        for x, theta, *_ in train_loader:
            x = x.to(device, non_blocking=True)
            theta = theta.to(device, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            loss = step_loss(x, theta)
            if not torch.isfinite(loss):
                continue
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()
            tr_sum += loss.item()
            n_tr += 1

        compressor.eval()
        if head is not None:
            head.eval()
        va_sum, n_va = 0.0, 0
        with torch.no_grad():
            for x, theta, *_ in val_loader:
                l = step_loss(x.to(device, non_blocking=True),
                              theta.to(device, non_blocking=True).float())
                if torch.isfinite(l):
                    va_sum += l.item()
                    n_va += 1
        tr, va = tr_sum / max(n_tr, 1), va_sum / max(n_va, 1)
        sched.step(va)
        loss_log.append([epoch + 1, tr, va])
        np.save(os.path.join(out_dir, "loss_history.npy"), np.array(loss_log))
        print(f"Epoch {epoch + 1:03d} LR:{opt.param_groups[0]['lr']:.2e} | train {tr:.6f} | val {va:.6f}",
              flush=True)

        if (epoch + 1) % int(c["probe_every"]) == 0:
            rf_log.append([epoch + 1] + probe_r2(compressor, val_loader, device, n_params))
            np.save(os.path.join(out_dir, "rf_r2_history.npy"), np.array(rf_log))
            if head is not None:
                sigma_log.append([epoch + 1] + probe_sigma(compressor, head, val_loader, device))
                np.save(os.path.join(out_dir, "sigma_history.npy"), np.array(sigma_log))

    torch.save(compressor.state_dict(), os.path.join(out_dir, "compressor.pt"))
