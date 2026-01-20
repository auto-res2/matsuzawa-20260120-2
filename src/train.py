"""src/train.py
End-to-end training script with Hydra, Optuna and WandB integration.
Fixes compared with the previous submission:
1. Mode overrides are now applied *after* the per-run YAML has been merged so
   that trial-mode values are not reverted.
2. `TORCH_HOME` is forced to `.cache/` before *any* model download so that both
   timm and torch-hub obey the caching requirement.
3. `val_top1_accuracy` is written to *both* WandB history *and* summary using
   exactly the same key that evaluate.py expects.
4. Real vs synthetic separation inside CCFD-Aug is now reliable: an explicit
   `is_real` flag is stored in the training `TensorDataset` instead of the
   incorrect `torch.arange (…) < half` shortcut.
5. Gradient-integrity assertion is retained and called right before every
   optimiser step.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from .model import MAPHead, ScoreNet, adversarial_feature_augment
from .preprocess import prepare_datasets

# -----------------------------------------------------------------------------
# Global cache directory for *everything* (datasets, timm weights, transformers…)
# -----------------------------------------------------------------------------
os.environ.setdefault("TORCH_HOME", ".cache")  # timm / torch-hub

# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Gradient-integrity guard
# -----------------------------------------------------------------------------

def _assert_gradients_exist(module: nn.Module) -> None:  # noqa: D401 – imperative
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            raise AssertionError(
                f"Gradient for parameter '{name}' is None just before optimizer.step()."
            )
        if torch.allclose(p.grad, torch.zeros_like(p.grad)):
            raise AssertionError(
                f"Gradient for parameter '{name}' is all zeros – most likely the graph was broken."
            )


# -----------------------------------------------------------------------------
# Tiny WandB stub – keeps Optuna trials completely offline
# -----------------------------------------------------------------------------
class _WandbStub:  # pragma: no cover – utility only
    def __init__(self) -> None:
        self.summary: Dict[str, Any] = {}

    def log(self, *_a, **_kw):  # type: ignore[no-self-use]
        pass

    def finish(self):  # type: ignore[no-self-use]
        pass

    def __repr__(self) -> str:  # type: ignore[no-self-use]
        return "<WandbStub>"


# -----------------------------------------------------------------------------
# Score-matching utilities (CCFD-Aug)
# -----------------------------------------------------------------------------

def _sigma(t: torch.Tensor, sigma_max: float, sigma_min: float = 1e-2) -> torch.Tensor:
    """Log-linear noise schedule."""
    return sigma_max * (sigma_min / sigma_max) ** t


def _dsm_step(
    scorenet: ScoreNet,
    opt: optim.Optimizer,
    feats: torch.Tensor,
    labels: torch.Tensor,
    sigma_max: float,
) -> float:
    """One denoising-score-matching step."""
    bsz, _ = feats.shape
    t = torch.rand(bsz, 1, device=feats.device)
    sig = _sigma(t, sigma_max)
    noise = torch.randn_like(feats)
    noisy = feats + sig * noise

    pred = scorenet(noisy, labels, t)
    loss = F.mse_loss(pred, noise)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    _assert_gradients_exist(scorenet)
    opt.step()
    return loss.item()


@torch.no_grad()
def _sample_features(
    scorenet: ScoreNet,
    class_labels: torch.Tensor,
    k: int,
    t_steps: int,
    feat_dim: int,
    sigma_max: float,
    sigma_min: float = 1e-2,
) -> torch.Tensor:
    """DDPM ancestral sampler – synthesises *k* features per label."""
    device = class_labels.device
    y = class_labels.repeat_interleave(k)
    feats = torch.randn(y.shape[0], feat_dim, device=device) * sigma_max
    for i in range(t_steps):
        t = torch.full((feats.size(0), 1), 1 - i / (t_steps - 1), device=device)
        sig = _sigma(t, sigma_max, sigma_min)
        grad = scorenet(feats, y, t)
        feats = feats + (sig ** 2) * grad  # Euler–Maruyama
        if i < t_steps - 1:
            t_next = torch.full_like(t, 1 - (i + 1) / (t_steps - 1))
            sig_next = _sigma(t_next, sigma_max, sigma_min)
            var = (sig_next ** 2 - sig ** 2).clamp(min=1e-5)
            feats = feats + var.sqrt() * torch.randn_like(feats)
    return feats.detach()


# -----------------------------------------------------------------------------
# Method-specific train loops
# -----------------------------------------------------------------------------

def _train_ccfd(cfg: DictConfig, data: Dict[str, Any], logger: Any) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim: int = cfg.model.feature_dim

    tr_feats, tr_lbls = data["train"]  # CPU tensors
    val_feats, val_lbls = data["val"]

    # ---------------- Score-net training ----------------
    scorenet = ScoreNet(feat_dim, data["n_classes"]).to(device)
    opt_s = optim.Adam(scorenet.parameters(), lr=cfg.training.learning_rate)

    steps_s = cfg.training.total_steps // 2  # half budget for score network
    dl_s = DataLoader(
        TensorDataset(tr_feats, tr_lbls),
        batch_size=cfg.training.batch_size,
        sampler=RandomSampler(tr_feats),
        drop_last=True,
    )
    it_s = iter(dl_s)
    for step in range(steps_s):
        while True:
            try:
                f_b, y_b = next(it_s)
                break
            except StopIteration:
                it_s = iter(dl_s)
        f_b, y_b = f_b.to(device), y_b.to(device)
        if step == 0:  # lifecycle assertion
            assert f_b.shape[1] == feat_dim, "Feature dimension mismatch at batch start."  # noqa: S101
        loss_s = _dsm_step(scorenet, opt_s, f_b, y_b, cfg.training.sigma_max)
        logger.log({"dsm_loss": loss_s}, step=step)

    # ---------------- Synthetic feature generation ----------------
    syn_feats = _sample_features(
        scorenet,
        tr_lbls.to(device),
        cfg.training.synthetic_ratio,
        cfg.training.diffusion_T,
        feat_dim,
        cfg.training.sigma_max,
    )
    syn_lbls = tr_lbls.repeat_interleave(cfg.training.synthetic_ratio).to(device)

    # ---------------- Classifier training ----------------
    clf = MAPHead(feat_dim, feat_dim, data["n_classes"]).to(device)
    opt_c = optim.SGD(clf.parameters(), lr=cfg.training.learning_rate * 50, momentum=0.9)
    sched_c = optim.lr_scheduler.CosineAnnealingLR(opt_c, cfg.training.total_steps)

    # Build dataset with explicit real/synthetic flag
    real_flag_real = torch.ones(tr_feats.size(0), dtype=torch.bool)
    real_flag_syn = torch.zeros(syn_feats.size(0), dtype=torch.bool)
    full_feats = torch.cat([tr_feats.to(device), syn_feats], 0)
    full_lbls = torch.cat([tr_lbls.to(device), syn_lbls], 0)
    full_flags = torch.cat([real_flag_real, real_flag_syn], 0).to(device)

    dset_c = TensorDataset(full_feats, full_lbls, full_flags)
    dl_c = DataLoader(
        dset_c,
        batch_size=cfg.training.batch_size,
        sampler=RandomSampler(full_feats),
        drop_last=True,
    )
    it_c = iter(dl_c)

    for step in range(cfg.training.total_steps):
        while True:
            try:
                f_b, y_b, is_real_b = next(it_c)
                break
            except StopIteration:
                it_c = iter(dl_c)
        f_b, y_b, is_real_b = f_b.to(device), y_b.to(device), is_real_b.to(device)

        f_real, y_real = f_b[is_real_b], y_b[is_real_b]
        f_syn, y_syn = f_b[~is_real_b], y_b[~is_real_b]
        if f_syn.numel() == 0:  # synthetic ratio may be 0 during Optuna search
            f_syn, y_syn = f_real, y_real

        logits_r = clf(f_real)
        logits_s = clf(f_syn)
        ce = F.cross_entropy(logits_r, y_real) + F.cross_entropy(logits_s, y_syn)
        kl = F.kl_div(
            F.log_softmax(logits_r, 1),
            F.softmax(logits_s.detach(), 1),
            reduction="batchmean",
        )
        loss = ce + cfg.training.lam_consistency * kl
        acc = (logits_r.argmax(1) == y_real).float().mean().item()

        opt_c.zero_grad(set_to_none=True)
        loss.backward()
        _assert_gradients_exist(clf)
        opt_c.step()
        sched_c.step()

        logger.log(
            {
                "classifier_loss": loss.item(),
                "train_top1_accuracy": acc,
                "lr": sched_c.get_last_lr()[0],
            },
            step=steps_s + step,
        )

    # ---------------- Validation ----------------
    clf.eval()
    with torch.no_grad():
        v_logits = clf(val_feats.to(device))
        preds = v_logits.argmax(1)
        val_acc = (preds == val_lbls.to(device)).float().mean().item()

    cm = confusion_matrix(val_lbls.cpu().numpy(), preds.cpu().numpy())
    logger.summary["confusion_matrix"] = cm.tolist()
    logger.summary["val_top1_accuracy"] = val_acc  # <-- critical for evaluate.py
    logger.log({"val_top1_accuracy": val_acc}, step=steps_s + cfg.training.total_steps)
    return {"val_top1_accuracy": val_acc}


def _train_ac_frofa(cfg: DictConfig, data: Dict[str, Any], logger: Any) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim: int = cfg.model.feature_dim

    tr_feats, tr_lbls = data["train"]
    val_feats, val_lbls = data["val"]

    clf = MAPHead(feat_dim, feat_dim, data["n_classes"]).to(device)
    opt = optim.SGD(clf.parameters(), lr=cfg.training.learning_rate, momentum=0.9)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.training.total_steps)

    dl = DataLoader(
        TensorDataset(tr_feats, tr_lbls),
        batch_size=cfg.training.batch_size,
        sampler=RandomSampler(tr_feats),
        drop_last=True,
    )
    it_dl = iter(dl)

    for step in range(cfg.training.total_steps):
        while True:
            try:
                f_b, y_b = next(it_dl)
                break
            except StopIteration:
                it_dl = iter(dl)
        f_b, y_b = f_b.to(device), y_b.to(device)

        f_adv = adversarial_feature_augment(f_b, y_b, clf, eps=cfg.training.adv_eps)
        logits_r = clf(f_b)
        logits_a = clf(f_adv)
        ce = F.cross_entropy(logits_r, y_b) + F.cross_entropy(logits_a, y_b)
        kl = F.kl_div(
            F.log_softmax(logits_r, 1),
            F.softmax(logits_a.detach(), 1),
            reduction="batchmean",
        )
        loss = ce + cfg.training.lam_consistency * kl
        acc = (logits_r.argmax(1) == y_b).float().mean().item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        _assert_gradients_exist(clf)
        opt.step()
        sched.step()

        logger.log(
            {
                "classifier_loss": loss.item(),
                "train_top1_accuracy": acc,
                "lr": sched.get_last_lr()[0],
            },
            step=step,
        )

    # ---------------- Validation ----------------
    clf.eval()
    with torch.no_grad():
        v_logits = clf(val_feats.to(device))
        preds = v_logits.argmax(1)
        val_acc = (preds == val_lbls.to(device)).float().mean().item()

    cm = confusion_matrix(val_lbls.cpu().numpy(), preds.cpu().numpy())
    logger.summary["confusion_matrix"] = cm.tolist()
    logger.summary["val_top1_accuracy"] = val_acc
    logger.log({"val_top1_accuracy": val_acc}, step=cfg.training.total_steps)
    return {"val_top1_accuracy": val_acc}


# -----------------------------------------------------------------------------
# Optuna helper (never logs online)
# -----------------------------------------------------------------------------

def _build_objective(cfg: DictConfig, train_fn, data):
    def _objective(trial: optuna.Trial):
        # Suggest all parameters declared in cfg.optuna.search_spaces
        for space in cfg.optuna.search_spaces:
            pname = space["param_name"]
            if space["distribution_type"] == "int":
                val = trial.suggest_int(pname, space["low"], space["high"])
            elif space["distribution_type"] == "uniform":
                val = trial.suggest_float(pname, space["low"], space["high"], log=False)
            elif space["distribution_type"] == "loguniform":
                val = trial.suggest_float(pname, space["low"], space["high"], log=True)
            else:
                raise ValueError(f"Unsupported distribution {space['distribution_type']}")
            OmegaConf.update(cfg, f"training.{pname}", val, merge=False)

        stub = _WandbStub()
        metrics = train_fn(cfg, data, logger=stub)
        return 1.0 - metrics["val_top1_accuracy"]  # minimise error

    return _objective


# -----------------------------------------------------------------------------
# Hydra entry point
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
# pylint: disable=too-many-branches,too-many-statements
def main(cfg: DictConfig) -> None:  # noqa: C901 – intentionally complex
    # ------------------------------------------------------------------
    # 1. Merge per-run YAML first so that trial/full overrides *stick*.
    # ------------------------------------------------------------------
    if cfg.run is None:
        raise ValueError("CLI argument run=<run_id> is mandatory – see documentation.")

    run_cfg_path = (
        Path(__file__).resolve().parent.parent / "config" / "runs" / f"{cfg.run}.yaml"
    )
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"Run-config file {run_cfg_path} not found.")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(run_cfg_path))

    # ------------------------------------------------------------------
    # 2. Apply mode-specific overrides *after* merge so they do not get reset.
    # ------------------------------------------------------------------
    if cfg.mode not in {"trial", "full"}:
        raise ValueError("mode must be either 'trial' or 'full'")
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        # Use *tiny* budgets to keep CI fast – never less than 1 step
        cfg.training.total_steps = max(5, cfg.training.total_steps // 40)
        cfg.training.batch_size = min(cfg.training.batch_size, 8)

    run_id: str = cfg.get("run_id", cfg.run)

    # ------------------------------------------------------------------
    # 3. Reproducibility & logging objects
    # ------------------------------------------------------------------
    _set_seed(cfg.training.seed)
    wandb_enabled = cfg.wandb.mode != "disabled"
    wb_run = None  # will be initialised lazily after Optuna search

    # ------------------------------------------------------------------
    # 4. Possibly iterate over several shot settings (1/5/10)
    # ------------------------------------------------------------------
    if hasattr(cfg.dataset, "selected_shot"):
        shot_values: List[int] = [int(cfg.dataset.selected_shot)]
    else:
        shot_values = [int(s) for s in cfg.dataset.shots_per_class]

    aggregated: Dict[str, float] = {}
    for shot in shot_values:
        # ----------------------------------------------------------
        # 4.a Prepare datasets (may generate synthetic random data in trial mode)
        # ----------------------------------------------------------
        data = prepare_datasets(cfg, shot=shot)
        train_impl = (
            _train_ccfd
            if cfg.method.startswith("Class-Conditional Feature Diffusion")
            else _train_ac_frofa
        )

        # ----------------------------------------------------------
        # 4.b Hyper-parameter search (Optuna) – offline only
        # ----------------------------------------------------------
        if int(cfg.optuna.n_trials) > 0:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=cfg.training.seed),
            )
            study.optimize(_build_objective(cfg, train_impl, data), n_trials=cfg.optuna.n_trials)
            # Apply best params to cfg for the *real* training run
            for k, v in study.best_params.items():
                OmegaConf.update(cfg, f"training.{k}", v, merge=False)

        # ----------------------------------------------------------
        # 4.c Initialise WandB *after* search so best params are recorded
        # ----------------------------------------------------------
        if wandb_enabled and wb_run is None:
            wb_run = wandb.init(
                entity=str(cfg.wandb.entity),
                project=str(cfg.wandb.project),
                id=run_id,
                resume="allow",
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=str(cfg.wandb.mode),
            )
            print(f"WandB run URL: {wb_run.url}")
        logger = wb_run if wandb_enabled else _WandbStub()

        # ----------------------------------------------------------
        # 4.d Actual training run with the best (or default) hyper-params
        # ----------------------------------------------------------
        metrics = train_impl(cfg, data, logger)
        aggregated[f"val_top1_accuracy_{shot}shot"] = metrics["val_top1_accuracy"]

    # ------------------------------------------------------------------
    # 5. Finalise WandB summary
    # ------------------------------------------------------------------
    if wandb_enabled and wb_run is not None:
        for k, v in aggregated.items():
            wb_run.summary[k] = v
        wb_run.finish()


if __name__ == "__main__":
    # make `python -m src.train` nicer in WandB UI
    sys.argv[0] = "src.train"
    main()
