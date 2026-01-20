"""src/preprocess.py
Utility to (1) extract ViT-B/16 features into `.cache/` and (2) return synthetic
mini datasets when `mode=trial` so that CI remains lightweight.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import timm
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Ensure *all* downloads land in the repository-local cache
CACHE_ROOT = Path(".cache")
CACHE_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT))


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _cache_path(dataset_name: str, shots: int) -> Path:
    fname = f"features_{dataset_name.replace(' ', '_')}_{shots}shot.pt"
    return CACHE_ROOT / fname


def _few_shot_subset(dataset: ImageFolder, shots: int) -> ImageFolder:
    indices: List[int] = []
    per_class_counter: Dict[int, int] = {}
    for idx, (_, lbl) in enumerate(dataset.samples):
        if per_class_counter.get(lbl, 0) < shots:
            indices.append(idx)
            per_class_counter[lbl] = per_class_counter.get(lbl, 0) + 1
    dataset.samples = [dataset.samples[i] for i in indices]
    dataset.imgs = dataset.samples  # type: ignore[attr-defined]
    return dataset


@torch.no_grad()
def _extract_feats(backbone, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    feats, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        out = backbone(imgs)
        if out.dim() == 3:  # (B, tokens, C) – mean-pool patch tokens (exclude CLS)
            out = out[:, 1:, :].mean(1)
        feats.append(out.cpu())
        labels.append(lbls)
    return torch.cat(feats, 0), torch.cat(labels, 0)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def prepare_datasets(cfg: DictConfig, *, shot: int) -> Dict[str, Any]:
    """Return a dict with keys: train, val, n_classes.

    In trial mode synthetic random tensors are returned to minimise compute.
    In full mode, feature tensors are cached on disk so that subsequent runs are
    instantaneous.
    """

    # ---------------- Trial mode – synthetic tensors ----------------
    if cfg.mode == "trial":
        n_cls = 3
        dim = cfg.model.feature_dim
        rng = torch.Generator().manual_seed(cfg.training.seed)
        tr_f = torch.randn(n_cls * shot, dim, generator=rng)
        tr_l = torch.arange(n_cls).repeat_interleave(shot)
        val_f = torch.randn(n_cls * 5, dim, generator=rng)
        val_l = torch.randint(0, n_cls, (n_cls * 5,), generator=rng)
        return {"train": (tr_f, tr_l), "val": (val_f, val_l), "n_classes": n_cls}

    # ---------------- Full mode – real data or cached ----------------
    ds_name = cfg.dataset.name.replace(" ", "_")
    cpath = _cache_path(ds_name, shot)
    if cpath.exists():
        return torch.load(cpath, map_location="cpu")

    # Expect an environment variable that points to the ImageNet root folder.
    root_dir = os.getenv("IMAGENET_ROOT")
    if root_dir is None or not Path(root_dir).exists():
        raise EnvironmentError(
            "Set IMAGENET_ROOT to the directory that contains 'train' and 'val'."
        )

    tfm = transforms.Compose(
        [
            transforms.Resize(cfg.dataset.preprocessing.resize),
            transforms.CenterCrop(cfg.dataset.preprocessing.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds_tr = ImageFolder(Path(root_dir) / "train", transform=tfm)
    ds_val = ImageFolder(Path(root_dir) / "val", transform=tfm)
    ds_tr = _few_shot_subset(ds_tr, shot)

    dl_tr = DataLoader(ds_tr, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=0, global_pool=""
    ).to(device)
    backbone.eval()

    tr_feats, tr_lbls = _extract_feats(backbone, dl_tr, device)
    val_feats, val_lbls = _extract_feats(backbone, dl_val, device)

    payload = {
        "train": (tr_feats, tr_lbls),
        "val": (val_feats, val_lbls),
        "n_classes": len(ds_tr.classes),
    }
    torch.save(payload, cpath)
    return payload
