"""src/evaluate.py
Independent evaluation & visualisation script – unchanged API, but now tolerant
of missing keys and fully compatible with the metric names emitted by the new
train.py (i.e. `val_top1_accuracy` lives both in history and summary).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_wandb_cfg() -> Dict[str, str]:
    root_cfg = OmegaConf.load(Path(__file__).resolve().parent.parent / "config" / "config.yaml")
    return {"entity": str(root_cfg.wandb.entity), "project": str(root_cfg.wandb.project)}  # type: ignore[attr-defined]


def _export_json(obj: Dict[str, Any], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def _fig_name(prefix: str, topic: str) -> str:
    return f"{prefix}_{topic}.pdf"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: C901 – complex but explicit
    parser = argparse.ArgumentParser(description="Aggregate WandB runs & generate plots")
    parser.add_argument("results_dir", type=Path, help="Directory where outputs will be stored")
    parser.add_argument("run_ids", type=str, help='JSON list – e.g. "[\"run-1\", \"run-2\"]"')
    args = parser.parse_args()

    run_ids: List[str] = json.loads(args.run_ids)
    if not run_ids:
        raise ValueError("run_ids list is empty – nothing to evaluate.")

    wandb_cfg = _load_wandb_cfg()
    api = wandb.Api()

    primary_metric = "val_top1_accuracy"
    aggregated: Dict[str, Dict[str, float]] = defaultdict(dict)
    per_run_metric_paths: List[Path] = []

    # ------------------------------------------------------------------
    # Per-run processing
    # ------------------------------------------------------------------
    for rid in run_ids:
        run = api.run(f"{wandb_cfg['entity']}/{wandb_cfg['project']}/{rid}")
        history: pd.DataFrame = run.history()
        summary = run.summary._json_dict  # type: ignore[attr-defined]
        cfg = dict(run.config)

        run_dir = args.results_dir / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        # ---------- dump raw metrics ----------
        payload = {"history": history.to_dict(orient="list"), "summary": summary, "config": cfg}
        mpath = run_dir / "metrics.json"
        _export_json(payload, mpath)
        per_run_metric_paths.append(mpath)

        # ---------- dynamic primary metric detection ----------
        for k in summary.keys():
            if k == "val_top1_accuracy":
                primary_metric = k
                break
        for col in history.columns:
            if col.startswith("val_top1_accuracy"):
                primary_metric = col
                break

        # ---------- learning curve ----------
        if not history.empty:
            plt.figure(figsize=(6, 4))
            if "train_top1_accuracy" in history.columns:
                sns.lineplot(data=history, x=history.index, y="train_top1_accuracy", label="train")
            val_cols = [c for c in history.columns if c.startswith("val_top1_accuracy")]
            for vcol in val_cols:
                sns.lineplot(data=history, x=history.index, y=vcol, label=vcol)
            if val_cols or "train_top1_accuracy" in history.columns:
                plt.xlabel("Step")
                plt.ylabel("Accuracy")
                plt.title(f"{rid} learning curve")
                plt.legend()
                plt.tight_layout()
                fname = _fig_name(rid, "learning_curve")
                plt.savefig(run_dir / fname)
                plt.close()
                print(run_dir / fname)

        # ---------- confusion matrix ----------
        if "confusion_matrix" in summary:
            cm = np.asarray(summary["confusion_matrix"], dtype=int)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(include_values=False, cmap="Blues")
            plt.title(f"{rid} confusion matrix")
            plt.tight_layout()
            fname = _fig_name(rid, "confusion_matrix")
            plt.savefig(run_dir / fname)
            plt.close()
            print(run_dir / fname)

        # ---------- collect numeric metrics ----------
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                aggregated[k][rid] = float(v)

    # ------------------------------------------------------------------
    # Aggregated analysis
    # ------------------------------------------------------------------
    comp_dir = args.results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    metric_runs = aggregated.get(primary_metric, {})
    best_prop_id, best_prop_val = None, -np.inf
    best_base_id, best_base_val = None, -np.inf
    for rid, val in metric_runs.items():
        if "proposed" in rid and val > best_prop_val:
            best_prop_id, best_prop_val = rid, val
        if ("baseline" in rid or "comparative" in rid) and val > best_base_val:
            best_base_id, best_base_val = rid, val

    gap = None
    if best_prop_val != -np.inf and best_base_val != -np.inf:
        gap = (best_prop_val - best_base_val) / max(best_base_val, 1e-12) * 100.0

    agg_json = {
        "primary_metric": primary_metric,
        "metrics": aggregated,
        "best_proposed": {"run_id": best_prop_id, "value": best_prop_val},
        "best_baseline": {"run_id": best_base_id, "value": best_base_val},
        "gap": gap,
    }
    _export_json(agg_json, comp_dir / "aggregated_metrics.json")
    print(comp_dir / "aggregated_metrics.json")

    # ---------- bar chart ----------
    if metric_runs:
        plt.figure(figsize=(max(6, len(metric_runs) * 0.9), 4))
        sns.barplot(x=list(metric_runs.keys()), y=list(metric_runs.values()))
        plt.ylabel(primary_metric)
        plt.xticks(rotation=45, ha="right")
        for i, (k, v) in enumerate(metric_runs.items()):
            plt.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        fname = _fig_name("comparison", "accuracy_bar_chart")
        plt.savefig(comp_dir / fname)
        plt.close()
        print(comp_dir / fname)

    # ---------- statistical significance ----------
    sig_json_path = comp_dir / "significance_test.json"
    if best_prop_id and best_base_id:
        prop_vals = [v for k, v in metric_runs.items() if "proposed" in k]
        base_vals = [v for k, v in metric_runs.items() if ("baseline" in k or "comparative" in k)]
        if len(prop_vals) >= 2 and len(base_vals) >= 2:
            t_stat, p_val = stats.ttest_ind(prop_vals, base_vals, equal_var=False)
            _export_json({"t_statistic": float(t_stat), "p_value": float(p_val)}, sig_json_path)
        else:
            _export_json({"warning": "Not enough samples for significance test"}, sig_json_path)
        print(sig_json_path)

    # ---------- CI artefact paths ----------
    for p in per_run_metric_paths:
        print(p)


if __name__ == "__main__":
    main()
