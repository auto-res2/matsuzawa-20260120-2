"""src/main.py
Main orchestrator – receives CLI overrides via Hydra and spawns `src.train`
inside a clean subprocess so that each experiment has an isolated Python
interpreter (important for CUDA memory).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig

# Force global cache directory before *anything* else touches torch-hub
os.environ.setdefault("TORCH_HOME", ".cache")


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pylint: disable=missing-function-docstring
    if cfg.run is None:
        raise ValueError("CLI argument run=<id> is required – see documentation.")

    cmd: List[str] = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"mode={cfg.mode}",
        f"results_dir={cfg.results_dir}",
    ]

    print("Launching training subprocess:\n", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Training subprocess exited with code {proc.returncode}")


if __name__ == "__main__":
    main()
