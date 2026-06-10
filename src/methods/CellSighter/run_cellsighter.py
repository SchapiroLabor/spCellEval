#!/usr/bin/env python3
"""
run_cellsighter.py  —  dataset-agnostic CellSighter orchestration
==================================================================
Runs CellSighter cross-validation for any dataset described in the single
per-method config, selected with --dataset. Stays dataset-agnostic: no dataset
constants live in this file (Lukas' request — all specifics in the config).

    python run_cellsighter.py --dataset immucan \
        --config src/methods/configs/cellsighter.json

    python run_cellsighter.py --dataset chl \
        --config src/methods/configs/cellsighter.json --fold 0 --n_runs 1

Pipeline per fold k:
  1. train_set = all prepared images NOT in fold k; val_set = images in fold k.
  2. write a CellSighter config.json into <work>/fold{k}/run{r}/.
  3. train_wrapper.py  -> weights.pth   (internal val split + early stopping)
  4. eval.py           -> val_results.csv on the held-out fold images
  5. (n_runs>1) average class probabilities across the ensemble runs.
  6. write predictions_{k}.csv in spCellEval format
     (sample_id, cell_id, label, prediction[, prob_*]).

Assumes prepare_dataset.py has already produced <output_root>/CellTypes/... .
Run from the CellSighter repo dir (so train_wrapper.py/eval.py imports resolve).
"""

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent     # the CellSighter repo dir


def load_config(config_path: str, dataset: str) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    if dataset not in cfg["datasets"]:
        raise ValueError(f"Dataset '{dataset}' not in {config_path}. "
                         f"Available: {list(cfg['datasets'])}")
    merged = dict(cfg.get("defaults", {}))
    merged.update(cfg["datasets"][dataset])
    merged["method"] = cfg.get("method", "cellsighter")
    merged["dataset"] = dataset
    return merged


def write_cellsighter_config(base: Path, cfg: dict, root_dir: Path,
                             channels_path: Path, train_set, val_set,
                             num_classes: int, weight_to_eval: str = "") -> None:
    """Write the config.json that upstream train_wrapper.py / eval.py consume."""
    cs_cfg = {
        "crop_input_size": cfg["crop_input_size"],
        "crop_size": cfg["crop_size"],
        "root_dir": str(root_dir),
        "train_set": list(train_set),
        "val_set": list(val_set),
        "num_classes": num_classes,
        "epoch_max": cfg["epoch_max"],
        "lr": cfg["lr"],
        "blacklist": [],                       # channel exclusion already applied in prepare
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
        "channels_path": str(channels_path),
        "weight_to_eval": weight_to_eval,
        "sample_batch": cfg["sample_batch"],
        "to_pad": cfg["to_pad"],
        "hierarchy_match": cfg["hierarchy_match"],
        "aug": cfg["aug"],
        "size_data": cfg["size_data"],
        "device": cfg.get("device", "cuda"),
        "early_stopping_patience": cfg["early_stopping_patience"],
        "internal_val_fraction": cfg["internal_val_fraction"],
    }
    if cs_cfg["size_data"] is None:
        cs_cfg.pop("size_data")
    base.mkdir(parents=True, exist_ok=True)
    with open(base / "config.json", "w") as f:
        json.dump(cs_cfg, f, indent=2)


def run_step(script: str, base: Path):
    """Invoke upstream train_wrapper.py / eval.py with the repo dir as cwd."""
    cmd = [sys.executable, str(HERE / script), "--base_path", str(base)]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(HERE), check=True)


def read_val_results(path: Path) -> pd.DataFrame:
    """Parse upstream val_results.csv -> tidy frame with per-class prob matrix."""
    df = pd.read_csv(path)
    probs = np.array([ast.literal_eval(p) for p in df["prob_list"]])
    out = pd.DataFrame({
        "sample_id": df["image_id"],
        "cell_id": df["cell_id"].astype(int),
        "label": df["label"].astype(int),
    })
    return out, probs


def main():
    ap = argparse.ArgumentParser(description="Dataset-agnostic CellSighter runner")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=None, help="run a single fold (default: all)")
    ap.add_argument("--n_runs", type=int, default=None, help="ensemble size (default: config)")
    ap.add_argument("--results_dir", default=None, help="where predictions_{k}.csv go")
    args = ap.parse_args()

    cfg = load_config(args.config, args.dataset)
    n_runs = args.n_runs or cfg.get("n_runs", 1)

    data_root = Path(cfg["output_root"])
    ct = data_root / "CellTypes"
    channels_path = ct / "channels.txt"
    with open(ct / "folds.json") as f:
        folds = json.load(f)
    with open(ct / "label_map.json") as f:
        label_map = json.load(f)              # {class_name: int}
    int2str = {v: k for k, v in label_map.items()}
    # output dim must exceed the largest label id (ids may be non-contiguous)
    num_classes = max(label_map.values()) + 1

    all_images = sorted({sid for v in folds.values() for sid in v})
    results_dir = Path(args.results_dir or (data_root / "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    fold_keys = [str(args.fold)] if args.fold is not None else sorted(folds, key=lambda k: int(k))

    for k in fold_keys:
        test_imgs = list(folds[k])
        train_imgs = [s for s in all_images if s not in set(test_imgs)]
        print(f"\n=== fold {k}: {len(train_imgs)} train imgs / {len(test_imgs)} test imgs ===")

        ensemble_probs, base_df = None, None
        for r in range(n_runs):
            base = results_dir / f"fold{k}" / f"run{r}"
            # train
            write_cellsighter_config(base, cfg, data_root, channels_path,
                                     train_imgs, test_imgs, num_classes)
            run_step("train_wrapper.py", base)
            # eval with the best checkpoint
            write_cellsighter_config(base, cfg, data_root, channels_path,
                                     train_imgs, test_imgs, num_classes,
                                     weight_to_eval=str(base / "weights.pth"))
            run_step("eval.py", base)

            df_r, probs_r = read_val_results(base / "val_results.csv")
            if ensemble_probs is None:
                base_df, ensemble_probs = df_r, probs_r
            else:
                # align by (sample_id, cell_id) then accumulate
                key = ["sample_id", "cell_id"]
                order = base_df.merge(df_r.reset_index(), on=key, how="left")["index"].values
                ensemble_probs = ensemble_probs + probs_r[order]

        ensemble_probs = ensemble_probs / n_runs
        pred_int = ensemble_probs.argmax(1)
        out = base_df.copy()
        out["prediction"] = [int2str.get(int(p), str(p)) for p in pred_int]
        out["label_name"] = [int2str.get(int(l), str(l)) for l in out["label"]]
        out["confidence"] = ensemble_probs.max(1)
        out_path = results_dir / f"predictions_{k}.csv"
        out.to_csv(out_path, index=False)
        print(f"  -> wrote {out_path}  ({len(out)} cells)")

    print("\nAll folds done.")


if __name__ == "__main__":
    main()