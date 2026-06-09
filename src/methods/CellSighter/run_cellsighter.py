#!/usr/bin/env python3
"""
run_cellsighter.py  —  CellSighter benchmark wrapper for spCellEval

Runs CellSighter on the preprocessed IMMUcan dataset (output of prepare_immucan.py).
For each fold:
  1. Writes config.json pointing at the clean preprocessed data
  2. Trains n_runs independent CellSighter models
  3. Merges ensemble results
  4. Saves predictions_{fold}.csv in spCellEval benchmark format

IMPORTANT: Run from inside ~/CellSighter/ so train.py imports work:
    cd ~/CellSighter
    python run_cellsighter.py --dataset immucan --fold 0 --n_runs 1

Preprocessing must be done first:
    python prepare_immucan.py
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# PATHS

BASE_DATA        = Path("/home/juliaoesterle/data/phenotyping_benchmark")
RESULTS_DIR      = Path("/home/juliaoesterle/results/cellsighter")
CELLSIGHTER_REPO = Path("/home/juliaoesterle/CellSighter")

# Preprocessed clean data — output of prepare_immucan.py
IMMUCAN_CLEAN = BASE_DATA / "IMMUcan_CellSighter"

IMMUCAN_CFG = {
    "clean_dir":       IMMUCAN_CLEAN,                          # root dir (has CellTypes/ inside)
    "folds_json":      IMMUCAN_CLEAN / "folds.json",           # short IDs
    "channels_txt":    IMMUCAN_CLEAN / "channels.txt",         # 37 channels
    "label_csv":       IMMUCAN_CLEAN / "labels_cell_type.csv",
    "id_map_csv":      IMMUCAN_CLEAN / "id_map.csv",
    "n_classes":       13,
    "crop_input_size": 60,    # paper default for MIBI/IMC data
    "crop_size":       128,   # ~2x crop_input_size for augmentation headroom
    "batch_size":      32,
    "num_workers":     4,
    "epoch_max":       30,
    "size_data":       3000,  # max cells per class sampled during training
}

DATASET_CFGS = {"immucan": IMMUCAN_CFG}

# DATA LOADING

def load_folds(cfg):
    with open(cfg["folds_json"]) as f:
        raw = json.load(f)
    return [{"train": raw[f"fold_{i}_train_set"], "test": raw[f"fold_{i}_test_set"]}
            for i in range(5)]


def load_label_map(cfg):
    df = pd.read_csv(cfg["label_csv"])   # columns: phenotype, label
    int2type = dict(zip(df["label"], df["phenotype"]))
    return int2type

# CONFIG

def write_config(run_dir, train_ids, val_ids, cfg, weight_to_eval=""):
    """
    Write config.json for one CellSighter run.
    root_dir points at the preprocessed clean directory —
    load_crops will append /CellTypes internally.
    """
    config = {
        "crop_input_size": cfg["crop_input_size"],
        "crop_size":       cfg["crop_size"],
        "root_dir":        str(cfg["clean_dir"]),   # load_crops appends /CellTypes
        "train_set":       train_ids,
        "val_set":         val_ids,
        "num_classes":     cfg["n_classes"],
        "epoch_max":       cfg["epoch_max"],
        "lr":              0.001,
        "to_pad":          False,
        "blacklist":       [],
        "channels_path":   str(cfg["channels_txt"]),
        "weight_to_eval":  weight_to_eval,
        "sample_batch":    True,
        "size_data":       cfg["size_data"],
        "aug":             True,
        "batch_size":      cfg["batch_size"],
        "num_workers":     cfg["num_workers"],
        "hierarchy_match": None,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


# TRAINING + EVALUATION


def run_cmd(cmd, cwd):
    print(f"  [cmd] {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=True)


def train_one_run(run_dir, repo):
    run_cmd([sys.executable, repo / "train.py",
             f"--base_path={run_dir}"], cwd=repo)


def eval_one_run(run_dir, repo):
    """Find latest checkpoint, update config, run eval.py."""
    pth_files = sorted(run_dir.glob("weights_*_count.pth"),
                       key=lambda p: int(p.stem.split("_")[1]))
    if not pth_files:
        raise FileNotFoundError(f"No weights_*_count.pth in {run_dir}")
    best = pth_files[-1]
    print(f"  [eval] Checkpoint: {best.name}")

    with open(run_dir / "config.json") as f:
        config = json.load(f)
    config["weight_to_eval"] = str(best)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    run_cmd([sys.executable, repo / "eval.py",
             f"--base_path={run_dir}"], cwd=repo)


# ENSEMBLE MERGING  (inlined from analyze_results/unified_ensemble.py)


def merge_ensemble(run_dirs, output_path):
    val_results = []
    for rd in run_dirs:
        candidates = sorted(rd.glob("val_results_*.csv"),
                            key=lambda p: int(p.stem.split("_")[-1]))
        if candidates:
            val_results.append(str(candidates[-1]))
        else:
            print(f"  [warn] No val_results_*.csv in {rd}")

    if not val_results:
        raise FileNotFoundError("No val_results CSV files found")

    print(f"  [merge] Merging {len(val_results)} run(s)...")
    ensemble_size = len(val_results)
    df_all = pd.DataFrame()

    for i, vr in enumerate(val_results):
        curr_df = pd.read_csv(vr, index_col=0)
        prob_list = curr_df["prob_list"].apply(eval)
        num_classes = len(prob_list.iloc[0])
        curr_df[[f"prob_class_{j}" for j in range(num_classes)]] = \
            prob_list.apply(pd.Series)
        curr_df.columns = [c + f"_ens_{i}" for c in curr_df.columns]
        df_all = pd.concat([df_all, curr_df], axis=1)

    for i in range(num_classes):
        df_all[f"prob_mean_class_{i}"] = \
            df_all[[f"prob_class_{i}_ens_{j}" for j in range(ensemble_size)]].mean(axis=1)

    df_all["pred"]     = df_all[[f"prob_mean_class_{i}" for i in range(num_classes)]].values.argmax(1)
    df_all["pred_prob"]= df_all[[f"prob_mean_class_{i}" for i in range(num_classes)]].max(axis=1)
    df_all["label"]    = df_all["label_ens_0"]
    df_all["cell_id"]  = df_all["cell_id_ens_0"]
    df_all["image_id"] = df_all["image_id_ens_0"]

    merged = df_all[["image_id", "cell_id", "label", "pred", "pred_prob"]].copy()
    merged.to_csv(output_path)
    print(f"  [merge] Saved → {output_path}")
    return merged

# OUTPUT  →  spCellEval benchmark format


def save_predictions(merged_df, fold_idx, int2type, out_path):
    """
    Convert to spCellEval predictions_{fold}.csv.
    Translates short IDs back to original IDs via id_map if needed.
    """
    df = merged_df.copy()
    df["true_phenotype"]      = df["label"].map(int2type).fillna("undefined")
    df["predicted_phenotype"] = df["pred"].map(int2type).fillna("undefined")
    df["confidence"]          = df["pred_prob"]
    df["fold"]                = fold_idx

    out = df[["image_id", "cell_id", "fold",
              "true_phenotype", "predicted_phenotype", "confidence"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    valid    = out[out["true_phenotype"] != "undefined"]
    macro_f1 = f1_score(valid["true_phenotype"], valid["predicted_phenotype"],
                        average="macro", zero_division=0)
    print(f"  [score] Fold {fold_idx} Macro F1 = {macro_f1:.4f}  (n={len(valid)} cells)")
    return macro_f1

# MAIN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="immucan", choices=list(DATASET_CFGS.keys()))
    p.add_argument("--fold",    type=int, default=None,
                   help="Single fold (0-4). Omit for all 5.")
    p.add_argument("--n_runs",  type=int, default=1,
                   help="Ensemble size. 1=smoke test, 10=paper setting.")
    return p.parse_args()


def main():
    args  = parse_args()
    cfg   = DATASET_CFGS[args.dataset]
    folds = load_folds(cfg)
    int2type = load_label_map(cfg)

    # Verify preprocessed data exists
    if not cfg["clean_dir"].exists():
        print(f"ERROR: Preprocessed data not found at {cfg['clean_dir']}")
        print("Run prepare_immucan.py first!")
        sys.exit(1)

    fold_indices = [args.fold] if args.fold is not None else list(range(5))
    work_base    = RESULTS_DIR / args.dataset / "work"
    output_base  = RESULTS_DIR / args.dataset / "level3"

    print(f"\n{'='*60}")
    print(f"CellSighter | dataset={args.dataset} | folds={fold_indices} | n_runs={args.n_runs}")
    print(f"Data      : {cfg['clean_dir']}")
    print(f"Work dir  : {work_base}")
    print(f"Output dir: {output_base}")
    print(f"{'='*60}\n")

    all_f1s = {}

    for fold_idx in fold_indices:
        print(f"\n─── FOLD {fold_idx} ───────────────────────────────────────")
        fold     = folds[fold_idx]
        fold_dir = work_base / f"fold_{fold_idx}"

        # Train ensemble
        print(f"[1/3] Training {args.n_runs} model(s)...")
        run_dirs = []
        for run_i in range(args.n_runs):
            run_dir = fold_dir / f"run_{run_i}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_config(run_dir, fold["train"], fold["test"], cfg)
            print(f"  Training run {run_i+1}/{args.n_runs}...")
            train_one_run(run_dir, CELLSIGHTER_REPO)
            run_dirs.append(run_dir)

        # Evaluate
        print("[2/3] Evaluating...")
        for run_dir in run_dirs:
            eval_one_run(run_dir, CELLSIGHTER_REPO)

        # Merge + save
        print("[3/3] Merging and saving predictions...")
        merged_df = merge_ensemble(run_dirs, fold_dir / "merged_ensemble.csv")
        out_path  = output_base / f"predictions_{fold_idx}.csv"
        f1 = save_predictions(merged_df, fold_idx, int2type, out_path)
        all_f1s[fold_idx] = f1
        print(f"  Saved → {out_path}")

    if all_f1s:
        mean_f1 = np.mean(list(all_f1s.values()))
        print(f"\n{'='*60}")
        print(f"DONE  |  Mean Macro F1: {mean_f1:.4f}")
        for fi, f1 in sorted(all_f1s.items()):
            print(f"  Fold {fi}: {f1:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()