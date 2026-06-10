#!/usr/bin/env python3
"""
train_wrapper.py  —  CellSighter training with internal val split + early stopping
===================================================================================
Drop-in replacement for KerenLab/CellSighter's train.py, called identically:

    python train_wrapper.py --base_path /path/to/run_dir

It reads config.json from base_path (written by run_cellsighter.py) and behaves
exactly like the upstream train.py EXCEPT:

  1. Internal StratifiedKFold split of the TRAIN images' cells into
     (internal_train / internal_val) using `internal_val_fraction`. The fold's
     test images (config["val_set"], from folds.json) are never loaded here, so
     there is no test leakage during training.
  2. Early stopping on internal-val loss with `early_stopping_patience`; only the
     best checkpoint is kept as weights.pth (upstream kept the LAST epoch only).
  3. batch_size / num_workers / lr / epoch_max read from config.

Deliberately UNCHANGED from upstream (do not "fix" these — they are correct):
  * Class imbalance is handled by the WeightedRandomSampler (sample_batch=true),
    NOT by a class-weighted loss. Stacking both double-corrects and hurts F1, so
    the criterion stays a plain CrossEntropyLoss, matching Marcel's good runs.
  * Channel count: num_channels = (#lines in channels.txt) + 1 - len(blacklist),
    then Model(num_channels + 1, ...). The two +1's are the all_cells_mask
    environment channel and the per-cell mask channel.

"""

import sys
sys.path.append(".")

import argparse
import json
import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Model
from data.data import CellCropsDataset
from data.utils import load_crops
from data.transform import train_transform, val_transform
from eval import val_epoch

# Reuse upstream helpers verbatim so behaviour matches train.py exactly.
from train import subsample_const_size, define_sampler, train_epoch


def stratified_internal_split(crops, val_fraction, seed=42):
    """Split crops into (train, val) stratified by label. Returns two arrays."""
    labels = np.array([c._label for c in crops])
    n_splits = max(2, int(round(1.0 / val_fraction)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(skf.split(np.zeros(len(labels)), labels))
    return crops[train_idx], crops[val_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    args = parser.parse_args()

    with open(os.path.join(args.base_path, "config.json")) as f:
        config = json.load(f)

    writer = SummaryWriter(log_dir=args.base_path)
    device = config.get("device", "cuda")

    # Load ONLY the fold's train images; never touch val_set (= fold test) 
    train_crops, _ = load_crops(
        config["root_dir"], config["channels_path"], config["crop_size"],
        config["train_set"], [],            # empty val_set -> nothing test-side loaded
        config["to_pad"], blacklist_channels=config["blacklist"],
    )
    train_crops = np.array([c for c in train_crops if c._label >= 0])
    if config.get("size_data"):
        train_crops = np.array(subsample_const_size(train_crops, config["size_data"]))

    #  Internal stratified train/val split
    inner_train, inner_val = stratified_internal_split(
        train_crops, config.get("internal_val_fraction", 0.2))
    print(f"internal split: {len(inner_train)} train / {len(inner_val)} val cells")

    crop_input_size = config.get("crop_input_size", 100)
    shift = 5
    aug = config.get("aug", True)
    tr_tf = train_transform(crop_input_size, shift) if aug else val_transform(crop_input_size)
    va_tf = val_transform(crop_input_size)

    train_ds = CellCropsDataset(inner_train, transform=tr_tf, mask=True)
    val_ds = CellCropsDataset(inner_val, transform=va_tf, mask=True)

    sampler = define_sampler(inner_train, config["hierarchy_match"])
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], num_workers=config["num_workers"],
        sampler=sampler if config["sample_batch"] else None,
        shuffle=False if config["sample_batch"] else True)
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], num_workers=config["num_workers"],
        shuffle=False)

    # Model (channel arithmetic identical to upstream) 
    n_channels = sum(1 for _ in open(config["channels_path"])) + 1 - len(config["blacklist"])
    model = Model(n_channels + 1, config["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    criterion = torch.nn.CrossEntropyLoss()           # sampler handles imbalance

    best_val = float("inf")
    patience = config.get("early_stopping_patience", 15)
    no_improve = 0
    best_path = os.path.join(args.base_path, "weights.pth")

    for epoch in range(config["epoch_max"]):
        train_epoch(model, train_loader, optimizer, criterion,
                    epoch=epoch, writer=writer, device=device)
        scheduler.step()

        # internal-val loss
        model.eval()
        with torch.no_grad():
            losses = []
            for batch in val_loader:
                x = batch["image"]
                m = batch.get("mask", None)
                if m is not None:
                    x = torch.cat([x, m], dim=1)
                x = x.to(device)
                y = batch["label"].to(device)
                losses.append(criterion(model(x), y).item())
        val_loss = float(np.mean(losses)) if losses else float("inf")
        writer.add_scalar("Loss/val", val_loss, epoch)
        print(f"epoch {epoch} | val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print("  ✓ new best model saved")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stopping at epoch {epoch} (best val_loss={best_val:.4f})")
                break

    if not os.path.exists(best_path):       # safety: never improved
        torch.save(model.state_dict(), best_path)
    print(f"training done. best weights -> {best_path}")


if __name__ == "__main__":
    main()