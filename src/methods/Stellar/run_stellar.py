#!/usr/bin/env python3
"""
run_stellar.py — supervised STELLAR (GNN) cell phenotyping, config-driven.

Per-dataset settings live in configs/stellar.json, keyed by the dataset folder name.
The neighbourhood radius is set in micrometres (neighbourhood_um) and converted to
pixels with the dataset's pixel_size_um. Ground truth and folds come from the
benchmark's quantification table and fold CSVs (utils/benchmark_io.py).

Model: Linear(in->hid) + ReLU + SAGEConv(hid->hid) + Linear(hid->classes), trained with
class-weighted cross-entropy on NeighborLoader mini-batches of the merged training
graph; the last epoch is evaluated. Expression features are z-scored with the
training split's statistics.

Output (<output-dir>/stellar_supervised/):
    level3/predictions_<fold>.csv, level3/fold_times.txt
    models/model_fold<fold>.pth, learning_curves/fold_<fold>.csv, fold_metrics_summary.csv

Usage:
    python run_stellar.py --dataset-path <data>/cHL_2_MIBI --config configs/stellar.json \
        --output-dir <results>/cHL_2_MIBI
    python run_stellar.py ... --fold 0 --epochs 2 --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.benchmark_io import (load_benchmark_data, load_config, read_image,  # noqa: E402
                                read_markers, read_segmentation)


#  MODEL

class StellarModel(nn.Module):
    def __init__(self, input_dim, hid_dim, num_classes):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, hid_dim)
        self.graph_conv = SAGEConv(hid_dim, hid_dim)
        self.fc_net = nn.Linear(hid_dim, num_classes)

    def forward(self, data):
        feat = F.relu(self.input_linear(data.x))
        out_feat = self.graph_conv(feat, data.edge_index)
        return self.fc_net(out_feat), out_feat


#  FEATURES AND GRAPHS

def image_features(img_path, seg_path, channel_idx):
    """Mean channel intensity and (row, col) centroid per segmentation label, indexed by label."""
    img = read_image(img_path, channel_idx)
    mask = read_segmentation(seg_path)
    if mask.shape != img.shape[1:]:
        raise ValueError(f"{img_path}: image {img.shape[1:]} and mask {mask.shape} differ")
    labels = mask.ravel()
    n = int(labels.max()) + 1
    area = np.bincount(labels, minlength=n)
    denom = np.maximum(area, 1)[:, None]
    expr = np.stack([np.bincount(labels, weights=ch.ravel(), minlength=n) for ch in img], axis=1)
    rows, cols = np.divmod(np.arange(labels.size), mask.shape[1])
    centroid = np.stack([np.bincount(labels, weights=rows, minlength=n),
                         np.bincount(labels, weights=cols, minlength=n)], axis=1)
    return (expr / denom).astype(np.float32), centroid / denom, area > 0


def build_split(cells, features, cat_to_idx, mean=None, std=None):
    """Arrays for one split, sorted by (image_id, cell_id). mean/std default to this split's own."""
    cells = cells.sort_values(["image_id", "cell_id"]).reset_index(drop=True)
    X, pos = [], []
    for img_id, grp in cells.groupby("image_id", sort=False):
        expr, centroid, present = features[img_id]
        cid = grp["cell_id"].to_numpy()
        missing = (cid >= len(present)) | ~present[np.minimum(cid, len(present) - 1)]
        if missing.any():
            raise ValueError(f"{img_id}: {int(missing.sum()):,} fold cells have no pixels in the segmentation")
        X.append(expr[cid])
        pos.append(centroid[cid])
    X = np.concatenate(X)
    if mean is None:
        mean, std = X.mean(axis=0), X.std(axis=0)
    return {
        "cells": cells,
        "X": ((X - mean) / (std + 1e-8)).astype(np.float32),
        "pos": np.concatenate(pos),
        "y": cells["label"].map(cat_to_idx).to_numpy(np.int64),
    }, mean, std


def radius_edges(pos, radius):
    """edge_index (2, E) of all cell pairs within `radius`, both directions, no self-loops."""
    if len(pos) < 2:
        return np.zeros((2, 0), dtype=np.int64)
    pairs = cKDTree(pos).query_pairs(r=radius, output_type="ndarray")
    edges = np.concatenate([pairs, pairs[:, ::-1]])
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    return edges.T.astype(np.int64)


def make_graphs(split, radius):
    graphs = []
    image_ids = split["cells"]["image_id"].to_numpy()
    for img_id in pd.unique(image_ids):
        idx = np.flatnonzero(image_ids == img_id)
        graphs.append(Data(
            x=torch.from_numpy(split["X"][idx]),
            edge_index=torch.from_numpy(radius_edges(split["pos"][idx], radius)),
            y=torch.from_numpy(split["y"][idx]),
        ))
    return graphs


def merge_graphs(graphs):
    offsets = np.cumsum([0] + [g.num_nodes for g in graphs[:-1]])
    return Data(
        x=torch.cat([g.x for g in graphs]),
        y=torch.cat([g.y for g in graphs]),
        edge_index=torch.cat([g.edge_index + int(o) for g, o in zip(graphs, offsets)], dim=1),
    )


#  TRAINING AND INFERENCE

def train_one_epoch(model, loader, optimizer, device, class_weights):
    model.train()
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    total_loss = total_acc = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out, _ = model(batch)
        n_seeds = batch.batch_size
        loss = ce_loss(out[:n_seeds], batch.y[:n_seeds])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += (out[:n_seeds].argmax(1) == batch.y[:n_seeds]).float().mean().item()
    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def run_inference(model, graphs, device):
    model.eval()
    logits = torch.cat([model(g.to(device))[0].cpu() for g in graphs])
    return logits.argmax(1).numpy(), F.softmax(logits, dim=1).numpy()


def run_fold(fold_idx, train, test, categories, args, out_dir):
    print(f"\n{'=' * 60}\n  FOLD {fold_idx}  |  train={len(train['y']):,}  test={len(test['y']):,}\n{'=' * 60}")
    device = torch.device(args.device)
    train_graph = merge_graphs(make_graphs(train, args.radius_px))
    test_graphs = make_graphs(test, args.radius_px)
    print(f"  Train graph: {train_graph.num_nodes:,} nodes, {train_graph.edge_index.shape[1]:,} edges")

    model = StellarModel(train["X"].shape[1], args.hid_dim, len(categories)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = NeighborLoader(train_graph, num_neighbors=args.num_neighbors,
                            batch_size=args.node_batch_size, shuffle=True, num_workers=0)
    counts = torch.bincount(train_graph.y, minlength=len(categories)).float()
    class_weights = (counts.sum() / (len(categories) * counts.clamp(min=1))).to(device)

    curve = []
    t_train = time.time()
    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(model, loader, optimizer, device, class_weights)
        curve.append({"epoch": epoch + 1, "loss": loss, "train_acc": acc})
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d}/{args.epochs}  loss={loss:.4f}  train_acc={acc:.4f}")
    train_time = time.time() - t_train
    (out_dir / "learning_curves").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve).to_csv(out_dir / "learning_curves" / f"fold_{fold_idx}.csv", index=False)

    t_pred = time.time()
    pred_codes, probs = run_inference(model, test_graphs, device)
    predict_time = time.time() - t_pred
    true_labels = test["cells"]["label"].to_numpy()
    pred_labels = categories[pred_codes]

    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", labels=categories, zero_division=0)
    wf1 = f1_score(true_labels, pred_labels, average="weighted", labels=categories, zero_division=0)
    print(f"\n  Fold {fold_idx}: Acc={acc:.4f}  MacroF1={macro_f1:.4f}  WF1={wf1:.4f}")
    print(classification_report(true_labels, pred_labels, labels=categories, zero_division=0))

    pd.DataFrame({
        "image_id": test["cells"]["image_id"].to_numpy(),
        "cell_id": test["cells"]["cell_id"].to_numpy(),
        "fold": fold_idx,
        "true_phenotype": true_labels,
        "predicted_phenotype": pred_labels,
        "confidence": probs.max(axis=1),
    }).to_csv(out_dir / "level3" / f"predictions_{fold_idx}.csv", index=False)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "models" / f"model_fold{fold_idx}.pth")
    return {"fold": fold_idx, "accuracy": acc, "macro_f1": macro_f1, "wf1": wf1,
            "train_time_s": train_time, "predict_time_s": predict_time}


def save_summary(metrics, out_dir, dataset):
    df = pd.DataFrame(metrics)
    print(f"\n{'=' * 60}\n  SUMMARY — {dataset}\n{'=' * 60}")
    print(df.to_string(index=False))
    if len(df) > 1:
        print(f"\n  Mean MacroF1 : {df['macro_f1'].mean():.4f} ± {df['macro_f1'].std():.4f}")
    df.to_csv(out_dir / "fold_metrics_summary.csv", index=False)
    with open(out_dir / "level3" / "fold_times.txt", "w") as f:
        for r in df.itertuples():
            f.write(f"fold_{r.fold}: {r.total_time_s:.2f}s (train={r.train_time_s:.2f}s, "
                    f"predict={r.predict_time_s:.2f}s)\n")
        f.write(f"total: {df['total_time_s'].sum():.2f}s\n")
        f.write(f"total_train: {df['train_time_s'].sum():.2f}s\n")
        f.write(f"total_predict: {df['predict_time_s'].sum():.2f}s\n")
        f.write(f"mean_fold: {df['total_time_s'].mean():.2f}s\n")


#  MAIN

def main():
    p = argparse.ArgumentParser(description="Supervised STELLAR GNN, config-driven")
    p.add_argument("--dataset-path", required=True,
                   help="Dataset directory (markers.txt, quantification/, raw_images/, segmentation/)")
    p.add_argument("--config", required=True, help="Path to stellar.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--folds-dir", default=None, help="Fold CSV directory, absolute or relative to --dataset-path")
    p.add_argument("--include-validation", action=argparse.BooleanOptionalAction, default=None,
                   help="Add fold_<i>_validation.csv cells to the training set")
    p.add_argument("--fold", type=int, default=None, help="Single fold (0-based); default: all")
    p.add_argument("--distance-threshold", type=float, default=None,
                   help="Neighbourhood radius in pixels; overrides neighbourhood_um / pixel_size_um")
    p.add_argument("--hid-dim", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--node-batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    cli = p.parse_args()

    data_root = Path(cli.dataset_path)
    cfg = load_config(cli.config, data_root.name)
    pick = lambda value, key, default=None: value if value is not None else cfg.get(key, default)

    if cli.distance_threshold is not None:
        radius_px = cli.distance_threshold
    elif "pixel_size_um" in cfg and "neighbourhood_um" in cfg:
        radius_px = cfg["neighbourhood_um"] / cfg["pixel_size_um"]
    else:
        raise ValueError(f"{cli.config}: set neighbourhood_um and pixel_size_um for {data_root.name}")

    args = argparse.Namespace(
        radius_px=radius_px,
        hid_dim=pick(cli.hid_dim, "hid_dim"),
        epochs=pick(cli.epochs, "epochs"),
        lr=pick(cli.lr, "lr"),
        weight_decay=pick(cli.weight_decay, "weight_decay"),
        node_batch_size=pick(cli.node_batch_size, "node_batch_size"),
        num_neighbors=cfg.get("num_neighbors", [10]),
        device=pick(cli.device, "device", "cuda:0"),
    )
    out_dir = Path(cli.output_dir) / "stellar_supervised"
    (out_dir / "level3").mkdir(parents=True, exist_ok=True)
    print(f"\nStellar — {data_root.name}  radius={radius_px:.2f}px"
          + (f" ({cfg['neighbourhood_um']} um at {cfg['pixel_size_um']} um/px)" if cli.distance_threshold is None else "")
          + f"  hid_dim={args.hid_dim}  epochs={args.epochs}  device={args.device}\n  output: {out_dir}")

    markers = read_markers(data_root)
    exclude = cfg.get("exclude_markers", [])
    unknown = sorted(set(exclude) - set(markers))
    if unknown:
        raise ValueError(f"exclude_markers not in markers.txt: {unknown}")
    channel_idx = [i for i, m in enumerate(markers) if m not in exclude]
    print(f"  Channels: {len(channel_idx)}/{len(markers)} kept (excluded: {exclude})")

    folds_dir = data_root / pick(cli.folds_dir, "folds_dir",
                                 "quantification/processed/kfolds_StratifiedGroupKFold_level3")
    meta, folds, img_paths, seg_paths = load_benchmark_data(
        data_root, folds_dir, cfg.get("exclude_labels", []),
        pick(cli.include_validation, "include_validation", False))
    missing = sorted(set(meta["image_id"]) - (set(img_paths) & set(seg_paths)))
    if missing:
        raise FileNotFoundError(f"No image or segmentation for {len(missing)} images, e.g. {missing[:3]}")

    features = {img_id: image_features(img_paths[img_id], seg_paths[img_id], channel_idx)
                for img_id in tqdm(sorted(img_paths), desc="Features")}
    categories = np.array(sorted(meta["label"].unique()))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    print(f"Classes ({len(categories)}): {categories.tolist()}")

    cell_keys = pd.Series(list(zip(meta["image_id"], meta["cell_id"])), index=meta.index)
    metrics = []
    for fi in ([cli.fold] if cli.fold is not None else range(len(folds))):
        t0 = time.time()
        train, mean, std = build_split(meta[cell_keys.isin(folds[fi]["train"])], features, cat_to_idx)
        test, _, _ = build_split(meta[cell_keys.isin(folds[fi]["test"])], features, cat_to_idx, mean, std)
        m = run_fold(fi, train, test, categories, args, out_dir)
        m["total_time_s"] = time.time() - t0
        metrics.append(m)
    save_summary(metrics, out_dir, data_root.name)


if __name__ == "__main__":
    main()
