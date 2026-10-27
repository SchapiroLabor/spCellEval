#!/usr/bin/env python3
"""
run_stellar.py
==============
Unified, config-driven Stellar (supervised GNN) pipeline (dataset-agnostic).

Per-dataset settings (distance threshold, excluded markers/labels) live in
src/methods/configs/stellar.json, keyed by the dataset folder name.

Architecture:
  - Per-image spatial neighbour graphs (distance threshold on cell centroids)
  - StellarModel: Linear(in->hid) + ReLU + SAGEConv(hid->hid) + Linear(hid->classes)
  - Purely supervised cross-entropy with inverse-frequency class weights
  - NeighborLoader mini-batch training on merged graph
  - Per-graph inference
  - Train-split mean/std are reused as-is for the test split (no leakage)

Env:    nix develop -> micromamba activate stellar
Output: <output_dir>/dt{threshold}/

Usage:
    # All 5 folds:
    python run_stellar.py --dataset-path /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan \
        --config src/methods/configs/stellar.json --output-dir /home/juliaoesterle/results/stellar/IMMUcan

    # Single fold, quick smoke test:
    python run_stellar.py --dataset-path /home/juliaoesterle/data/phenotyping_benchmark/cHL_2_MIBI \
        --config src/methods/configs/stellar.json --output-dir /home/juliaoesterle/results/stellar/cHL_2_MIBI \
        --fold 0 --epochs 2 --device cpu
"""

import os
import sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage.measure import regionprops
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

import anndata


#  CONFIG LOADING

def _rebase_local_paths(obj, prefix="/home/juliaoesterle", env_var="SPCELLEVAL_ROOT"):
    """stellar.json hardcodes absolute paths rooted at beauty (/home/juliaoesterle).
    Set SPCELLEVAL_ROOT to re-root every such path onto another machine's
    data root (e.g. Helix's /mnt/sds-hd/sd22c003) without editing the JSON."""
    root = os.environ.get(env_var)
    if not root:
        return obj
    if isinstance(obj, str):
        return root.rstrip('/') + obj[len(prefix):] if obj.startswith(prefix) else obj
    if isinstance(obj, dict):
        return {k: _rebase_local_paths(v, prefix, env_var) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rebase_local_paths(v, prefix, env_var) for v in obj]
    return obj


def load_config(config_path: str, dataset: str) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    merged = dict(cfg.get("defaults", {}))
    merged.update(cfg["datasets"].get(dataset, {}))
    merged["dataset"] = dataset
    return _rebase_local_paths(merged)


#  MODEL

class StellarModel(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int, num_classes: int):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, hid_dim)
        self.graph_conv   = SAGEConv(hid_dim, hid_dim)
        self.fc_net       = nn.Linear(hid_dim, num_classes)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        feat     = F.relu(self.input_linear(x))
        out_feat = self.graph_conv(feat, edge_index)
        out      = self.fc_net(out_feat)
        return out, out_feat


#  GRAPH CONSTRUCTION

def get_edges(pos: np.ndarray, distance_threshold: float) -> np.ndarray:
    """Distance-threshold spatial graph. Returns edge_index (2, E)."""
    if len(pos) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    diff  = pos[:, None, :] - pos[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    adj   = dists <= distance_threshold
    np.fill_diagonal(adj, False)
    return np.array(np.where(adj), dtype=np.int64)


def make_graph_list(adata: anndata.AnnData, distance_threshold: float) -> List[Data]:
    graphs = []
    for sample_id in tqdm(adata.obs["sample_id"].cat.categories,
                          desc="Building graphs", leave=False):
        sel      = (adata.obs["sample_id"] == sample_id).values
        cell_ids = np.array(adata.obs[sel].index)
        pos      = adata.obs[sel][["Pos_X", "Pos_Y"]].values.astype(np.float32)
        exprs    = adata.layers["exprs"][sel].astype(np.float32)
        y        = adata.obs[sel]["cell_label_idx"].values.astype(np.int64)
        edges    = get_edges(pos, distance_threshold)
        graphs.append(Data(
            x          = torch.FloatTensor(exprs),
            edge_index = torch.LongTensor(edges),
            y          = torch.LongTensor(y),
            cell_ids   = cell_ids,
        ))
    return graphs


#  TRAINING AND EVALUATION

def compute_class_weights(graphs: List[Data], num_classes: int, device) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for g in graphs:
        for c in range(num_classes):
            counts[c] += (g.y == c).sum()
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights.to(device)


def train_one_epoch(model, loader, optimizer, device, class_weights):
    model.train()
    ce_loss    = nn.CrossEntropyLoss(weight=class_weights)
    total_loss = total_acc = n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out, _ = model(batch)
        n_seeds = batch.batch_size
        loss = ce_loss(out[:n_seeds], batch.y[:n_seeds])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc  += (out[:n_seeds].argmax(1) == batch.y[:n_seeds]).float().mean().item()
        n += 1
    return total_loss / n, total_acc / n


@torch.no_grad()
def run_inference(model, graphs, device):
    model.eval()
    all_logits = []
    for g in graphs:
        out, _ = model(g.to(device))
        all_logits.append(out.cpu())
    logits = torch.cat(all_logits, dim=0)
    probs  = F.softmax(logits, dim=1).numpy()
    codes  = logits.argmax(1).numpy()
    return codes, probs


def merge_graphs_for_loader(graphs: List[Data]) -> Data:
    all_x, all_y, all_ei = [], [], []
    offset = 0
    for g in graphs:
        all_x.append(g.x)
        all_y.append(g.y)
        all_ei.append(g.edge_index + offset)
        offset += g.x.shape[0]
    return Data(
        x          = torch.cat(all_x, dim=0),
        y          = torch.cat(all_y, dim=0),
        edge_index = torch.cat(all_ei, dim=1),
    )


def build_adata_from_arrays(
    expressions, pos_x, pos_y, sample_ids, cell_ids, cell_types,
    categories, cat_to_idx, mean=None, std=None,
) -> Tuple[anndata.AnnData, np.ndarray, np.ndarray]:
    """
    mean/std: if given (the training split's stats), applied as-is -- this is
    the test-split call. If None (the train-split call), computed from X here
    and returned so the caller can pass them into the matching test call.
    (Each split must never compute its own normalisation stats independently
    -- that leaks test-set statistics into the split boundary.)
    """
    X = expressions.astype(np.float32)
    if mean is None or std is None:
        mean = X.mean(axis=0, keepdims=True)
        std  = X.std(axis=0, keepdims=True)
    X = (X - mean) / (std + 1e-8)

    obs = pd.DataFrame({
        "sample_id":      pd.Categorical(sample_ids),
        "cell_id":        cell_ids,
        "Pos_X":          pos_x.astype(np.float32),
        "Pos_Y":          pos_y.astype(np.float32),
        "cell_labels":    pd.Categorical(cell_types, categories=categories),
        "cell_label_idx": [cat_to_idx[l] for l in cell_types],
    })
    obs.index = [f"{s}_{c}" for s, c in zip(sample_ids, cell_ids)]

    adata = anndata.AnnData(X=X, obs=obs)
    adata.layers["exprs"] = X.copy()
    print(f"  -> {adata.n_obs:,} cells | {obs['sample_id'].nunique()} images")
    return adata, mean, std


#  DATA LOADING

def get_channel_info(cfg, data_root):
    excl = set(cfg.get("excluded_markers", []))
    with open(data_root / "markers.txt") as f:
        all_names = [l.strip() for l in f if l.strip()]
    keep_idx = [i for i, n in enumerate(all_names) if n not in excl]
    print(f"  Channels: {len(keep_idx)}/{len(all_names)} kept (excluded: {excl})")
    return all_names, keep_idx


def _file_stem(name):
    for ext in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def find_image(img_dir, img_id):
    files = [p for p in img_dir.iterdir() if not p.name.startswith(".")]
    exact = [p for p in files if _file_stem(p.name) == img_id]
    if exact:
        return exact[0]
    prefixed = [p for p in files if _file_stem(p.name).startswith(f"{img_id}_")]
    return prefixed[0] if len(prefixed) == 1 else None


def find_segmentation(seg_dir, img_id):
    for ext in (".tiff", ".tif"):
        if (seg_dir / f"{img_id}{ext}").exists():
            return seg_dir / f"{img_id}{ext}"
    if (seg_dir / img_id).is_dir():
        seg_files = sorted(f for f in (seg_dir / img_id).rglob("segmentationMap.tif")
                           if not f.name.startswith("."))
        if seg_files:
            return seg_files[0]
    return None


def extract_expressions(img_path, mask, keep_idx, image_scale=1.0):
    img      = tifffile.imread(str(img_path)).astype(np.float32)
    img_keep = img[keep_idx]
    img_keep /= image_scale
    C        = img_keep.shape[0]

    props = regionprops(mask)
    if not props:
        return np.array([]), np.array([]), np.array([])

    cell_ids  = np.array([p.label for p in props], dtype=np.int32)
    centroids = np.array([p.centroid for p in props], dtype=np.float32)

    flat_mask     = mask.ravel()
    sort_idx      = np.argsort(flat_mask, kind="stable")
    sorted_labels = flat_mask[sort_idx]
    boundaries    = np.searchsorted(sorted_labels, np.arange(1, int(cell_ids.max()) + 2))
    img_flat      = img_keep.reshape(C, -1)

    expressions = np.zeros((len(cell_ids), C), dtype=np.float32)
    for i, cid in enumerate(cell_ids):
        start   = boundaries[cid - 1]
        end     = boundaries[cid]
        pix_idx = sort_idx[start:end]
        if len(pix_idx) > 0:
            expressions[i] = img_flat[:, pix_idx].mean(axis=1)
    return cell_ids, expressions, centroids


def build_anndata(subset_df, img_paths, seg_paths, keep_idx, image_scale, categories,
                  cat_to_idx, excluded_labels, desc="Loading", mean=None, std=None):
    all_exprs, all_sample, all_cell_id = [], [], []
    all_pos_x, all_pos_y, all_labels   = [], [], []

    for img_id in tqdm(sorted(subset_df["sample_id"].unique()), desc=desc):
        if img_id not in img_paths:
            print(f"  [WARN] No image for {img_id}, skipping")
            continue
        if img_id not in seg_paths:
            print(f"  [WARN] No segmentation for {img_id}, skipping")
            continue
        mask = tifffile.imread(str(seg_paths[img_id])).astype(np.int32)

        cell_ids, exprs, centroids = extract_expressions(img_paths[img_id], mask, keep_idx, image_scale)
        if len(cell_ids) == 0:
            continue

        img_df = subset_df[subset_df["sample_id"] == img_id].copy()
        img_df["cell_id"] = img_df["cell_id"].astype(int)
        cid_to_row = {int(cid): i for i, cid in enumerate(cell_ids)}

        for _, row in img_df.iterrows():
            cid = int(row["cell_id"])
            ct  = str(row["cell_type"])
            if ct in excluded_labels or ct not in cat_to_idx or cid not in cid_to_row:
                continue
            ridx = cid_to_row[cid]
            all_exprs.append(exprs[ridx])
            all_sample.append(str(img_id))
            all_cell_id.append(cid)
            all_pos_x.append(float(centroids[ridx, 1]))
            all_pos_y.append(float(centroids[ridx, 0]))
            all_labels.append(ct)

    if not all_exprs:
        raise RuntimeError("No cells loaded - check paths and cell_id matching")

    return build_adata_from_arrays(
        expressions=np.stack(all_exprs, axis=0),
        pos_x=np.array(all_pos_x), pos_y=np.array(all_pos_y),
        sample_ids=all_sample, cell_ids=all_cell_id, cell_types=all_labels,
        categories=categories, cat_to_idx=cat_to_idx, mean=mean, std=std,
    )


def get_folds_and_meta(data_root, folds_dir, include_validation=False):
    """
    Returns meta_df (sample_id, cell_id, cell_type) and folds (cell-level:
    sets of (sample_id, cell_id) tuples read from the per-fold
    fold_{i}_train.csv/fold_{i}_test.csv files), img_paths, seg_paths.
    """
    quant_csv = data_root / "quantification" / "processed" / f"{data_root.name}_quantification.csv"
    meta_df = pd.read_csv(quant_csv, usecols=["sample_id", "cell_id", "cell_type"])
    meta_df["sample_id"] = meta_df["sample_id"].astype(str).str.replace(".csv", "", regex=False)
    print(f"{len(meta_df):,} cells, {meta_df['sample_id'].nunique()} images")

    def load_keys(path):
        df = pd.read_csv(path, usecols=["sample_id", "cell_id"])
        df["sample_id"] = df["sample_id"].astype(str).str.replace(".csv", "", regex=False)
        return set(zip(df["sample_id"], df["cell_id"]))

    folds, fold_i = [], 1
    while (folds_dir / f"fold_{fold_i}_train.csv").exists():
        train_keys = load_keys(folds_dir / f"fold_{fold_i}_train.csv")
        if include_validation and (folds_dir / f"fold_{fold_i}_validation.csv").exists():
            train_keys |= load_keys(folds_dir / f"fold_{fold_i}_validation.csv")
        folds.append({
            "train": train_keys,
            "test":  load_keys(folds_dir / f"fold_{fold_i}_test.csv"),
        })
        fold_i += 1
    print(f"{len(folds)} folds (cell-level) from {folds_dir}")

    img_dir = data_root / "raw_images" / "multistack_tiffs"
    seg_dir = data_root / "segmentation"
    img_paths, seg_paths = {}, {}
    for img_id in sorted(meta_df["sample_id"].unique()):
        img_path = find_image(img_dir, img_id)
        if img_path is not None:
            img_paths[img_id] = img_path
        seg_path = find_segmentation(seg_dir, img_id)
        if seg_path is not None:
            seg_paths[img_id] = seg_path
    print(f"{len(img_paths)} images, {len(seg_paths)} segmentation maps found")
    return meta_df, folds, img_paths, seg_paths


#  FOLD LOOP

def run_fold(fold_idx, train_adata, test_adata, categories, args, output_dir):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx}  |  train={train_adata.n_obs:,}  test={test_adata.n_obs:,}")
    print(f"{'='*60}")
    device = torch.device(args.device)

    train_graphs = make_graph_list(train_adata, args.distance_threshold)
    test_graphs  = make_graph_list(test_adata,  args.distance_threshold)

    n_features = train_adata.X.shape[1]
    model = StellarModel(n_features, args.hid_dim, len(categories)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    merged_train = merge_graphs_for_loader(train_graphs)
    print(f"  Merged train graph: {merged_train.x.shape[0]:,} nodes, "
          f"{merged_train.edge_index.shape[1]:,} edges")
    loader = NeighborLoader(
        merged_train,
        num_neighbors=args.num_neighbors,
        batch_size=args.node_batch_size,
        shuffle=True, num_workers=0,
    )

    class_weights = compute_class_weights([merged_train], len(categories), device)
    print(f"  Class weights: min={class_weights.min():.2f} "
          f"max={class_weights.max():.2f} mean={class_weights.mean():.2f}")

    learning_curve = []
    t_train = time.time()
    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(model, loader, optimizer, device, class_weights)
        if (epoch + 1) % args.eval_every == 0 or epoch == 0 or epoch == args.epochs - 1:
            ep_codes, _ = run_inference(model, test_graphs, device)
            ep_preds    = categories[ep_codes]
            ep_true     = np.concatenate([
                test_adata.obs.loc[g.cell_ids, "cell_labels"].values for g in test_graphs])
            ep_acc = accuracy_score(ep_true, ep_preds)
            ep_f1  = f1_score(ep_true, ep_preds, average="macro", labels=categories, zero_division=0)
            ep_wf1 = f1_score(ep_true, ep_preds, average="weighted", labels=categories, zero_division=0)
            learning_curve.append({"epoch": epoch + 1, "loss": loss, "train_acc": acc,
                                    "accuracy": ep_acc, "macro_f1": ep_f1, "wf1": ep_wf1})
            print(f"  Epoch {epoch+1:3d}/{args.epochs}  loss={loss:.4f}  "
                  f"train_acc={acc:.4f}  val_F1={ep_f1:.4f}  val_Acc={ep_acc:.4f}")
        else:
            learning_curve.append({"epoch": epoch + 1, "loss": loss, "train_acc": acc,
                                    "accuracy": None, "macro_f1": None, "wf1": None})

    train_time = time.time() - t_train
    print(f"  Training: {train_time:.1f}s")

    lc_dir = output_dir / "learning_curves"
    lc_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(learning_curve).to_csv(lc_dir / f"fold_{fold_idx}.csv", index=False)

    predict_start = time.time()
    pred_codes, probs = run_inference(model, test_graphs, device)
    predict_time = time.time() - predict_start
    print(f"  Prediction: {predict_time:.1f}s")
    pred_labels = categories[pred_codes]
    true_labels = np.concatenate([
        test_adata.obs.loc[g.cell_ids, "cell_labels"].values for g in test_graphs])

    acc      = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", labels=categories, zero_division=0)
    wf1      = f1_score(true_labels, pred_labels, average="weighted", labels=categories, zero_division=0)
    print(f"\n  Fold {fold_idx}: Acc={acc:.4f}  MacroF1={macro_f1:.4f}  WF1={wf1:.4f}")
    print(classification_report(true_labels, pred_labels, labels=categories, zero_division=0))

    pred_dir = output_dir / "stellar_supervised" / "level3"
    pred_dir.mkdir(parents=True, exist_ok=True)
    test_obs = pd.concat([test_adata.obs.loc[g.cell_ids] for g in test_graphs])
    pd.DataFrame({
        "image_id":            test_obs["sample_id"].values,
        "cell_id":             test_obs["cell_id"].values,
        "fold":                fold_idx,
        "true_phenotype":      true_labels,
        "predicted_phenotype": pred_labels,
        "confidence":          probs.max(axis=1),
    }).to_csv(pred_dir / f"predictions_{fold_idx}.csv", index=False)
    print(f"  Saved -> {pred_dir}/predictions_{fold_idx}.csv")

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / f"model_fold{fold_idx}.pth")

    return {"fold": fold_idx, "accuracy": acc, "macro_f1": macro_f1,
            "wf1": wf1, "train_time_s": train_time, "predict_time_s": predict_time}


#  MAIN

def main():
    parser = argparse.ArgumentParser(description="Supervised Stellar GNN — config-driven")
    parser.add_argument("--dataset-path", required=True,
                        help="Dataset directory (markers.txt, quantification/, raw_images/, segmentation/)")
    parser.add_argument("--folds-dir", default=None,
                        help="k-fold directory, absolute or relative to --dataset-path")
    parser.add_argument("--include-validation", action=argparse.BooleanOptionalAction, default=None,
                        help="Add fold_{i}_validation.csv cells to the training set")
    parser.add_argument("--config",  required=True, help="Path to stellar.json")
    parser.add_argument("--fold", type=int, default=None, help="Single fold (0-based). Default: all.")
    # optional overrides (default None -> fall back to config)
    parser.add_argument("--hid-dim",            type=int,   default=None)
    parser.add_argument("--distance-threshold", type=float, default=None)
    parser.add_argument("--epochs",             type=int,   default=None)
    parser.add_argument("--lr",                 type=float, default=None)
    parser.add_argument("--weight-decay",       type=float, default=None)
    parser.add_argument("--eval-every",         type=int,   default=None)
    parser.add_argument("--node-batch-size",    type=int,   default=None)
    parser.add_argument("--device",             type=str,   default=None)
    parser.add_argument("--output-dir",         type=str,   required=True)
    cli = parser.parse_args()

    data_root = Path(cli.dataset_path)
    cfg = load_config(cli.config, data_root.name)

    def pick(cli_val, key, default=None):
        if cli_val is not None:
            return cli_val
        return cfg.get(key, default)

    args = argparse.Namespace(
        dataset            = data_root.name,
        fold               = cli.fold,
        hid_dim            = pick(cli.hid_dim, "hid_dim"),
        distance_threshold = pick(cli.distance_threshold, "distance_threshold"),
        epochs             = pick(cli.epochs, "epochs"),
        lr                 = pick(cli.lr, "lr"),
        weight_decay       = pick(cli.weight_decay, "weight_decay"),
        eval_every         = pick(cli.eval_every, "eval_every"),
        node_batch_size    = pick(cli.node_batch_size, "node_batch_size"),
        num_neighbors      = cfg.get("num_neighbors", [10]),
        device             = pick(cli.device, "device", "cuda:0"),
    )

    folds_dir  = data_root / pick(cli.folds_dir, "folds_dir",
                                  "quantification/processed/kfolds_StratifiedGroupKFold_level3")
    dt_str     = f"dt{args.distance_threshold:.2f}".replace(".", "_")
    output_dir = Path(cli.output_dir) / dt_str
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStellar — dataset={args.dataset}  distance_threshold={args.distance_threshold}px")
    print(f"  hid_dim={args.hid_dim}  epochs={args.epochs}  device={args.device}")
    print(f"  output_dir: {output_dir}\n")

    excluded_labels = set(cfg.get("excluded_labels", []))
    image_scale     = cfg.get("image_scale", 1.0)

    meta_df, folds, img_paths, seg_paths = get_folds_and_meta(
        data_root, folds_dir, pick(cli.include_validation, "include_validation", False))
    _, keep_idx = get_channel_info(cfg, data_root)

    valid_types = sorted(ct for ct in meta_df["cell_type"].unique()
                         if ct not in excluded_labels)
    categories = np.array(valid_types)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    print(f"Classes ({len(categories)}): {categories.tolist()}")

    cell_keys   = pd.Series(list(zip(meta_df["sample_id"], meta_df["cell_id"])), index=meta_df.index)
    fold_range  = [args.fold] if args.fold is not None else range(len(folds))
    all_metrics = []
    for fi in fold_range:
        t0        = time.time()
        train_df  = meta_df[cell_keys.isin(folds[fi]["train"])].copy()
        test_df   = meta_df[cell_keys.isin(folds[fi]["test"])].copy()
        print(f"\nFold {fi}: {len(train_df):,} train / {len(test_df):,} test cells")
        train_adata, tr_mean, tr_std = build_anndata(
            train_df, img_paths, seg_paths, keep_idx, image_scale, categories,
            cat_to_idx, excluded_labels, desc=f"F{fi} train")
        test_adata, _, _ = build_anndata(
            test_df, img_paths, seg_paths, keep_idx, image_scale, categories,
            cat_to_idx, excluded_labels, desc=f"F{fi} test",
            mean=tr_mean, std=tr_std)
        m = run_fold(fi, train_adata, test_adata, categories, args, output_dir)
        m["total_time_s"] = time.time() - t0
        all_metrics.append(m)

    _print_and_save_summary(all_metrics, args, output_dir)


def _print_and_save_summary(all_metrics, args, output_dir):
    print(f"\n{'='*60}\n  SUMMARY — {args.dataset}\n{'='*60}")
    df = pd.DataFrame(all_metrics)
    time_cols = [c for c in ("train_time_s", "predict_time_s", "total_time_s") if c in df.columns]
    print(df[["fold", "accuracy", "macro_f1", "wf1"] + time_cols].to_string(index=False))
    if len(df) > 1:
        print(f"\n  Mean Acc     : {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
        print(f"  Mean MacroF1 : {df['macro_f1'].mean():.4f} ± {df['macro_f1'].std():.4f}")
        print(f"  Mean WF1     : {df['wf1'].mean():.4f} ± {df['wf1'].std():.4f}")
    df.to_csv(output_dir / "fold_metrics_summary.csv", index=False)
    pred_dir = output_dir / "stellar_supervised" / "level3"
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / "fold_times.txt", "w") as f:
        for _, r in df.iterrows():
            f.write(f"fold_{int(r['fold'])}: {r['total_time_s']:.2f}s "
                    f"(train={r['train_time_s']:.2f}s, predict={r['predict_time_s']:.2f}s)\n")
        f.write(f"total: {df['total_time_s'].sum():.2f}s\n")
        f.write(f"total_train: {df['train_time_s'].sum():.2f}s\n")
        f.write(f"total_predict: {df['predict_time_s'].sum():.2f}s\n")
        f.write(f"mean_fold: {df['total_time_s'].mean():.2f}s\n")
    print(f"\nSummary -> {output_dir / 'fold_metrics_summary.csv'}")


if __name__ == "__main__":
    main()
