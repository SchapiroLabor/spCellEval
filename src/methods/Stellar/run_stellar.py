#!/usr/bin/env python3
"""
run_stellar.py
==============
Unified, config-driven Stellar (supervised GNN) pipeline for IMMUcan and cHL_2_MIBI.

All dataset-specific constants (paths, channels, label maps, excluded labels)
live in the per-method config JSON (src/methods/configs/stellar.json) and are
selected with --dataset.

Architecture:
  - Per-image spatial neighbour graphs (distance threshold on cell centroids)
  - StellarModel: Linear(in->hid) + ReLU + SAGEConv(hid->hid) + Linear(hid->classes)
  - Purely supervised cross-entropy with inverse-frequency class weights
  - NeighborLoader mini-batch training on merged graph
  - Per-graph inference

Env:    nix develop -> micromamba activate stellar
Output: <output_root>/dt{threshold}/

Usage:
    # One-time preprocessing (extract expressions once):
    python prepare_dataset.py --dataset immucan --config src/methods/configs/stellar.json

    # Fast path - load prepared AnnData, subset per fold:
    python run_stellar.py --dataset immucan --config src/methods/configs/stellar.json \
        --prepared <output_root>/prepared/immucan_cells.h5ad

    # IMMUcan - all 5 image-level folds (inline extraction, no prepared file):
    python run_stellar.py --dataset immucan --config src/methods/configs/stellar.json

    # cHL - all 5 cell-level folds:
    python run_stellar.py --dataset chl --config src/methods/configs/stellar.json

    # Single fold, quick smoke test:
    python run_stellar.py --dataset immucan --config src/methods/configs/stellar.json \
        --fold 0 --epochs 2 --device cpu

    # Full run on GPU:
    python run_stellar.py --dataset immucan --config src/methods/configs/stellar.json \
        --device cuda:0
"""

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


# config loading 

def load_config(config_path: str, dataset: str) -> dict:
    """Load per-method config, merge defaults with the selected dataset block."""
    with open(config_path) as f:
        cfg = json.load(f)
    if dataset not in cfg["datasets"]:
        raise ValueError(f"Dataset '{dataset}' not in {config_path}. "
                         f"Available: {list(cfg['datasets'])}")
    merged = dict(cfg.get("defaults", {}))
    merged.update(cfg["datasets"][dataset])
    merged["dataset"] = dataset
    return merged


# model

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


# graph construction 
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


# Training and Evaluation

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
    categories, cat_to_idx,
) -> anndata.AnnData:
    X    = expressions.astype(np.float32)
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True)
    X    = (X - mean) / (std + 1e-8)

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
    return adata


# prepared AnnData loading and subsetting
def load_prepared(prepared_path: str) -> Tuple[anndata.AnnData, np.ndarray]:
    """Load prepared <dataset>_cells.h5ad and recover the class category order."""
    adata = anndata.read_h5ad(prepared_path)
    if "categories" in adata.uns:
        categories = np.array(list(adata.uns["categories"]))
    else:
        # fall back to the categorical order stored in obs
        categories = np.array(adata.obs["cell_labels"].cat.categories)
    print(f"Loaded prepared AnnData: {adata.n_obs:,} cells | "
          f"{adata.n_vars} features | {len(categories)} classes")
    return adata, categories


def subset_prepared_by_images(adata: anndata.AnnData, image_ids) -> anndata.AnnData:
    """Image-level fold subset (IMMUcan): keep cells whose sample_id is in the fold."""
    sel = adata.obs["sample_id"].isin(set(image_ids)).values
    return adata[sel].copy()


def subset_prepared_by_cellkeys(adata: anndata.AnnData, keys) -> anndata.AnnData:
    """
    Cell-level fold subset (cHL): keys are obs index strings '{sample_id}_{cell_id}'.
    Missing keys (cells not extracted) are silently dropped.
    """
    present = adata.obs_names.intersection(pd.Index(keys))
    return adata[present].copy()

def load_immucan_image_data(img_name, cfg, paths):
    n_raw     = cfg["n_raw_channels"]
    keep      = [i for i in range(n_raw) if i not in set(cfg["excluded_channels"])]
    scale     = cfg["image_scale"]

    img = np.load(paths["image_dir"] / f"{img_name}.npz")["data"].astype(np.float32)
    img /= scale

    mask       = tifffile.imread(str(paths["segmentation_dir"] / f"{img_name}.tiff")).astype(np.int32)
    raw_labels = np.loadtxt(str(paths["label_dir"] / f"{img_name}.txt"), dtype=np.int32)

    props = regionprops(mask)
    if not props:
        return None, None, None, None

    cell_ids  = np.array([p.label for p in props], dtype=np.int32)
    centroids = np.array([p.centroid for p in props], dtype=np.float32)

    n_cells         = len(cell_ids)
    expressions_all = np.zeros((n_cells, n_raw), dtype=np.float32)
    flat_mask       = mask.ravel()
    sort_idx        = np.argsort(flat_mask, kind="stable")
    sorted_labels   = flat_mask[sort_idx]
    boundaries      = np.searchsorted(sorted_labels, np.arange(1, cell_ids.max() + 2))
    img_flat        = img.reshape(n_raw, -1)

    for i, cid in enumerate(cell_ids):
        start   = boundaries[cid - 1]
        end     = boundaries[cid]
        pix_idx = sort_idx[start:end]
        expressions_all[i] = img_flat[:, pix_idx].mean(axis=1)

    expressions = expressions_all[:, keep]

    cell_types = np.array(
        [raw_labels[cid - 1] if (cid - 1) < len(raw_labels) else -1
         for cid in cell_ids],
        dtype=np.int32,
    )
    return expressions, centroids, cell_ids, cell_types


def build_immucan_anndata(img_names, cfg, paths, label_map, categories,
                          cat_to_idx, excluded_labels, desc="Loading"):
    all_exprs, all_sample, all_cell_id = [], [], []
    all_pos_x, all_pos_y, all_labels   = [], [], []

    for img_name in tqdm(img_names, desc=desc):
        exprs, centroids, cell_ids, cell_types = load_immucan_image_data(img_name, cfg, paths)
        if exprs is None:
            print(f"  Warning: skipping {img_name} (empty mask)")
            continue

        valid = cell_types != -1
        if valid.sum() == 0:
            continue
        exprs, centroids = exprs[valid], centroids[valid]
        cell_ids, cell_types = cell_ids[valid], cell_types[valid]

        known = np.array([label_map[int(ct)] not in excluded_labels for ct in cell_types])
        if known.sum() == 0:
            continue
        exprs, centroids = exprs[known], centroids[known]
        cell_ids, cell_types = cell_ids[known], cell_types[known]

        n = exprs.shape[0]
        all_exprs.append(exprs)
        all_sample.extend([img_name] * n)
        all_cell_id.extend(cell_ids.tolist())
        all_pos_x.extend(centroids[:, 1].tolist())
        all_pos_y.extend(centroids[:, 0].tolist())
        all_labels.extend([label_map[int(ct)] for ct in cell_types])

    return build_adata_from_arrays(
        expressions=np.concatenate(all_exprs, axis=0),
        pos_x=np.array(all_pos_x), pos_y=np.array(all_pos_y),
        sample_ids=all_sample, cell_ids=all_cell_id, cell_types=all_labels,
        categories=categories, cat_to_idx=cat_to_idx,
    )


def get_immucan_folds(folds_json):
    with open(folds_json) as f:
        raw = json.load(f)
    n = sum(1 for k in raw if k.endswith("_train_set"))
    return (
        [raw[f"fold_{i}_train_set"] for i in range(n)],
        [raw[f"fold_{i}_test_set"]  for i in range(n)],
    )


# cHL data loading

def get_chl_channel_info(cfg, paths):
    excl = set(cfg["excluded_markers"])
    sc_dir = paths["single_channel_dir"]
    if sc_dir.exists():
        all_names = sorted(f.stem for f in sc_dir.rglob("*.tiff")
                           if not f.name.startswith("."))
    else:
        with open(paths["markers_txt"]) as f:
            all_names = [l.strip() for l in f if l.strip()]
    keep_idx = [i for i, n in enumerate(all_names) if n not in excl]
    print(f"  [cHL] Channels: {len(keep_idx)}/{len(all_names)} kept (excluded: {excl})")
    return all_names, keep_idx


def load_chl_seg(img_id, paths):
    seg_dir   = paths["segmentation_dir"] / str(img_id)
    seg_files = [f for f in seg_dir.rglob("segmentationMap.tif")
                 if not f.name.startswith(".")]
    if not seg_files:
        raise FileNotFoundError(f"No segmentationMap.tif under {seg_dir}")
    return tifffile.imread(str(seg_files[0])).astype(np.int32)


def extract_chl_expressions(img_path, mask, keep_idx):
    img      = tifffile.imread(str(img_path)).astype(np.float32)
    img_keep = img[keep_idx]
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


def build_chl_anndata(subset_df, img_paths, keep_idx, categories, cat_to_idx,
                      excluded_labels, paths, desc="Loading"):
    all_exprs, all_sample, all_cell_id = [], [], []
    all_pos_x, all_pos_y, all_labels   = [], [], []

    for img_id in tqdm(sorted(subset_df["sample_id"].unique()), desc=desc):
        if img_id not in img_paths:
            print(f"  [WARN] No image for {img_id}, skipping")
            continue
        try:
            mask = load_chl_seg(img_id, paths)
        except FileNotFoundError as e:
            print(f"  [WARN] {e}, skipping")
            continue

        cell_ids, exprs, centroids = extract_chl_expressions(img_paths[img_id], mask, keep_idx)
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
        categories=categories, cat_to_idx=cat_to_idx,
    )


def get_chl_folds_and_meta(cfg, paths):
    meta_df = pd.read_csv(paths["quant_csv"])
    meta_df["sample_id"] = meta_df["sample_id"].astype(str).str.replace(".csv", "", regex=False)
    print(f"[cHL] {len(meta_df):,} cells, {meta_df['sample_id'].nunique()} images")

    with open(paths["folds_json"]) as f:
        fold_data = json.load(f)
    folds = fold_data["folds"]
    print(f"[cHL] {len(folds)} folds (cell-level)")

    img_paths = {}
    for p in sorted(paths["image_dir"].glob("*_stacked.ome.tif")):
        if p.name.startswith("."):
            continue
        img_paths[p.name.replace("_stacked.ome.tif", "")] = p
    print(f"[cHL] {len(img_paths)} images found")
    return meta_df, folds, img_paths


# FOLD LOOP  

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


# Main 

def main():
    parser = argparse.ArgumentParser(description="Supervised Stellar GNN — config-driven")
    parser.add_argument("--dataset", choices=["immucan", "chl"], required=True)
    parser.add_argument("--config",  required=True, help="Path to stellar.json")
    parser.add_argument("--fold", type=int, default=None, help="Single fold (0-based). Default: all.")
    parser.add_argument("--prepared", type=str, default=None,
                        help="Path to a prepared <dataset>_cells.h5ad from prepare_dataset.py. "
                             "If given, expressions are loaded from it and only subset per fold "
                             "(fast). If omitted, expressions are extracted from images inline.")
    # optional overrides (default None -> fall back to config)
    parser.add_argument("--hid-dim",            type=int,   default=None)
    parser.add_argument("--distance-threshold", type=float, default=None)
    parser.add_argument("--epochs",             type=int,   default=None)
    parser.add_argument("--lr",                 type=float, default=None)
    parser.add_argument("--weight-decay",       type=float, default=None)
    parser.add_argument("--eval-every",         type=int,   default=None)
    parser.add_argument("--node-batch-size",    type=int,   default=None)
    parser.add_argument("--device",             type=str,   default=None)
    parser.add_argument("--output-dir",         type=str,   default=None)
    cli = parser.parse_args()

    cfg = load_config(cli.config, cli.dataset)

    # CLI overrides config; config overrides nothing else
    def pick(cli_val, key, default=None):
        if cli_val is not None:
            return cli_val
        return cfg.get(key, default)

    # Resolve hyperparameters into a single namespace used downstream
    args = argparse.Namespace(
        dataset            = cli.dataset,
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

    data_root = Path(cfg["data_root"])
    dt_str    = f"dt{args.distance_threshold:.2f}".replace(".", "_")
    out_base  = Path(cli.output_dir) if cli.output_dir else Path(cfg["output_root"])
    output_dir = out_base / dt_str
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStellar — dataset={args.dataset}  distance_threshold={args.distance_threshold}px")
    print(f"  hid_dim={args.hid_dim}  epochs={args.epochs}  device={args.device}")
    print(f"  output_dir: {output_dir}\n")

    excluded_labels = set(cfg["excluded_labels"])

    #  Fast path: load prepared AnnData once, subset per fold 
    if cli.prepared is not None:
        adata, categories = load_prepared(cli.prepared)

        if args.dataset == "immucan":
            train_folds, test_folds = get_immucan_folds(data_root / cfg["folds_json"])
            fold_range  = [args.fold] if args.fold is not None else range(len(train_folds))
            all_metrics = []
            for fi in fold_range:
                t0 = time.time()
                train_adata = subset_prepared_by_images(adata, train_folds[fi])
                test_adata  = subset_prepared_by_images(adata, test_folds[fi])
                print(f"\n[prepared] Fold {fi}: "
                      f"{train_adata.n_obs:,} train / {test_adata.n_obs:,} test cells")
                m = run_fold(fi, train_adata, test_adata, categories, args, output_dir)
                m["total_time_s"] = time.time() - t0
                all_metrics.append(m)
        else:  # chl — cell-level folds reference rows of the quant CSV
            paths = {
                "quant_csv":  data_root / cfg["quant_csv"],
                "folds_json": data_root / cfg["folds_json"],
                "image_dir":  data_root / cfg["image_dir"],
            }
            meta_df, folds, _ = get_chl_folds_and_meta(cfg, paths)
            fold_range  = [args.fold] if args.fold is not None else range(len(folds))
            all_metrics = []
            for fi in fold_range:
                t0        = time.time()
                train_idx = np.array(folds[fi]["train"])
                test_idx  = np.array(folds[fi]["test"])
                train_df  = meta_df.iloc[train_idx]
                test_df   = meta_df.iloc[test_idx]
                # build obs-index keys '{sample_id}_{cell_id}' to subset prepared adata
                train_keys = [f"{s}_{int(c)}" for s, c in
                              zip(train_df["sample_id"], train_df["cell_id"])]
                test_keys  = [f"{s}_{int(c)}" for s, c in
                              zip(test_df["sample_id"], test_df["cell_id"])]
                train_adata = subset_prepared_by_cellkeys(adata, train_keys)
                test_adata  = subset_prepared_by_cellkeys(adata, test_keys)
                print(f"\n[prepared] Fold {fi}: "
                      f"{train_adata.n_obs:,} train / {test_adata.n_obs:,} test cells")
                m = run_fold(fi, train_adata, test_adata, categories, args, output_dir)
                m["total_time_s"] = time.time() - t0
                all_metrics.append(m)

        _print_and_save_summary(all_metrics, args, output_dir)
        return

    if args.dataset == "immucan":
        paths = {
            "image_dir":        data_root / cfg["image_dir"],
            "label_dir":        data_root / cfg["label_dir"],
            "segmentation_dir": data_root / cfg["segmentation_dir"],
        }
        label_map  = {int(k): v for k, v in cfg["label_map"].items()}
        categories = np.array(sorted(v for v in label_map.values() if v not in excluded_labels))
        cat_to_idx = {c: i for i, c in enumerate(categories)}

        train_folds, test_folds = get_immucan_folds(data_root / cfg["folds_json"])
        fold_range = [args.fold] if args.fold is not None else range(len(train_folds))
        all_metrics = []
        for fi in fold_range:
            t0 = time.time()
            train_adata = build_immucan_anndata(train_folds[fi], cfg, paths, label_map,
                                                 categories, cat_to_idx, excluded_labels,
                                                 desc=f"F{fi} train")
            test_adata  = build_immucan_anndata(test_folds[fi], cfg, paths, label_map,
                                                 categories, cat_to_idx, excluded_labels,
                                                 desc=f"F{fi} test")
            m = run_fold(fi, train_adata, test_adata, categories, args, output_dir)
            m["total_time_s"] = time.time() - t0
            all_metrics.append(m)

    else:  # chl
        paths = {
            "quant_csv":          data_root / cfg["quant_csv"],
            "folds_json":         data_root / cfg["folds_json"],
            "image_dir":          data_root / cfg["image_dir"],
            "segmentation_dir":   data_root / cfg["segmentation_dir"],
            "single_channel_dir": data_root / cfg["single_channel_dir"],
            "markers_txt":        data_root / cfg["markers_txt"],
        }
        meta_df, folds, img_paths = get_chl_folds_and_meta(cfg, paths)
        _, keep_idx = get_chl_channel_info(cfg, paths)

        valid_types = sorted(ct for ct in meta_df["cell_type"].unique()
                             if ct not in excluded_labels)
        categories = np.array(valid_types)
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        print(f"[cHL] Classes ({len(categories)}): {categories.tolist()}")

        fold_range  = [args.fold] if args.fold is not None else range(len(folds))
        all_metrics = []
        for fi in fold_range:
            t0        = time.time()
            train_idx = np.array(folds[fi]["train"])
            test_idx  = np.array(folds[fi]["test"])
            train_df  = meta_df.iloc[train_idx].copy()
            test_df   = meta_df.iloc[test_idx].copy()
            print(f"\n[cHL] Fold {fi}: {len(train_df):,} train / {len(test_df):,} test cells")
            train_adata = build_chl_anndata(train_df, img_paths, keep_idx, categories,
                                            cat_to_idx, excluded_labels, paths, desc=f"F{fi} train")
            test_adata  = build_chl_anndata(test_df, img_paths, keep_idx, categories,
                                            cat_to_idx, excluded_labels, paths, desc=f"F{fi} test")
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
    print(f"\nSummary -> {output_dir / 'fold_metrics_summary.csv'}")


if __name__ == "__main__":
    main()