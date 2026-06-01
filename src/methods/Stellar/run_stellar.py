"""
run_stellar.py
==============
Author: Julia Oesterle
Date:   June 2026

Unified Stellar (supervised GNN) pipeline for IMMUcan and cHL_2_MIBI.
Technically identical to run_stellar_immucan.py — same model, same graph
construction, same training loop, same inference. Only the data loading
and fold handling forks per dataset.

Architecture:
  - Per-image spatial neighbour graphs (distance threshold on cell centroids)
  - StellarModel: Linear(in→hid) + ReLU + SAGEConv(hid→hid) + Linear(hid→classes)
  - Purely supervised cross-entropy with inverse-frequency class weights
  - NeighborLoader mini-batch training on merged graph
  - Per-graph inference

Env:    nix develop → micromamba activate stellar
Output: /home/juliaoesterle/results/stellar/{immucan,chl}/dt{threshold}/

Usage:
    # IMMUcan — all 5 image-level folds:
    python run_stellar.py --dataset immucan

    # cHL — all 5 cell-level folds:
    python run_stellar.py --dataset chl

    # Single fold, quick smoke test:
    python run_stellar.py --dataset immucan --fold 0 --epochs 2 --device cpu
    python run_stellar.py --dataset chl     --fold 0 --epochs 2 --device cpu

    # Full run on GPU:
    python run_stellar.py --dataset immucan --device cuda:0
    python run_stellar.py --dataset chl     --device cuda:0 \
        --output-dir /scratch/juliaoesterle/results/stellar/chl
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
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

import anndata


BASE = Path("/home/juliaoesterle/data/phenotyping_benchmark")

# IMMUcan 
IMMUCAN_DIR        = BASE / "IMMUcan"
IMMUCAN_IMAGE_DIR  = IMMUCAN_DIR / "CellTypes" / "data" / "images"
IMMUCAN_LABEL_DIR  = IMMUCAN_DIR / "CellTypes" / "cells2labels"
IMMUCAN_SEG_DIR    = IMMUCAN_DIR / "segmentation"
IMMUCAN_FOLDS_JSON = IMMUCAN_DIR / "CellTypes" / "folds.json"
IMMUCAN_OUTPUT     = Path("/home/juliaoesterle/results/stellar/immucan")

IMMUCAN_EXCLUDED_CHANNELS = [0, 1, 2]   # DNA1, DNA2, HistoneH3
IMMUCAN_KEEP_CHANNELS     = [i for i in range(40) if i not in IMMUCAN_EXCLUDED_CHANNELS]
IMMUCAN_N_CHANNELS        = len(IMMUCAN_KEEP_CHANNELS)  # 37

IMMUCAN_LABEL_MAP = {
    0:  "Cancer",
    1:  "Stroma",
    2:  "CD8+_T_cell",
    3:  "M2_Macrophage",
    4:  "CD4+_T_cell",
    5:  "undefined",
    6:  "Plasma_cell",
    7:  "Neutrophil",
    8:  "BnT",
    9:  "Treg",
    10: "Dendritic_cell",
    11: "B_cell",
    12: "NK_cell",
    13: "Plasmacytoid_dendritic_cell",
}
IMMUCAN_CATEGORIES = np.array(sorted(
    v for v in IMMUCAN_LABEL_MAP.values() if v != "undefined"
))
IMMUCAN_CAT_TO_IDX = {c: i for i, c in enumerate(IMMUCAN_CATEGORIES)}

# cHL_2_MIBI 
CHL_DIR        = BASE / "cHL_2_MIBI"
CHL_QUANT_CSV  = CHL_DIR / "quantification" / "processed" / "cHL_2_MIBI_quantification.csv"
CHL_FOLD_JSON  = CHL_DIR / "quantification" / "processed" / "kfolds_StratifiedGroupKFold_level3" / "fold_indices.json"
CHL_IMG_DIR    = CHL_DIR / "raw_images" / "multistack_tiffs"
CHL_SEG_BASE   = CHL_DIR / "segmentation"
CHL_OUTPUT     = Path("/home/juliaoesterle/results/stellar/chl")

CHL_N_CHANNELS_RAW  = 46
# Exclude non-protein/nuclear markers (by name, resolved to indices at runtime)
CHL_EXCLUDE_MARKERS = {"dsDNA", "Histone H3", "anti-H2AX", "pSLP-76", "SLP-76"}
CHL_EXCLUDE_LABELS  = {"undefined", "unedfined"}   # note dataset typo



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

#Graph Construction

def get_edges(pos: np.ndarray, distance_threshold: float) -> np.ndarray:
    """Distance-threshold spatial graph. Returns edge_index (2, E)."""
    if len(pos) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    diff  = pos[:, None, :] - pos[None, :, :]    # (N, N, 2)
    dists = np.linalg.norm(diff, axis=-1)         # (N, N)
    adj   = dists <= distance_threshold
    np.fill_diagonal(adj, False)
    return np.array(np.where(adj), dtype=np.int64)  # (2, E)


def make_graph_list(adata: anndata.AnnData, distance_threshold: float) -> List[Data]:
    """One graph per image — identical to run_stellar_immucan.py."""
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
    """Per-graph inference — identical to run_stellar_immucan.py."""
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
    """Merge list of per-image graphs into one big graph for NeighborLoader."""
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
    expressions: np.ndarray,     # (N, C) already filtered channels
    pos_x: np.ndarray,           # (N,) centroid X
    pos_y: np.ndarray,           # (N,) centroid Y
    sample_ids: List[str],       # (N,) image name per cell
    cell_ids: List,              # (N,) mask label IDs
    cell_types: List[str],       # (N,) string cell type
    categories: np.ndarray,
    cat_to_idx: dict,
) -> anndata.AnnData:
    """
    Shared AnnData builder. Standardises features, builds obs DataFrame.
    Identical layout to run_stellar_immucan.py build_anndata().
    """
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
    print(f"  → {adata.n_obs:,} cells | {obs['sample_id'].nunique()} images")
    return adata


# IMMUcan DATA LOADING

def load_immucan_image_data(img_name: str):
    """Vectorised expression extraction — identical to run_stellar_immucan.py."""
    img = np.load(IMMUCAN_IMAGE_DIR / f"{img_name}.npz")["data"].astype(np.float32)
    img /= 65535.0

    mask       = tifffile.imread(str(IMMUCAN_SEG_DIR / f"{img_name}.tiff")).astype(np.int32)
    raw_labels = np.loadtxt(str(IMMUCAN_LABEL_DIR / f"{img_name}.txt"), dtype=np.int32)

    props = regionprops(mask)
    if not props:
        return None, None, None, None

    cell_ids  = np.array([p.label for p in props], dtype=np.int32)
    centroids = np.array([p.centroid for p in props], dtype=np.float32)

    n_cells          = len(cell_ids)
    expressions_all  = np.zeros((n_cells, 40), dtype=np.float32)
    H, W             = mask.shape
    flat_mask        = mask.ravel()
    sort_idx         = np.argsort(flat_mask, kind="stable")
    sorted_labels    = flat_mask[sort_idx]
    boundaries       = np.searchsorted(sorted_labels, np.arange(1, cell_ids.max() + 2))
    img_flat         = img.reshape(40, -1)

    for i, cid in enumerate(cell_ids):
        start   = boundaries[cid - 1]
        end     = boundaries[cid]
        pix_idx = sort_idx[start:end]
        expressions_all[i] = img_flat[:, pix_idx].mean(axis=1)

    expressions = expressions_all[:, IMMUCAN_KEEP_CHANNELS]

    cell_types = np.array(
        [raw_labels[cid - 1] if (cid - 1) < len(raw_labels) else -1
         for cid in cell_ids],
        dtype=np.int32,
    )
    return expressions, centroids, cell_ids, cell_types


def build_immucan_anndata(img_names: List[str], desc: str = "Loading") -> anndata.AnnData:
    all_exprs, all_sample, all_cell_id = [], [], []
    all_pos_x, all_pos_y, all_labels   = [], [], []

    for img_name in tqdm(img_names, desc=desc):
        exprs, centroids, cell_ids, cell_types = load_immucan_image_data(img_name)
        if exprs is None:
            print(f"  Warning: skipping {img_name} (empty mask)")
            continue

        valid = cell_types != -1
        if valid.sum() == 0:
            continue

        exprs      = exprs[valid]
        centroids  = centroids[valid]
        cell_ids   = cell_ids[valid]
        cell_types = cell_types[valid]

        # Filter to known categories (exclude "undefined")
        known = np.array([IMMUCAN_LABEL_MAP[ct] != "undefined" for ct in cell_types])
        if known.sum() == 0:
            continue
        exprs      = exprs[known]
        centroids  = centroids[known]
        cell_ids   = cell_ids[known]
        cell_types = cell_types[known]

        n = exprs.shape[0]
        all_exprs.append(exprs)
        all_sample.extend([img_name] * n)
        all_cell_id.extend(cell_ids.tolist())
        all_pos_x.extend(centroids[:, 1].tolist())   # col = X
        all_pos_y.extend(centroids[:, 0].tolist())   # row = Y
        all_labels.extend([IMMUCAN_LABEL_MAP[ct] for ct in cell_types])

    return build_adata_from_arrays(
        expressions = np.concatenate(all_exprs, axis=0),
        pos_x       = np.array(all_pos_x),
        pos_y       = np.array(all_pos_y),
        sample_ids  = all_sample,
        cell_ids    = all_cell_id,
        cell_types  = all_labels,
        categories  = IMMUCAN_CATEGORIES,
        cat_to_idx  = IMMUCAN_CAT_TO_IDX,
    )


def get_immucan_folds():
    with open(IMMUCAN_FOLDS_JSON) as f:
        raw = json.load(f)
    n = sum(1 for k in raw if k.endswith("_train_set"))
    return (
        [raw[f"fold_{i}_train_set"] for i in range(n)],
        [raw[f"fold_{i}_test_set"]  for i in range(n)],
    )


# CHL Data Loading

def get_chl_channel_info() -> Tuple[List[str], List[int]]:
    """Channel order = alphabetical from single_channel_tiffs/1/."""
    sc_dir = CHL_DIR / "raw_images" / "single_channel_tiffs" / "1"
    if sc_dir.exists():
        all_names = sorted(
            f.stem for f in sc_dir.rglob("*.tiff") if not f.name.startswith(".")
        )
    else:
        with open(CHL_DIR / "markers.txt") as f:
            all_names = [l.strip() for l in f if l.strip()]
    keep_idx = [i for i, n in enumerate(all_names) if n not in CHL_EXCLUDE_MARKERS]
    print(f"  [cHL] Channels: {len(keep_idx)}/{len(all_names)} kept "
          f"(excluded: {CHL_EXCLUDE_MARKERS})")
    return all_names, keep_idx


def load_chl_seg(img_id: str) -> np.ndarray:
    seg_dir   = CHL_SEG_BASE / str(img_id)
    seg_files = [f for f in seg_dir.rglob("segmentationMap.tif")
                 if not f.name.startswith(".")]
    if not seg_files:
        raise FileNotFoundError(f"No segmentationMap.tif under {seg_dir}")
    return tifffile.imread(str(seg_files[0])).astype(np.int32)


def extract_chl_expressions(img_path: Path, mask: np.ndarray,
                              keep_idx: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised expression extraction for cHL OME-TIFF.
    Same approach as IMMUcan (argsort flat-index lookup).
    Returns: cell_ids, expressions (N, C_kept), centroids (N, 2)
    """
    img      = tifffile.imread(str(img_path)).astype(np.float32)  # (46, H, W)
    img_keep = img[keep_idx]                                        # (C_kept, H, W)
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


def build_chl_anndata(
    subset_df: pd.DataFrame,   # rows from quant CSV belonging to this split
    img_paths: dict,
    keep_idx:  List[int],
    categories: np.ndarray,
    cat_to_idx: dict,
    desc: str = "Loading",
) -> anndata.AnnData:
    """
    Build AnnData for cHL cells.
    For each image, extract expressions from OME-TIFF using segmentation mask,
    then join to subset_df on cell_id + sample_id.
    """
    all_exprs, all_sample, all_cell_id = [], [], []
    all_pos_x, all_pos_y, all_labels   = [], [], []

    for img_id in tqdm(sorted(subset_df["sample_id"].unique()), desc=desc):
        if img_id not in img_paths:
            print(f"  [WARN] No image for {img_id}, skipping")
            continue
        try:
            mask = load_chl_seg(img_id)
        except FileNotFoundError as e:
            print(f"  [WARN] {e}, skipping")
            continue

        cell_ids, exprs, centroids = extract_chl_expressions(
            img_paths[img_id], mask, keep_idx
        )
        if len(cell_ids) == 0:
            continue

        # Join to subset_df
        img_df   = subset_df[subset_df["sample_id"] == img_id].copy()
        img_df["cell_id"] = img_df["cell_id"].astype(int)
        cid_set  = set(img_df["cell_id"].values)

        # Build lookup: mask_label → (expression_row, centroid)
        cid_to_row = {int(cid): i for i, cid in enumerate(cell_ids)}

        for _, row in img_df.iterrows():
            cid  = int(row["cell_id"])
            ct   = str(row["cell_type"])
            if ct in CHL_EXCLUDE_LABELS:
                continue
            if ct not in cat_to_idx:
                continue
            if cid not in cid_to_row:
                continue
            ridx = cid_to_row[cid]
            all_exprs.append(exprs[ridx])
            all_sample.append(str(img_id))
            all_cell_id.append(cid)
            all_pos_x.append(float(centroids[ridx, 1]))   # col = X
            all_pos_y.append(float(centroids[ridx, 0]))   # row = Y
            all_labels.append(ct)

    if not all_exprs:
        raise RuntimeError("No cells loaded — check paths and cell_id matching")

    return build_adata_from_arrays(
        expressions = np.stack(all_exprs, axis=0),
        pos_x       = np.array(all_pos_x),
        pos_y       = np.array(all_pos_y),
        sample_ids  = all_sample,
        cell_ids    = all_cell_id,
        cell_types  = all_labels,
        categories  = categories,
        cat_to_idx  = cat_to_idx,
    )


def get_chl_folds_and_meta():
    meta_df = pd.read_csv(CHL_QUANT_CSV)
    meta_df["sample_id"] = meta_df["sample_id"].astype(str).str.replace(".csv", "", regex=False)
    print(f"[cHL] {len(meta_df):,} cells, {meta_df['sample_id'].nunique()} images")

    with open(CHL_FOLD_JSON) as f:
        fold_data = json.load(f)
    folds = fold_data["folds"]
    print(f"[cHL] {len(folds)} folds (cell-level)")

    img_paths = {}
    for p in sorted(CHL_IMG_DIR.glob("*_stacked.ome.tif")):
        if p.name.startswith("."): continue
        img_paths[p.name.replace("_stacked.ome.tif", "")] = p
    print(f"[cHL] {len(img_paths)} images found")

    return meta_df, folds, img_paths


# Fold loop

def run_fold(
    fold_idx:     int,
    train_adata:  anndata.AnnData,
    test_adata:   anndata.AnnData,
    categories:   np.ndarray,
    args:         argparse.Namespace,
    output_dir:   Path,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx}  |  train={train_adata.n_obs:,}  test={test_adata.n_obs:,}")
    print(f"{'='*60}")
    device = torch.device(args.device)

    # graphs
    train_graphs = make_graph_list(train_adata, args.distance_threshold)
    test_graphs  = make_graph_list(test_adata,  args.distance_threshold)

    # model
    n_features = train_adata.X.shape[1]
    model = StellarModel(
        input_dim   = n_features,
        hid_dim     = args.hid_dim,
        num_classes = len(categories),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    # merge train graphs → NeighborLoader
    merged_train = merge_graphs_for_loader(train_graphs)
    print(f"  Merged train graph: {merged_train.x.shape[0]:,} nodes, "
          f"{merged_train.edge_index.shape[1]:,} edges")
    loader = NeighborLoader(
        merged_train,
        num_neighbors     = [10],
        batch_size        = args.node_batch_size,
        shuffle           = True,
        num_workers       = 0,
    )

    # class weights
    class_weights = compute_class_weights([merged_train], len(categories), device)
    print(f"  Class weights: min={class_weights.min():.2f} "
          f"max={class_weights.max():.2f} mean={class_weights.mean():.2f}")

    # training loop with periodic test evaluation
    learning_curve = []
    t_train = time.time()
    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(model, loader, optimizer, device, class_weights)

        if (epoch + 1) % args.eval_every == 0 or epoch == 0 or epoch == args.epochs - 1:
            ep_codes, _ = run_inference(model, test_graphs, device)
            ep_preds    = categories[ep_codes]
            ep_true     = np.concatenate([
                test_adata.obs.loc[g.cell_ids, "cell_labels"].values
                for g in test_graphs
            ])
            ep_acc  = accuracy_score(ep_true, ep_preds)
            ep_f1   = f1_score(ep_true, ep_preds, average="macro",
                               labels=categories, zero_division=0)
            ep_wf1  = f1_score(ep_true, ep_preds, average="weighted",
                               labels=categories, zero_division=0)
            learning_curve.append({
                "epoch": epoch + 1, "loss": loss, "train_acc": acc,
                "accuracy": ep_acc, "macro_f1": ep_f1, "wf1": ep_wf1,
            })
            print(f"  Epoch {epoch+1:3d}/{args.epochs}  loss={loss:.4f}  "
                  f"train_acc={acc:.4f}  val_F1={ep_f1:.4f}  val_Acc={ep_acc:.4f}")
        else:
            learning_curve.append({
                "epoch": epoch + 1, "loss": loss, "train_acc": acc,
                "accuracy": None, "macro_f1": None, "wf1": None,
            })

    train_time = time.time() - t_train
    print(f"  Training: {train_time:.1f}s")

    lc_dir = output_dir / "learning_curves"
    lc_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(learning_curve).to_csv(lc_dir / f"fold_{fold_idx}.csv", index=False)

    # final evaluation
    pred_codes, probs = run_inference(model, test_graphs, device)
    pred_labels = categories[pred_codes]
    true_labels = np.concatenate([
        test_adata.obs.loc[g.cell_ids, "cell_labels"].values
        for g in test_graphs
    ])

    acc      = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro",
                        labels=categories, zero_division=0)
    wf1      = f1_score(true_labels, pred_labels, average="weighted",
                        labels=categories, zero_division=0)
    print(f"\n  Fold {fold_idx}: Acc={acc:.4f}  MacroF1={macro_f1:.4f}  WF1={wf1:.4f}")
    print(classification_report(true_labels, pred_labels,
                                labels=categories, zero_division=0))

    # save predictions
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
    print(f"  Saved → {pred_dir}/predictions_{fold_idx}.csv")

    # save model
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / f"model_fold{fold_idx}.pth")

    return {"fold": fold_idx, "accuracy": acc, "macro_f1": macro_f1,
            "wf1": wf1, "train_time_s": train_time}


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Supervised Stellar GNN — IMMUcan or cHL_2_MIBI"
    )
    parser.add_argument("--dataset", choices=["immucan", "chl"], required=True)
    parser.add_argument("--fold", type=int, default=None,
                        help="Single fold (0-based). Default: all 5.")
    parser.add_argument("--hid-dim",         type=int,   default=160)
    parser.add_argument("--distance-threshold", type=float, default=14.28)
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--lr",              type=float, default=0.001)
    parser.add_argument("--weight-decay",    type=float, default=0.0001)
    parser.add_argument("--eval-every",      type=int,   default=5)
    parser.add_argument("--node-batch-size", type=int,   default=512)
    parser.add_argument("--device",          type=str,   default="cuda:0")
    parser.add_argument("--output-dir",      type=str,   default=None)
    args = parser.parse_args()

    dt_str     = f"dt{args.distance_threshold:.2f}".replace(".", "_")
    default_out = IMMUCAN_OUTPUT if args.dataset == "immucan" else CHL_OUTPUT
    output_dir  = Path(args.output_dir) / dt_str if args.output_dir else default_out / dt_str
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStellar — dataset={args.dataset}  distance_threshold={args.distance_threshold}px")
    print(f"  hid_dim={args.hid_dim}  epochs={args.epochs}  device={args.device}")
    print(f"  output_dir: {output_dir}\n")

    # IMMUcan 
    if args.dataset == "immucan":
        train_folds, test_folds = get_immucan_folds()
        fold_range = [args.fold] if args.fold is not None else range(len(train_folds))
        all_metrics = []

        for fi in fold_range:
            t0 = time.time()
            train_adata = build_immucan_anndata(train_folds[fi], desc=f"F{fi} train")
            test_adata  = build_immucan_anndata(test_folds[fi],  desc=f"F{fi} test")
            m = run_fold(fi, train_adata, test_adata, IMMUCAN_CATEGORIES, args, output_dir)
            m["total_time_s"] = time.time() - t0
            all_metrics.append(m)

    # cHL 
    elif args.dataset == "chl":
        meta_df, folds, img_paths = get_chl_folds_and_meta()
        _, keep_idx = get_chl_channel_info()

        # Build cHL categories from training labels (exclude undefined)
        valid_types = sorted(
            ct for ct in meta_df["cell_type"].unique()
            if ct not in CHL_EXCLUDE_LABELS
        )
        chl_categories = np.array(valid_types)
        chl_cat_to_idx = {c: i for i, c in enumerate(chl_categories)}
        print(f"[cHL] Classes ({len(chl_categories)}): {chl_categories.tolist()}")

        fold_range  = [args.fold] if args.fold is not None else range(len(folds))
        all_metrics = []

        for fi in fold_range:
            t0        = time.time()
            train_idx = np.array(folds[fi]["train"])
            test_idx  = np.array(folds[fi]["test"])
            train_df  = meta_df.iloc[train_idx].copy()
            test_df   = meta_df.iloc[test_idx].copy()
            print(f"\n[cHL] Fold {fi}: {len(train_df):,} train / {len(test_df):,} test cells")

            train_adata = build_chl_anndata(
                train_df, img_paths, keep_idx, chl_categories, chl_cat_to_idx,
                desc=f"F{fi} train"
            )
            test_adata = build_chl_anndata(
                test_df, img_paths, keep_idx, chl_categories, chl_cat_to_idx,
                desc=f"F{fi} test"
            )
            m = run_fold(fi, train_adata, test_adata, chl_categories, args, output_dir)
            m["total_time_s"] = time.time() - t0
            all_metrics.append(m)

    # ── summary ──
    print(f"\n{'='*60}\n  SUMMARY — {args.dataset}\n{'='*60}")
    df = pd.DataFrame(all_metrics)
    print(df[["fold", "accuracy", "macro_f1", "wf1"]].to_string(index=False))
    if len(df) > 1:
        print(f"\n  Mean Acc     : {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
        print(f"  Mean MacroF1 : {df['macro_f1'].mean():.4f} ± {df['macro_f1'].std():.4f}")
        print(f"  Mean WF1     : {df['wf1'].mean():.4f} ± {df['wf1'].std():.4f}")
    df.to_csv(output_dir / "fold_metrics_summary.csv", index=False)
    print(f"\nSummary → {output_dir / 'fold_metrics_summary.csv'}")


if __name__ == "__main__":
    main()
