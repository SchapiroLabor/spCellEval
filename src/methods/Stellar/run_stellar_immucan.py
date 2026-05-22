"""
run_stellar_immucan.py
Stellar (supervised GNN variant) on IMMUcan dataset.
Based on the dav3794/IMC-models refactor of snap-stanford/stellar.

Architecture:
  - Per-image spatial neighbor graphs (distance threshold on cell centroids)
  - GNN encoder (SAGEConv) + NormedLinear classification head
  - Purely supervised cross-entropy training (no semi-supervised STELLAR, like in dav3794/IMC-models)
  - Uses predefined image-level 5-fold CV splits from folds.json

Env:    nix develop → micromamba activate stellar
Output: /scratch/juliaoesterle/results/stellar/

Usage:
    # Default run (SAGEConv, distance_threshold=14.28, all 5 folds):
    python run_stellar_immucan.py

    # Single fold for quick testing:
    python run_stellar_immucan.py --fold 0 --epochs 2

    # Hyperparameter sweep over distance threshold:
    python run_stellar_immucan.py --distance-threshold 20.0

    # Explicit GPU:
    python run_stellar_immucan.py --device cuda:0
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
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

import anndata

# Paths
IMMUCAN_DIR = Path("/home/juliaoesterle/data/phenotyping_benchmark/IMMUcan")
IMAGE_DIR   = IMMUCAN_DIR / "CellTypes" / "data" / "images"
LABEL_DIR   = IMMUCAN_DIR / "CellTypes" / "cells2labels"
SEG_DIR     = IMMUCAN_DIR / "segmentation"
FOLDS_JSON  = IMMUCAN_DIR / "CellTypes" / "folds.json"
OUTPUT_BASE = Path("/home/juliaoesterle/results/stellar")

# IMMUcan-specific 
# NOTE: verify channel indices against your panel if results look off
EXCLUDED_CHANNELS = [0, 1, 2]
KEEP_CHANNELS     = [i for i in range(40) if i not in EXCLUDED_CHANNELS]
N_CHANNELS_USED   = len(KEEP_CHANNELS)  # 37

# Integer -->  string cell type mapping for IMMUcan
# -1 = Unknown, excluded from training/evaluation
LABEL_MAP = {
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
# Sorted category list — consistent label encoding across all folds
CATEGORIES = np.array(sorted(LABEL_MAP.values()))
# Map string label 
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}


# data loading 

def load_image_data(
    img_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one IMMUcan image and return per-cell features.

    Expression extraction: vectorised via flat index lookup into image array,
    much faster than looping regionprops per channel.

    Returns:
        expressions : (N, 37) float32 — mean marker intensity per cell
        centroids   : (N, 2)  float32 — (row=Y, col=X) centroids
        cell_ids    : (N,)    int32   — mask label IDs (1-indexed)
        cell_types  : (N,)    int32   — cell type ints per LABEL_MAP (-1=unknown)
    """
    #  image: (40, H, W) uint16 
    img = np.load(IMAGE_DIR / f"{img_name}.npz")["data"].astype(np.float32)
    img /= 65535.0  # normalise to [0, 1]

    #  segmentation mask: (H, W) uint16 → int32 
    mask = tifffile.imread(str(SEG_DIR / f"{img_name}.tiff")).astype(np.int32)

    # cell type labels: one int per line, order = mask label 1, 2, 3,...
    raw_labels = np.loadtxt(str(LABEL_DIR / f"{img_name}.txt"), dtype=np.int32)

    # get cell properties from mask 
    props = regionprops(mask)
    if not props:
        return None, None, None, None

    cell_ids  = np.array([p.label for p in props], dtype=np.int32)
    centroids = np.array([p.centroid for p in props], dtype=np.float32)

    # vectorised expression extraction 
    # For each cell, collect all pixel indices, then mean over channels at once.
    # This avoids 40 separate regionprops calls.
    n_cells = len(cell_ids)
    expressions_all = np.zeros((n_cells, 40), dtype=np.float32)
    H, W = mask.shape
    flat_mask = mask.ravel() 

    # Build per-cell pixel index arrays using argsort
    sort_idx  = np.argsort(flat_mask, kind="stable")
    sorted_labels = flat_mask[sort_idx]
    # Find boundaries between label regions (label 0 = background)
    boundaries = np.searchsorted(sorted_labels, np.arange(1, cell_ids.max() + 2))

    img_flat = img.reshape(40, -1)  

    for i, cid in enumerate(cell_ids):
        start = boundaries[cid - 1]
        end   = boundaries[cid]
        pix_idx = sort_idx[start:end]
        expressions_all[i] = img_flat[:, pix_idx].mean(axis=1)

    # Keep only non-DNA channels
    expressions = expressions_all[:, KEEP_CHANNELS]  

    # cell type labels 
    # raw_labels[i] = cell type for mask label (i+1)
    cell_types = np.array(
        [raw_labels[cid - 1] if (cid - 1) < len(raw_labels) else -1
         for cid in cell_ids],
        dtype=np.int32,
    )

    return expressions, centroids, cell_ids, cell_types


def build_anndata(img_names: List[str], desc: str = "Loading") -> anndata.AnnData:
    """
    Build AnnData from a list of IMMUcan image names.
    Cells with label -1 (Unknown) are excluded.

    Layout:
        .X / .layers["exprs"]  (N, 37) float32
        .obs["sample_id"]      Categorical image name
        .obs["cell_id"]        int — mask label ID
        .obs["Pos_X"]          float — centroid col
        .obs["Pos_Y"]          float — centroid row
        .obs["cell_labels"]    Categorical string cell type
        .obs["cell_label_idx"] int — index into CATEGORIES array
    """
    all_exprs, all_sample, all_cell_id = [], [], []
    all_pos_x, all_pos_y, all_labels   = [], [], []

    for img_name in tqdm(img_names, desc=desc):
        exprs, centroids, cell_ids, cell_types = load_image_data(img_name)
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

        n = exprs.shape[0]
        all_exprs.append(exprs)
        all_sample.extend([img_name] * n)
        all_cell_id.extend(cell_ids.tolist())
        all_pos_x.extend(centroids[:, 1].tolist())  # col = X
        all_pos_y.extend(centroids[:, 0].tolist())  # row = Y
        all_labels.extend([LABEL_MAP[ct] for ct in cell_types])

    X = np.concatenate(all_exprs, axis=0)

    # Standardise features to zero mean, unit variance per marker, else input_linear gradients vanish (~1e-5) and training fails.
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True)
    X    = (X - mean) / (std + 1e-8)

    obs = pd.DataFrame({
        "sample_id":      pd.Categorical(all_sample),
        "cell_id":        all_cell_id,
        "Pos_X":          np.array(all_pos_x, dtype=np.float32),
        "Pos_Y":          np.array(all_pos_y, dtype=np.float32),
        "cell_labels":    pd.Categorical(all_labels, categories=CATEGORIES),
        "cell_label_idx": [CAT_TO_IDX[l] for l in all_labels],
    })
    obs.index = [f"{s}_{c}" for s, c in zip(all_sample, all_cell_id)]

    adata = anndata.AnnData(X=X, obs=obs)
    adata.layers["exprs"] = X.copy()
    print(f"  → {adata.n_obs:,} cells | {len(obs['sample_id'].cat.categories)} images")
    return adata


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
    """One graph per image"""
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


# training & evaluation 

def compute_class_weights(graphs: List[Data], num_classes: int, device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a list of graphs.
    Balances the loss against the strong class imbalance in IMMUcan
    (Cancer ~210k vs NK_cell ~3k).
    """
    counts = torch.zeros(num_classes)
    for g in graphs:
        for c in range(num_classes):
            counts[c] += (g.y == c).sum()
    # Inverse frequency, normalised to mean=1
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
        # Only compute loss on seed nodes (first N in batch) — the rest are neighbours included by NeighborLoader for message passing, but not part of the supervised loss.
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
    """Returns (pred_codes, probs) for all cells across graphs."""
    model.eval()
    # Run per-graph 
    all_logits = []
    for g in graphs:
        out, _ = model(g.to(device))
        all_logits.append(out.cpu())
    logits = torch.cat(all_logits, dim=0)
    probs  = F.softmax(logits, dim=1).numpy()
    codes  = logits.argmax(1).numpy()
    return codes, probs


# fold runner 

def run_fold(fold_idx, train_imgs, test_imgs, args, output_dir) -> dict:
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx}  |  train={len(train_imgs)}  test={len(test_imgs)}")
    print(f"{'='*60}")
    device = torch.device(args.device)

    # data 
    t0 = time.time()
    train_adata = build_anndata(train_imgs, desc=f"F{fold_idx} train")
    test_adata  = build_anndata(test_imgs,  desc=f"F{fold_idx} test")
    train_graphs = make_graph_list(train_adata, args.distance_threshold)
    test_graphs  = make_graph_list(test_adata,  args.distance_threshold)
    print(f"  Data ready in {time.time()-t0:.1f}s")

    # model 
    model = StellarModel(
        input_dim   = N_CHANNELS_USED,
        hid_dim     = args.hid_dim,
        num_classes = len(CATEGORIES),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    # Merge all train graphs into a single Data object (offsetting edge indices),  so NeighborLoader treats individual cells as nodes, not graphs; Batch.from_data_list does NOT work here — PyG sees it as graph-level batching. TBD 
    from torch_geometric.loader import NeighborLoader
    all_x, all_y, all_ei = [], [], []
    offset = 0
    for g in train_graphs:
        all_x.append(g.x)
        all_y.append(g.y)
        all_ei.append(g.edge_index + offset)
        offset += g.x.shape[0]
    merged_train = Data(
        x          = torch.cat(all_x, dim=0),
        y          = torch.cat(all_y, dim=0),
        edge_index = torch.cat(all_ei, dim=1),
    )
    print(f"  Merged graph: {merged_train.x.shape[0]:,} nodes, "
          f"{merged_train.edge_index.shape[1]:,} edges")
    loader = NeighborLoader(
        merged_train,
        num_neighbors = [10],
        batch_size    = args.node_batch_size,
        shuffle       = True,
        num_workers   = 0,
    )

    # Class weights
    class_weights = compute_class_weights([merged_train], len(CATEGORIES), device)
    print(f"  Class weights (min={class_weights.min():.2f} "
          f"max={class_weights.max():.2f} mean={class_weights.mean():.2f})")

    # training 
    # Track per-epoch metrics on test set for learning curves
    learning_curve = []
    t_train = time.time()
    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(model, loader, optimizer, device, class_weights)

        # Evaluate on test set every eval_every epochs
        if (epoch + 1) % args.eval_every == 0 or epoch == 0 or epoch == args.epochs - 1:
            ep_codes, _ = run_inference(model, test_graphs, device)
            ep_preds    = CATEGORIES[ep_codes]
            ep_true     = np.concatenate([
                test_adata.obs.loc[g.cell_ids, "cell_labels"].values
                for g in test_graphs
            ])
            ep_acc  = accuracy_score(ep_true, ep_preds)
            ep_f1   = f1_score(ep_true, ep_preds, average="macro",
                               labels=CATEGORIES, zero_division=0)
            ep_wf1  = f1_score(ep_true, ep_preds, average="weighted",
                               labels=CATEGORIES, zero_division=0)
            learning_curve.append({
                "epoch":    epoch + 1,
                "loss":     loss,
                "train_acc": acc,
                "accuracy": ep_acc,
                "macro_f1": ep_f1,
                "wf1":      ep_wf1,
            })
            print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
                  f"loss={loss:.4f}  train_acc={acc:.4f}  "
                  f"val_F1={ep_f1:.4f}  val_Acc={ep_acc:.4f}")
        else:
            learning_curve.append({
                "epoch":    epoch + 1,
                "loss":     loss,
                "train_acc": acc,
                "accuracy": None,
                "macro_f1": None,
                "wf1":      None,
            })

    train_time = time.time() - t_train
    print(f"  Training: {train_time:.1f}s")

    # Save learning curve
    lc_dir = output_dir / "learning_curves"
    lc_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(learning_curve).to_csv(
        lc_dir / f"fold_{fold_idx}.csv", index=False
    )

    # evaluation 
    pred_codes, probs = run_inference(model, test_graphs, device)
    pred_labels = CATEGORIES[pred_codes]

    # Ground truth
    true_labels = np.concatenate([
        test_adata.obs.loc[g.cell_ids, "cell_labels"].values
        for g in test_graphs
    ])

    acc      = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro",
                        labels=CATEGORIES, zero_division=0)
    wf1      = f1_score(true_labels, pred_labels, average="weighted",
                        labels=CATEGORIES, zero_division=0)

    print(f"\n  Fold {fold_idx}: Acc={acc:.4f}  MacroF1={macro_f1:.4f}  WF1={wf1:.4f}")
    print(classification_report(true_labels, pred_labels,
                                 labels=CATEGORIES, zero_division=0))

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


# main 

def main():
    parser = argparse.ArgumentParser(
        description="Supervised Stellar GNN on IMMUcan"
    )
    parser.add_argument("--fold", type=int, default=None,
                        help="Single fold to run (0-4). Default: all 5.")
    parser.add_argument("--hid-dim", type=int, default=160)
    parser.add_argument("--distance-threshold", type=float, default=14.28,
                        help="Spatial neighbour threshold in pixels (default: 14.28)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Image graphs per batch (default: 1)")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Evaluate on test set every N epochs (default: 5)")
    parser.add_argument("--node-batch-size", type=int, default=512,
                        help="Seed nodes per NeighborLoader mini-batch (default: 512)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir. Default: OUTPUT_BASE/dt{threshold}/")
    args = parser.parse_args()

    # ensure easy parameter sweep
    if args.output_dir is None:
        dt_str = f"dt{args.distance_threshold:.2f}".replace(".", "_")
        output_dir = OUTPUT_BASE / dt_str
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStellar IMMUcan — supervised GNN")
    print(f"  distance_threshold : {args.distance_threshold} px")
    print(f"  hid_dim            : {args.hid_dim}")
    print(f"  epochs             : {args.epochs}")
    print(f"  device             : {args.device}")
    print(f"  output_dir         : {output_dir}")
    print(f"  categories ({len(CATEGORIES)}): {list(CATEGORIES)}\n")

    with open(FOLDS_JSON) as f:
        folds = json.load(f)

    fold_indices = [args.fold] if args.fold is not None else list(range(5))
    all_metrics, fold_times = [], []

    for fold_idx in fold_indices:
        train_imgs = folds[f"fold_{fold_idx}_train_set"]
        test_imgs  = folds[f"fold_{fold_idx}_test_set"]
        t0 = time.time()
        metrics = run_fold(fold_idx, train_imgs, test_imgs, args, output_dir)
        fold_times.append(f"Fold {fold_idx}: {time.time()-t0:.1f}s")
        all_metrics.append(metrics)

    # summary 
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    df = pd.DataFrame(all_metrics)
    print(df[["fold", "accuracy", "macro_f1", "wf1"]].to_string(index=False))
    if len(df) > 1:
        print(f"\n  Mean Acc     : {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
        print(f"  Mean MacroF1 : {df['macro_f1'].mean():.4f} ± {df['macro_f1'].std():.4f}")
        print(f"  Mean WF1     : {df['wf1'].mean():.4f} ± {df['wf1'].std():.4f}")

    df.to_csv(output_dir / "fold_metrics_summary.csv", index=False)

    times_path = output_dir / "stellar_supervised" / "level3" / "fold_times.txt"
    times_path.parent.mkdir(parents=True, exist_ok=True)
    times_path.write_text("\n".join(fold_times))
    print(f"\nSummary → {output_dir / 'fold_metrics_summary.csv'}")


if __name__ == "__main__":
    main()