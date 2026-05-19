#!/usr/bin/env python3
"""
run_kronos_original.py — KRONOS Cell Phenotyping: Original Tutorial Approach
=============================================================================
Author: Julia Oesterle
Date:   May 2026

Replicates the KRONOS authors' original pipeline from tutorials/2-Cell-phenotyping.ipynb
as closely as possible, adapted for the IMMUcan dataset and our predefined folds.json splits.

Key differences from run_kronos.py (our custom approach):
  - Patch extraction saves .h5 files per cell (authors' approach)
  - Feature extraction loads from .h5 files via CellPhenotypingDataset
  - Classifier: Logistic Regression + Optuna hyperparameter tuning (not RF)
  - Normalisation: marker_max_values=65535.0 (uint16 dtype) then per-marker mean/std
  - Note: IMMUcan images are uint16 but max value ~1225 (not full range)
    so 65535.0 divisor produces very small values — kept as-is for exact replication

Purpose: Validate that our custom pipeline (run_kronos.py) gives comparable
results to the authors' exact approach. Any large discrepancy would indicate
a bug in our implementation.

Usage
-----
# Extract patches + features + run LogReg classifier
python3 run_kronos_original.py all \\
    --data-dir        /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan/ \\
    --kronos-dir      /home/juliaoesterle/KRONOS_mine \\
    --marker-meta     /home/juliaoesterle/KRONOS_mine/model_assets/marker_metadata.csv \\
    --output-dir      /home/juliaoesterle/results/kronos_original \\
    --device          cuda:0

# Extract patches only
python3 run_kronos_original.py extract_patches \\
    --data-dir    /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan/ \\
    --kronos-dir  /home/juliaoesterle/KRONOS_mine \\
    --marker-meta /home/juliaoesterle/KRONOS_mine/model_assets/marker_metadata.csv \\
    --output-dir  /home/juliaoesterle/results/kronos_original \\
    --device      cuda:0

# Extract features from saved patches
python3 run_kronos_original.py extract_features \\
    --output-dir /home/juliaoesterle/results/kronos_original \\
    --device     cuda:0

# Run LogReg classifier on saved features
python3 run_kronos_original.py classify \\
    --data-dir   /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan/ \\
    --output-dir /home/juliaoesterle/results/kronos_original

Output structure
----------------
{output_dir}/
├── patches/                    ← .h5 files per cell (image_cellid.h5)
├── features/                   ← .npy per cell feature vector
├── KRONOS_original_supervised/
│   └── level3/
│       ├── predictions_0.csv ... predictions_4.csv
│       └── fold_times.txt

Notes on IMMUcan adaptation
----------------------------
- Images are .npz (not .tiff) with shape (40, 600, 600) uint16
- Segmentation masks are .tiff (uint16 cell IDs)
- Ground truth: cell_labels[cell_id-1], -1 = Unknown
- Folds from folds.json (image-level, predefined) — NOT generated from data
- marker_max_values=65535.0 to match authors (images are uint16 dtype)
"""

#  Standard library 
import os
import sys
import time
import h5py
import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

#  CUDA env vars — must be set before torch import 
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.3')
os.environ.setdefault('LD_LIBRARY_PATH',
                      '/usr/local/cuda-12.3/lib64:' +
                      os.environ.get('LD_LIBRARY_PATH', ''))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

#  Third-party 
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

#  Shared benchmark utilities
from utils.utils_foundational_models import (
    load_label_map,
    load_folds,
    get_label,
    add_shared_args,
)

warnings.filterwarnings('ignore')


#  Marker definitions (same as run_kronos.py) 

ALL_BIOMARKERS = [
    "MPO", "HistoneH3", "SMA", "CD16", "CD38", "HLADR", "CD27", "CD15",
    "CD45RA", "CD163", "B2M", "CD20", "CD68", "Ido1", "CD3", "LAG3",
    "CD11c", "PD1", "PDGFRb", "CD7", "GrzB", "PDL1", "TCF7", "CD45RO",
    "FOXP3", "ICOS", "CD8a", "CarbonicAnhydrase", "CD33", "Ki67",
    "VISTA", "CD40", "CD4", "CD14", "Ecad", "CD303", "CD206",
    "cleavedPARP", "DNA1", "DNA2"
]

DEFAULT_EXCLUDE = ['DNA1', 'DNA2', 'HistoneH3']

KRONOS_NAME_MAPPINGS = {
    'GrzB':        'GZMB',
    'HLADR':       'HLA_DR',
    'CD8a':        'CD8',
    'cleavedPARP': 'CLEAVED_CASP3',
}

KRONOS_NO_EQUIVALENT = {
    'SMA', 'PDGFRb', 'TCF7', 'CarbonicAnhydrase', 'Ecad','CD33', 'CD303'
}

# uint16 max — matches authors exactly
MARKER_MAX_VALUES = 65535.0


def get_clean_markers(exclude):
    exclude_set = set(exclude)
    clean_idx   = [i for i, m in enumerate(ALL_BIOMARKERS) if m not in exclude_set]
    clean_names = [m for m in ALL_BIOMARKERS if m not in exclude_set]
    return clean_idx, clean_names


# ── Build marker_info_with_metadata.csv for IMMUcan ───────────────────────────

def build_marker_info_csv(clean_names, marker_meta_path, output_path):
    """
    Build marker_info_with_metadata.csv in the format CellPhenotypingDataset expects.
    Columns: channel_id, marker_name, marker_id, marker_mean, marker_std

    This is the IMMUcan-specific adaptation of the authors' marker metadata file.
    For markers without KRONOS equivalent: mean=0, std=1 (identity normalisation).
    """
    df_meta = pd.read_csv(marker_meta_path)
    lookup  = {
        row['marker_name'].upper(): (row['marker_id'], row['marker_mean'], row['marker_std'])
        for _, row in df_meta.iterrows()
    }

    rows = []
    for channel_id, m in enumerate(clean_names):
        kronos_name = KRONOS_NAME_MAPPINGS.get(m, m)
        key         = kronos_name.upper()
        if key in lookup:
            mid, mean, std = lookup[key]
        else:
            mid, mean, std = 0, 0.0, 1.0
        rows.append({
            'channel_id':  channel_id,
            'marker_name': m,
            'marker_id':   int(mid),
            'marker_mean': float(mean),
            'marker_std':  float(std) if std > 0 else 1.0,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"Saved marker_info_with_metadata.csv ({len(df_out)} markers) → {output_path}")
    return df_out


#  Step 1: Patch Extraction (authors' .h5 approach) 

def extract_patches(args, all_images, clean_indices, clean_names, label_map):
    """
    Extract cell-centered patches and save as .h5 files.
    Replicates CellPhenotyping.patch_extraction() from the authors' tutorial.

    Each .h5 file contains:
      - 'mask': binary cell mask (uint8, 1=cell, 0=background)
      - One dataset per marker name with raw intensity values (uint16)
    File naming: {image_name}_{cell_id:06d}.h5
    """
    data_dir  = Path(args.data_dir)
    patch_dir = Path(args.output_dir) / 'patches'
    patch_dir.mkdir(parents=True, exist_ok=True)

    NPZ_DIR    = data_dir / 'CellTypes' / 'data' / 'images'
    MASK_DIR   = data_dir / 'segmentation'
    LABELS_DIR = data_dir / 'CellTypes' / 'cells2labels'

    print(f"\nExtracting patches → {patch_dir}")
    print(f"  patch_size={args.patch_size} | markers={len(clean_names)}")
    print("=" * 60)

    n_cells_total = 0
    n_skipped     = 0

    for img_name in tqdm(all_images, desc="Patch extraction"):
        npz_file   = NPZ_DIR    / f"{img_name}.npz"
        mask_file  = MASK_DIR   / f"{img_name}.tiff"
        label_file = LABELS_DIR / f"{img_name}.txt"
        if not all(f.exists() for f in [npz_file, mask_file, label_file]):
            n_skipped += 1
            continue

        # Load raw image — keep as uint16, do NOT normalise here
        # Authors normalise in the Dataset __getitem__ using marker_max_values
        img = np.load(npz_file)['data']         # (40, H, W) uint16
        img = img[clean_indices]                 # (37, H, W) uint16
        img = img.transpose(1, 2, 0)             # (H, W, 37) uint16

        mask = tifffile.imread(mask_file)
        with open(label_file) as f:
            cell_labels = [int(line.strip()) for line in f.readlines()]

        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids > 0]

        for cell_id in cell_ids:
            h5_path = patch_dir / f"{img_name}_{int(cell_id):06d}.h5"
            if h5_path.exists():
                continue  # crash-safe skip

            ys, xs = np.where(mask == cell_id)
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            crop = img[y_min:y_max+1, x_min:x_max+1, :]  # (h, w, 37) uint16

            # Safety: centre-crop if exceeds patch_size
            ch, cw = crop.shape[:2]
            if ch > args.patch_size or cw > args.patch_size:
                cy, cx = ch // 2, cw // 2
                half   = args.patch_size // 2
                crop   = crop[max(0,cy-half):cy+half,
                              max(0,cx-half):cx+half, :]
                ch, cw = crop.shape[:2]

            # Symmetric zero-padding to patch_size
            pad_h = args.patch_size - ch
            pad_w = args.patch_size - cw
            padded = np.pad(crop,
                            ((pad_h//2, pad_h-pad_h//2),
                             (pad_w//2, pad_w-pad_w//2),
                             (0, 0)))  # (patch_size, patch_size, 37)

            # Build binary cell mask (1 = cell pixels, 0 = background/padding)
            cell_mask_crop = (mask[y_min:y_max+1, x_min:x_max+1] == cell_id).astype(np.uint8)
            cell_mask_pad  = np.pad(cell_mask_crop,
                                    ((pad_h//2, pad_h-pad_h//2),
                                     (pad_w//2, pad_w-pad_w//2)))

            # Get label
            label = get_label(int(cell_id), cell_labels, label_map)

            # Save .h5 — one dataset per marker + mask + label metadata
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('mask',      data=cell_mask_pad, dtype=np.uint8)
                f.attrs['label']    = label
                f.attrs['image_id'] = img_name
                f.attrs['cell_id']  = int(cell_id)
                for c, marker_name in enumerate(clean_names):
                    f.create_dataset(marker_name,
                                     data=padded[:, :, c],
                                     dtype=np.uint16)

            n_cells_total += 1

    print(f"\nPatch extraction complete:")
    print(f"  {n_cells_total:,} cells saved as .h5 | {n_skipped} images skipped")
    print(f"  Output: {patch_dir}")
    return patch_dir


#  Step 2: Feature Extraction (authors' DataLoader approach) 

class IMMUcanPatchDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset adapting the authors' CellPhenotypingDataset for IMMUcan .h5 patches.
    Replicates the exact normalisation from the authors' tutorial:
        marker = raw_uint16 / marker_max_values   (scale to [0,1])
        marker = (marker - marker_mean) / marker_std   (standardise)
    """
    def __init__(self, patch_dir, marker_info_df, marker_names,
                 max_values=65535.0):
        self.patch_dir    = Path(patch_dir)
        self.patch_list   = sorted(os.listdir(patch_dir))
        self.marker_names = marker_names
        self.max_values   = max_values
        # Build fast lookup from marker_info_df
        self.marker_meta  = {
            row['marker_name']: (row['marker_id'], row['marker_mean'], row['marker_std'])
            for _, row in marker_info_df.iterrows()
        }

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        patch_path = self.patch_dir / self.patch_list[idx]
        with h5py.File(patch_path, 'r') as f:
            label    = f.attrs['label']
            image_id = f.attrs['image_id']
            cell_id  = int(f.attrs['cell_id'])

            patches  = []
            mids     = []
            for m in self.marker_names:
                mid, mean, std = self.marker_meta[m]
                raw    = f[m][:].astype(np.float32)
                scaled = raw / self.max_values            # scale to [0,1]
                normed = (scaled - mean) / (std if std > 0 else 1.0)
                patches.append(torch.tensor(normed, dtype=torch.float32))
                mids.append(int(mid))

        patch_tensor = torch.stack(patches, dim=0)     # (C, H, W)
        return patch_tensor, torch.tensor(mids), label, image_id, cell_id, self.patch_list[idx]


def extract_features_from_patches(args, patch_dir, marker_info_df,
                                   clean_names, model):
    """
    Extract KRONOS features from saved .h5 patches using DataLoader.
    Replicates CellPhenotyping.feature_extraction() from the authors' tutorial.
    Saves one .npy file per cell in features/.
    """
    feature_dir = Path(args.output_dir) / 'features'
    feature_dir.mkdir(parents=True, exist_ok=True)

    dataset = IMMUcanPatchDataset(patch_dir, marker_info_df, clean_names,
                                   max_values=MARKER_MAX_VALUES)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         num_workers=4, shuffle=False)

    print(f"\nExtracting features from {len(dataset):,} patches → {feature_dir}")
    print("=" * 60)

    metadata_rows = []
    model.eval()

    for patches, mids, labels, image_ids, cell_ids, patch_names in tqdm(
            loader, desc="Feature extraction"):
        patches = patches.to(args.device)
        with torch.no_grad():
            patch_emb, _, _ = model(patches)    # (B, 384)
        feats = patch_emb.cpu().numpy()

        for i, patch_name in enumerate(patch_names):
            feat_path = feature_dir / patch_name.replace('.h5', '.npy')
            np.save(feat_path, feats[i])
            metadata_rows.append({
                'patch_name': patch_name,
                'image_id':   image_ids[i],
                'cell_id':    int(cell_ids[i]),
                'label':      labels[i],
            })
        torch.cuda.empty_cache()

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(Path(args.output_dir) / 'feature_metadata.csv', index=False)
    print(f"Feature extraction complete: {len(metadata_df):,} cells")
    return feature_dir, metadata_df


#  Step 3: LogReg + Optuna Classifier (authors' approach)

def run_logreg_optuna(args, folds, feature_dir, metadata_df):
    """
    Train Logistic Regression with Optuna hyperparameter tuning.
    Replicates CellPhenotyping.train_classification_model() from the authors' tutorial.

    Key differences from our RF approach:
    - LogReg instead of RF
    - Optuna searches C in [C_low, C_high] = [1e-10, 1e5]
    - StandardScaler applied to features
    - max_cells_per_type=1000 balancing (optional, set None to disable)
    - Saves predictions in same spCellEval format as run_kronos.py for direct comparison
    """
    import optuna
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    method_name = 'KRONOS_original_supervised'
    out_dir     = Path(args.spceleval_dir or args.output_dir) / method_name / 'level3'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build fast lookup: patch_name → (image_id, cell_id, label)
    meta_lookup = {
        row['patch_name']: (row['image_id'], int(row['cell_id']), row['label'])
        for _, row in metadata_df.iterrows()
    }
    # Build per-image patch list
    img_to_patches = defaultdict(list)
    for patch_name, (img_id, cell_id, label) in meta_lookup.items():
        img_to_patches[img_id].append((patch_name, cell_id, label))

    print(f"\nLogReg + Optuna | n_trials={args.n_trials} | folds={args.n_folds}")
    print("=" * 60)

    fold_times   = []
    fold_reports = {}

    for fold_idx in range(args.n_folds):
        fold_start   = time.time()
        train_images = folds[fold_idx]['train']
        test_images  = folds[fold_idx]['test']
        print(f"\n── Fold {fold_idx} ──────────────────────")

        #  Collect train features 
        X_train_list, y_train_list = [], []
        for img_id in train_images:
            for patch_name, cell_id, label in img_to_patches.get(img_id, []):
                if label == 'Unknown':
                    continue
                feat_path = feature_dir / patch_name.replace('.h5', '.npy')
                if feat_path.exists():
                    X_train_list.append(np.load(feat_path))
                    y_train_list.append(label)

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)

        # Optional: balance classes by max_cells_per_type
        if args.max_cells_per_type is not None:
            idx_balanced = []
            for cls in np.unique(y_train):
                idx_cls = np.where(y_train == cls)[0]
                chosen  = np.random.RandomState(42).choice(
                    idx_cls,
                    min(len(idx_cls), args.max_cells_per_type),
                    replace=False
                )
                idx_balanced.extend(chosen)
            X_train = X_train[idx_balanced]
            y_train = y_train[idx_balanced]

        print(f"  Train: {len(X_train):,} cells | {len(train_images)} images")

        #  StandardScaler (authors use this before LogReg) 
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        #  Optuna hyperparameter search 
        def objective(trial):
            C = trial.suggest_float('C', args.c_low, args.c_high, log=True)
            clf = LogisticRegression(
                C=C, max_iter=args.max_iter,
                random_state=42, class_weight='balanced',
                solver='lbfgs', multi_class='multinomial',
                n_jobs=args.n_jobs
            )
            # Use 20% of train set as quick validation
            n_val   = max(1, int(0.2 * len(X_train)))
            idx_val = np.random.RandomState(fold_idx).choice(
                len(X_train), n_val, replace=False)
            idx_tr  = np.setdiff1d(np.arange(len(X_train)), idx_val)
            clf.fit(X_train[idx_tr], y_train[idx_tr])
            y_pred = clf.predict(X_train[idx_val])
            from sklearn.metrics import f1_score
            return f1_score(y_train[idx_val], y_pred,
                            average='macro', zero_division=0)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
        best_C = study.best_params['C']
        print(f"  Best C={best_C:.2e} (Optuna, {args.n_trials} trials)")

        #  Train final model with best C 
        clf = LogisticRegression(
            C=best_C, max_iter=args.max_iter,
            random_state=42, class_weight='balanced',
            solver='lbfgs', multi_class='multinomial',
            n_jobs=args.n_jobs
        )
        clf.fit(X_train, y_train)

        #  Predict test cells 
        fold_preds = []
        for img_id in test_images:
            for patch_name, cell_id, label in img_to_patches.get(img_id, []):
                feat_path = feature_dir / patch_name.replace('.h5', '.npy')
                if not feat_path.exists():
                    continue
                feat   = scaler.transform(np.load(feat_path).reshape(1, -1))
                y_pred = clf.predict(feat)[0]
                y_prob = clf.predict_proba(feat)[0].max()
                fold_preds.append({
                    'image_id':            img_id,
                    'cell_id':             cell_id,
                    'fold':                fold_idx,
                    'true_phenotype':      label,
                    'predicted_phenotype': y_pred,
                    'confidence':          float(y_prob),
                })

        fold_df = pd.DataFrame(fold_preds)
        fold_df.to_csv(out_dir / f"predictions_{fold_idx}.csv", index=False)

        fold_time = time.time() - fold_start
        fold_times.append(fold_time)

        known  = fold_df[fold_df['true_phenotype'] != 'Unknown']
        report = classification_report(
            known['true_phenotype'], known['predicted_phenotype'],
            output_dict=True, zero_division=0
        )
        fold_reports[fold_idx] = report
        print(f"  Test:     {len(fold_df):,} cells | {len(test_images)} images")
        print(f"  Accuracy: {report['accuracy']:.3f} | "
              f"Macro F1: {report['macro avg']['f1-score']:.3f}")
        print(f"  Time:     {fold_time:.1f}s")

    # Save fold_times.txt
    with open(out_dir / 'fold_times.txt', 'w') as f:
        for i, t in enumerate(fold_times):
            f.write(f"fold_{i}: {t:.2f}s\n")
        f.write(f"total: {sum(fold_times):.2f}s\n")
        f.write(f"mean:  {np.mean(fold_times):.2f}s\n")

    # Summary
    accs  = [fold_reports[i]['accuracy']                 for i in range(args.n_folds)]
    f1s   = [fold_reports[i]['macro avg']['f1-score']    for i in range(args.n_folds)]
    wf1s  = [fold_reports[i]['weighted avg']['f1-score'] for i in range(args.n_folds)]
    print(f"\n{method_name} Results (LogReg + Optuna):")
    print(f"  Accuracy:    {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Macro F1:    {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"  Weighted F1: {np.mean(wf1s):.3f} ± {np.std(wf1s):.3f}")
    print(f"\n  Compare with KRONOS RF:  Acc=0.647, F1=0.434")
    print(f"  Compare with Eva v7 RF:  Acc=0.697, F1=0.458")
    print(f"  Output: {out_dir}")


#  Model loading 

def load_kronos_model(kronos_dir, device):
    sys.path.insert(0, str(kronos_dir))
    from kronos import create_model_from_pretrained
    cache_dir = Path(kronos_dir) / 'model_assets'
    model, precision, embedding_dim = create_model_from_pretrained(
        checkpoint_path='hf_hub:MahmoodLab/kronos',
        cache_dir=str(cache_dir),
    )
    model = model.to(device)
    model.eval()
    print(f"KRONOS loaded | device={device} | embedding_dim={embedding_dim}")
    return model


#  Argument Parser 

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_kronos_original.py',
        description='KRONOS Original Tutorial Approach — IMMUcan Benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)

    for m in ['extract_patches', 'extract_features', 'classify', 'all']:
        p = sub.add_parser(m, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                           help={
                               'extract_patches':  'Extract .h5 patch files per cell',
                               'extract_features': 'Extract KRONOS features from .h5 patches',
                               'classify':         'LogReg + Optuna on saved features',
                               'all':              'extract_patches → extract_features → classify',
                           }[m])

        # Paths
        p.add_argument('--data-dir',    required=False, default=None,
                       help='IMMUcan data root (required for extract_patches/all)')
        p.add_argument('--kronos-dir',  required=False, default=None,
                       help='KRONOS repo dir (required for extract_features/all)')
        p.add_argument('--marker-meta', required=False, default=None,
                       help='Path to marker_metadata.csv')
        p.add_argument('--output-dir',  required=True,
                       help='Output directory')
        p.add_argument('--spceleval-dir', default=None,
                       help='spCellEval output root (defaults to output-dir)')

        # Extraction
        p.add_argument('--patch-size',  type=int, default=64)
        p.add_argument('--batch-size',  type=int, default=16)
        p.add_argument('--exclude-markers', nargs='+', default=DEFAULT_EXCLUDE)
        p.add_argument('--device',      default='cuda:0')

        # LogReg / Optuna
        p.add_argument('--n-folds',           type=int,   default=5)
        p.add_argument('--n-trials',          type=int,   default=25,
                       help='Optuna trials for C search')
        p.add_argument('--c-low',             type=float, default=1e-10,
                       help='Lower bound for LogReg C parameter')
        p.add_argument('--c-high',            type=float, default=1e5,
                       help='Upper bound for LogReg C parameter')
        p.add_argument('--max-iter',          type=int,   default=10000,
                       help='LogReg max iterations')
        p.add_argument('--max-cells-per-type',type=int,   default=None,
                       help='Max cells per type for training (None = no limit)')
        p.add_argument('--n-jobs',            type=int,   default=-1)

    return parser


#  Main 

def main():
    parser = build_parser()
    args   = parser.parse_args()

    total_start = time.time()
    print(f"\n{'='*60}")
    print(f"run_kronos_original.py | mode={args.mode}")
    print(f"KRONOS authors' approach: LogReg + Optuna (not RF)")
    print(f"{'='*60}")

    # Setup markers
    clean_indices, clean_names = (
        None, None
    ) if args.mode == 'classify' else (
        lambda e: (
            [i for i, m in enumerate(ALL_BIOMARKERS) if m not in set(e)],
            [m for m in ALL_BIOMARKERS if m not in set(e)]
        )
    )(args.exclude_markers)

    if clean_names is None:
        clean_indices, clean_names = (
            [i for i, m in enumerate(ALL_BIOMARKERS) if m not in set(DEFAULT_EXCLUDE)],
            [m for m in ALL_BIOMARKERS if m not in set(DEFAULT_EXCLUDE)]
        )

    # Build marker info CSV (needed for DataLoader normalisation)
    marker_info_path = Path(args.output_dir) / 'marker_info_with_metadata.csv'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.marker_meta and not marker_info_path.exists():
        marker_info_df = build_marker_info_csv(
            clean_names, args.marker_meta, marker_info_path
        )
    elif marker_info_path.exists():
        marker_info_df = pd.read_csv(marker_info_path)
    else:
        raise ValueError("--marker-meta required for first run")

    #  Extract patches 
    if args.mode in ('extract_patches', 'all'):
        if not args.data_dir:
            raise ValueError("--data-dir required for extract_patches")
        label_map   = load_label_map(args.data_dir)
        _, all_images = load_folds(args.data_dir, args.n_folds)
        patch_dir   = extract_patches(
            args, all_images, clean_indices, clean_names, label_map
        )
    else:
        patch_dir = Path(args.output_dir) / 'patches'

    #  Extract features 
    if args.mode in ('extract_features', 'all'):
        if not args.kronos_dir:
            raise ValueError("--kronos-dir required for extract_features")
        model       = load_kronos_model(args.kronos_dir, args.device)
        feature_dir, metadata_df = extract_features_from_patches(
            args, patch_dir, marker_info_df, clean_names, model
        )
    else:
        feature_dir  = Path(args.output_dir) / 'features'
        metadata_df  = pd.read_csv(Path(args.output_dir) / 'feature_metadata.csv')

    #  Classify 
    if args.mode in ('classify', 'all'):
        if not args.data_dir:
            raise ValueError("--data-dir required for classify")
        folds, _ = load_folds(args.data_dir, args.n_folds)
        run_logreg_optuna(args, folds, feature_dir, metadata_df)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"run_kronos_original.py complete | total: {total_time:.1f}s "
          f"({total_time/3600:.2f}h)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()