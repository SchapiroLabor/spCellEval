#!/usr/bin/env python3
"""
run_kronos_initial.py — KRONOS Cell Phenotyping: Authors' Original Approach
============================================================================

Replicates the KRONOS authors' original pipeline from:
  tutorials/2-Cell-phenotyping.ipynb
adapted for IMMUcan and  predefined folds.json splits.

Key differences from run_kronos.py (our custom bbox+RF approach):
  - Patch extraction saves .h5 files per cell (authors' approach)
  - Feature extraction loads from .h5 files via DataLoader
  - Classifier: Logistic Regression + Optuna hyperparameter tuning (not RF)
  - Normalisation: raw / 65535.0 then per-marker mean/std from marker_metadata.csv
  - patch_size=32 (best on IMMUcan — tested 32, 64, 128; see kronos_patchsize_plots/)

Patch size comparison on IMMUcan (LogReg+Optuna, 5-fold CV):
  patch32:  Acc=0.720±0.012 | Macro F1=0.612±0.015  ← best
  patch64:  Acc=0.697±0.011 | Macro F1=0.588±0.016

Usage
-----
export TMPDIR=/home/juliaoesterle/tmp   # required if /tmp is full
source /home/juliaoesterle/eva/venv/bin/activate
cd /home/juliaoesterle/KRONOS_mine

# Full pipeline
python3 run_kronos_initial.py all \\
    --data-dir    /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan/ \\
    --kronos-dir  /home/juliaoesterle/KRONOS_mine \\
    --marker-meta /home/juliaoesterle/KRONOS_mine/model_assets/marker_metadata.csv \\
    --output-dir  /home/juliaoesterle/results/kronos_patch32 \\
    --patch-size  32 \\
    --n-jobs      4 \\
    --device      cuda:0

# Classify only (patches + features already extracted)
python3 run_kronos_initial.py classify \\
    --data-dir   /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan/ \\
    --output-dir /home/juliaoesterle/results/kronos_patch32 \\
    --n-trials   25 --n-folds 5 --n-jobs 4

Important notes
---------------
- patch_size=32 is best for IMMUcan (small cells ~10x10px)
- n_jobs=4 (NOT -1) — avoids memory thrashing on 362k cells x 384 features
- H5 patches = ~45GB per patch size — DELETE after feature extraction!
- IMMUcan images are uint16 but max ~1225 (not 65535) so raw/65535 → ~0.019

Output structure
----------------
{output_dir}/
├── patches/                         ← .h5 files per cell (delete after features!)
├── features/                        ← .npy per cell (384-dim)
├── feature_metadata.csv
├── marker_info_with_metadata.csv
└── KRONOS_original_supervised/
    └── level3/
        ├── predictions_0.csv ... predictions_4.csv
        └── fold_times.txt
"""

# Standard library 
import os
import sys
import time
import h5py
import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

# CUDA env vars 
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.3')
os.environ.setdefault('LD_LIBRARY_PATH',
                      '/usr/local/cuda-12.3/lib64:' +
                      os.environ.get('LD_LIBRARY_PATH', ''))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Third-party 
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Shared benchmark utilities 
sys.path.insert(0, str(Path(__file__).parent))
from utils_foundational_models import (
    load_label_map,
    load_folds,
    get_label,
    add_shared_args,
)

warnings.filterwarnings('ignore')


# Marker definitions 

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

# Markers with no KRONOS equivalent --> identity normalisation (mean=0, std=1)
KRONOS_NO_EQUIVALENT = {
    'SMA', 'PDGFRb', 'TCF7', 'CarbonicAnhydrase', 'Ecad', 'CD33', 'CD303'
}

MARKER_MAX_VALUES = 65535.0  # uint16 range — matches authors exactly


def get_clean_markers(exclude):
    exclude_set = set(exclude)
    clean_idx   = [i for i, m in enumerate(ALL_BIOMARKERS) if m not in exclude_set]
    clean_names = [m for m in ALL_BIOMARKERS if m not in exclude_set]
    print(f"Using {len(clean_names)} markers (excluded: {exclude_set})")
    return clean_idx, clean_names


# Marker info CSV 

def build_marker_info_csv(clean_names, marker_meta_path, output_path):
    """
    Build marker_info_with_metadata.csv mapping IMMUcan markers to KRONOS
    normalisation statistics from marker_metadata.csv.
    Markers without KRONOS equivalent use identity normalisation (mean=0, std=1).
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


# Patch Extraction 

def extract_patches(args, all_images, clean_indices, clean_names, label_map):
    """
    Extract cell-centred patches and save as .h5 files (authors' approach).
    Each cell → one .h5 file with raw uint16 intensities per marker + binary mask.
    460k cells × ~100KB each = ~45GB per patch size. Delete after features!
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

        img = np.load(npz_file)['data']       # (40, H, W) uint16
        img = img[clean_indices]               # (37, H, W) uint16
        img = img.transpose(1, 2, 0)           # (H, W, 37) uint16 

        mask = tifffile.imread(mask_file)
        with open(label_file) as f:
            cell_labels = [int(line.strip()) for line in f.readlines()]

        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids > 0]

        for cell_id in cell_ids:
            h5_path = patch_dir / f"{img_name}_{int(cell_id):06d}.h5"
            if h5_path.exists():
                continue

            ys, xs   = np.where(mask == cell_id)
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            crop = img[y_min:y_max+1, x_min:x_max+1, :]  # (h, w, C)

            ch, cw = crop.shape[:2]
            if ch > args.patch_size or cw > args.patch_size:
                cy, cx = ch//2, cw//2
                half   = args.patch_size // 2
                crop   = crop[max(0,cy-half):cy+half,
                              max(0,cx-half):cx+half, :]
                ch, cw = crop.shape[:2]

            pad_h  = args.patch_size - ch
            pad_w  = args.patch_size - cw
            padded = np.pad(crop, ((pad_h//2, pad_h-pad_h//2),
                                   (pad_w//2, pad_w-pad_w//2),
                                   (0, 0)))

            cell_mask_crop = (mask[y_min:y_max+1, x_min:x_max+1] == cell_id).astype(np.uint8)
            cell_mask_pad  = np.pad(cell_mask_crop,
                                    ((pad_h//2, pad_h-pad_h//2),
                                     (pad_w//2, pad_w-pad_w//2)))

            label = get_label(int(cell_id), cell_labels, label_map)

            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('mask',      data=cell_mask_pad, dtype=np.uint8)
                f.attrs['label']    = label
                f.attrs['image_id'] = img_name
                f.attrs['cell_id']  = int(cell_id)
                for c, marker_name in enumerate(clean_names):
                    f.create_dataset(marker_name, data=padded[:,:,c], dtype=np.uint16)

            n_cells_total += 1

    print(f"\nPatch extraction complete:")
    print(f"  {n_cells_total:,} cells saved as .h5 | {n_skipped} images skipped")
    print(f"  Output: {patch_dir}")
    print(f"  ⚠ Delete patches/ after feature extraction to save disk space!")
    return patch_dir


# Step 2: Feature Extraction 

class IMMUcanPatchDataset(Dataset):
    """
    Loads IMMUcan .h5 patches and applies KRONOS normalisation:
        normalised = (raw / 65535.0 - marker_mean) / marker_std
    """
    def __init__(self, patch_dir, marker_info_df, marker_names):
        self.patch_dir    = Path(patch_dir)
        self.patch_list   = sorted(p for p in os.listdir(patch_dir)
                                   if p.endswith('.h5'))
        self.marker_names = marker_names
        self.marker_meta  = {
            row['marker_name']: (int(row['marker_id']),
                                 float(row['marker_mean']),
                                 float(row['marker_std']))
            for _, row in marker_info_df.iterrows()
        }

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        with h5py.File(self.patch_dir / self.patch_list[idx], 'r') as f:
            label    = f.attrs['label']
            image_id = f.attrs['image_id']
            cell_id  = int(f.attrs['cell_id'])
            patches, mids = [], []
            for m in self.marker_names:
                mid, mean, std = self.marker_meta[m]
                raw    = f[m][:].astype(np.float32)
                normed = (raw / MARKER_MAX_VALUES - mean) / (std if std > 0 else 1.0)
                patches.append(torch.tensor(normed, dtype=torch.float32))
                mids.append(mid)
        return torch.stack(patches), torch.tensor(mids), label, image_id, cell_id, self.patch_list[idx]


def extract_features_from_patches(args, patch_dir, marker_info_df, clean_names, model):
    feature_dir = Path(args.output_dir) / 'features'
    feature_dir.mkdir(parents=True, exist_ok=True)

    dataset = IMMUcanPatchDataset(patch_dir, marker_info_df, clean_names)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         num_workers=4, shuffle=False, pin_memory=True)

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
            np.save(feature_dir / patch_name.replace('.h5', '.npy'), feats[i])
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


# Step 3: LogReg + Optuna 

def run_logreg_optuna(args, folds, feature_dir, metadata_df):
    """
    Logistic Regression + Optuna hyperparameter search on KRONOS features.
    Uses StandardScaler + C search in [1e-10, 1e5].
    n_jobs=4 recommended (not -1) to avoid memory thrashing on 362k cells.
    """
    import optuna
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir = Path(args.output_dir) / 'KRONOS_original_supervised' / 'level3'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-image patch lookup
    img_to_patches = defaultdict(list)
    for _, row in metadata_df.iterrows():
        img_to_patches[row['image_id']].append(
            (row['patch_name'], int(row['cell_id']), row['label'])
        )

    print(f"\nLogReg + Optuna | n_trials={args.n_trials} | folds={args.n_folds}")
    print("=" * 60)

    fold_times, fold_reports = [], {}

    for fold_idx in range(args.n_folds):
        # Skip if already done
        pred_path = out_dir / f'predictions_{fold_idx}.csv'
        if pred_path.exists():
            print(f"\n── Fold {fold_idx} — already done, skipping")
            df = pd.read_csv(pred_path)
            known = df[df['true_phenotype'] != 'Unknown']
            r = classification_report(known['true_phenotype'],
                                       known['predicted_phenotype'],
                                       output_dict=True, zero_division=0)
            fold_reports[fold_idx] = r
            continue

        t0           = time.time()
        train_images = folds[fold_idx]['train']
        test_images  = folds[fold_idx]['test']
        print(f"\n── Fold {fold_idx} ──────────────────────")
        print(f"  Train: {sum(len(img_to_patches[i]) for i in train_images):,} cells"
              f" | {len(train_images)} images")

        # Collect train features
        X_train_list, y_train_list = [], []
        for img_id in train_images:
            for patch_name, cell_id, label in img_to_patches.get(img_id, []):
                if label == 'Unknown': continue
                feat_path = feature_dir / patch_name.replace('.h5', '.npy')
                if feat_path.exists():
                    X_train_list.append(np.load(feat_path))
                    y_train_list.append(label)

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)

        # StandardScaler
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # Optuna C search
        def objective(trial):
            C   = trial.suggest_float('C', args.c_low, args.c_high, log=True)
            clf = LogisticRegression(C=C, max_iter=args.max_iter,
                                     random_state=42, class_weight='balanced',
                                     n_jobs=args.n_jobs)
            n_val   = max(1, int(0.2 * len(X_train)))
            idx_val = np.random.RandomState(fold_idx).choice(
                len(X_train), n_val, replace=False)
            idx_tr  = np.setdiff1d(np.arange(len(X_train)), idx_val)
            clf.fit(X_train[idx_tr], y_train[idx_tr])
            return f1_score(y_train[idx_val], clf.predict(X_train[idx_val]),
                            average='macro', zero_division=0)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
        best_C = study.best_params['C']
        print(f"  Best C={best_C:.2e} (Optuna, {args.n_trials} trials)")

        # Final model
        clf = LogisticRegression(C=best_C, max_iter=args.max_iter,
                                  random_state=42, class_weight='balanced',
                                  n_jobs=args.n_jobs)
        clf.fit(X_train, y_train)

        # Predict test cells
        fold_preds = []
        for img_id in test_images:
            for patch_name, cell_id, label in img_to_patches.get(img_id, []):
                feat_path = feature_dir / patch_name.replace('.h5', '.npy')
                if not feat_path.exists(): continue
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
        fold_df.to_csv(pred_path, index=False)

        elapsed = time.time() - t0
        fold_times.append(elapsed)

        known  = fold_df[fold_df['true_phenotype'] != 'Unknown']
        report = classification_report(known['true_phenotype'],
                                        known['predicted_phenotype'],
                                        output_dict=True, zero_division=0)
        fold_reports[fold_idx] = report
        print(f"  Test:     {len(fold_df):,} cells | {len(test_images)} images")
        print(f"  Accuracy: {report['accuracy']:.3f} | "
              f"Macro F1: {report['macro avg']['f1-score']:.3f}")
        print(f"  Time:     {elapsed:.1f}s")

    if fold_times:
        with open(out_dir / 'fold_times.txt', 'w') as f:
            for i, t in enumerate(fold_times):
                f.write(f"fold_{i}: {t:.2f}s\n")

    accs  = [fold_reports[i]['accuracy']                 for i in range(args.n_folds)
             if i in fold_reports]
    f1s   = [fold_reports[i]['macro avg']['f1-score']    for i in range(args.n_folds)
             if i in fold_reports]
    wf1s  = [fold_reports[i]['weighted avg']['f1-score'] for i in range(args.n_folds)
             if i in fold_reports]

    print(f"\nKRONOS original (LogReg+Optuna) Results:")
    print(f"  Accuracy:    {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Macro F1:    {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"  Weighted F1: {np.mean(wf1s):.3f} ± {np.std(wf1s):.3f}")
    print(f"  Output: {out_dir}")


# Model loading 

def load_kronos_model(kronos_dir, device):
    sys.path.insert(0, str(kronos_dir))
    from kronos import create_model_from_pretrained
    model, _, embedding_dim = create_model_from_pretrained(
        checkpoint_path='hf_hub:MahmoodLab/kronos',
        cache_dir=str(Path(kronos_dir) / 'model_assets'),
    )
    model = model.to(device)
    model.eval()
    print(f"KRONOS loaded | device={device} | embedding_dim={embedding_dim}")
    return model


# Argument Parser 

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_kronos_initial.py',
        description='KRONOS Authors\' Original Approach — IMMUcan Benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)

    for m in ['extract_patches', 'extract_features', 'classify', 'all']:
        p = sub.add_parser(m, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('--data-dir',    default=None)
        p.add_argument('--kronos-dir',  default=None)
        p.add_argument('--marker-meta', default=None)
        p.add_argument('--output-dir',  required=True)
        p.add_argument('--patch-size',  type=int,   default=32,
                       help='Cell patch size — 32 is best for IMMUcan')
        p.add_argument('--batch-size',  type=int,   default=16)
        p.add_argument('--exclude-markers', nargs='+', default=DEFAULT_EXCLUDE)
        p.add_argument('--device',      default='cuda:0')
        p.add_argument('--n-folds',     type=int,   default=5)
        p.add_argument('--n-trials',    type=int,   default=25)
        p.add_argument('--c-low',       type=float, default=1e-10)
        p.add_argument('--c-high',      type=float, default=1e5)
        p.add_argument('--max-iter',    type=int,   default=10000)
        p.add_argument('--n-jobs',      type=int,   default=4,
                       help='Keep at 4 — avoids memory thrashing on 362k cells')
    return parser


# Main 

def main():
    parser = build_parser()
    args   = parser.parse_args()

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"run_kronos_initial.py | mode={args.mode}")
    print(f"KRONOS authors' approach: LogReg + Optuna (not RF)")
    print(f"{'='*60}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    clean_indices, clean_names = get_clean_markers(args.exclude_markers)

    # Build/load marker info CSV
    marker_info_path = Path(args.output_dir) / 'marker_info_with_metadata.csv'
    if args.marker_meta and not marker_info_path.exists():
        marker_info_df = build_marker_info_csv(
            clean_names, args.marker_meta, marker_info_path)
    elif marker_info_path.exists():
        marker_info_df = pd.read_csv(marker_info_path)
    else:
        raise ValueError("--marker-meta required on first run")

    # Extract patches
    if args.mode in ('extract_patches', 'all'):
        if not args.data_dir:
            raise ValueError("--data-dir required")
        label_map     = load_label_map(args.data_dir)
        _, all_images = load_folds(args.data_dir, args.n_folds)
        patch_dir     = extract_patches(
            args, all_images, clean_indices, clean_names, label_map)
    else:
        patch_dir = Path(args.output_dir) / 'patches'

    # Extract features
    if args.mode in ('extract_features', 'all'):
        if not args.kronos_dir:
            raise ValueError("--kronos-dir required")
        model = load_kronos_model(args.kronos_dir, args.device)
        feature_dir, metadata_df = extract_features_from_patches(
            args, patch_dir, marker_info_df, clean_names, model)
    else:
        feature_dir = Path(args.output_dir) / 'features'
        metadata_df = pd.read_csv(Path(args.output_dir) / 'feature_metadata.csv')

    # Classify
    if args.mode in ('classify', 'all'):
        if not args.data_dir:
            raise ValueError("--data-dir required")
        folds, _ = load_folds(args.data_dir, args.n_folds)
        run_logreg_optuna(args, folds, feature_dir, metadata_df)

    print(f"\n{'='*60}")
    print(f"run_kronos_initial.py complete | {time.time()-t0:.1f}s")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()