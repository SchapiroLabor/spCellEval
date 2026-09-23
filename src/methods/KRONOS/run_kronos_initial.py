#!/usr/bin/env python3
"""
run_kronos_initial.py — KRONOS Cell Phenotyping: config-driven 
===================================================================================

Replicates the KRONOS authors' pipeline (tutorials/2-Cell-phenotyping.ipynb):
  - .h5 patch extraction per cell
  - feature extraction via DataLoader
  - Logistic Regression + Optuna hyperparameter tuning
  - marker_max_values normalisation + per-marker mean/std

All dataset-specific constants (biomarkers, name mappings, normalisation,
paths) now live in src/methods/configs/kronos.json and are selected with
--dataset. The pipeline logic is identical to the previous version. 

Usage
-----
# Full pipeline (extract patches -> features -> LogReg)
python3 run_kronos_initial.py all \
    --dataset     immucan \
    --config      src/methods/configs/kronos.json \
    --kronos-dir  /home/juliaoesterle/KRONOS_mine \
    --marker-meta /home/juliaoesterle/KRONOS_mine/model_assets/marker_metadata.csv \
    --output-dir  /home/juliaoesterle/results/kronos_original \
    --device      cuda:0

# Individual steps:
python3 run_kronos_initial.py extract_patches  --dataset immucan --config ... --output-dir ...
python3 run_kronos_initial.py extract_features --dataset immucan --config ... --output-dir ... --kronos-dir ...
python3 run_kronos_initial.py classify         --dataset immucan --config ... --output-dir ...

Output
------
{output_dir}/
  patches/                          .h5 per cell
  features/                         .npy per cell
  KRONOS_original_supervised/level3/predictions_{0..4}.csv + fold_times.txt
"""

# Standard library
import os
import sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)
import time
import h5py
import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

# CUDA env vars — must be set before torch import
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.3')
os.environ.setdefault('LD_LIBRARY_PATH',
                      '/usr/local/cuda-12.3/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Third-party
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import ndimage

# Shared benchmark utilities
from utils.utils_foundational_models import (
    load_label_map, load_folds, get_label, add_shared_args,
)

warnings.filterwarnings('ignore')

# CONFIG LOADING

def load_config(config_path: str, dataset: str) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    if dataset not in cfg["datasets"]:
        raise ValueError(f"Dataset '{dataset}' not in {config_path}. "
                         f"Available: {list(cfg['datasets'])}")
    merged = dict(cfg.get("defaults", {}))
    merged.update(cfg["datasets"][dataset])
    merged["dataset"] = dataset
    return merged


def get_clean_markers(biomarkers, exclude):
    exclude_set = set(exclude)
    clean_idx   = [i for i, m in enumerate(biomarkers) if m not in exclude_set]
    clean_names = [m for m in biomarkers if m not in exclude_set]
    return clean_idx, clean_names


# MARKER INFO CSV  (config-driven name mapping)

def build_marker_info_csv(clean_names, name_mappings, marker_meta_path, output_path):
    df_meta = pd.read_csv(marker_meta_path)
    lookup  = {row['marker_name'].upper(): (row['marker_id'], row['marker_mean'], row['marker_std'])
               for _, row in df_meta.iterrows()}

    rows = []
    for channel_id, m in enumerate(clean_names):
        kronos_name = name_mappings.get(m, m)
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
    print(f"Saved marker_info_with_metadata.csv ({len(df_out)} markers) -> {output_path}")
    return df_out


#PATCH EXTRACTION  (config-driven paths)

def extract_patches(args, cfg, all_images, clean_indices, clean_names, label_map):
    data_root = Path(cfg["data_root"])
    patch_dir = Path(args.output_dir) / 'patches'
    patch_dir.mkdir(parents=True, exist_ok=True)

    NPZ_DIR    = data_root / cfg["image_dir"]
    MASK_DIR   = data_root / cfg["segmentation_dir"]
    LABELS_DIR = data_root / cfg["label_dir"]

    print(f"\nExtracting patches -> {patch_dir}")
    print(f"  patch_size={args.patch_size} | markers={len(clean_names)}")
    print("=" * 60)

    n_cells_total = n_skipped = 0
    for img_name in tqdm(all_images, desc="Patch extraction"):
        npz_file   = NPZ_DIR    / f"{img_name}.npz"
        mask_file  = MASK_DIR   / f"{img_name}.tiff"
        label_file = LABELS_DIR / f"{img_name}.txt"
        if not all(f.exists() for f in [npz_file, mask_file, label_file]):
            n_skipped += 1
            continue

        img = np.load(npz_file)['data'][clean_indices].transpose(1, 2, 0)  # (H,W,C) uint16
        mask = tifffile.imread(mask_file)
        with open(label_file) as f:
            cell_labels = [int(line.strip()) for line in f.readlines()]

        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids > 0]

        for cell_id in cell_ids:
            h5_path = patch_dir / f"{img_name}_{int(cell_id):06d}.h5"
            if h5_path.exists():
                continue

            ys, xs = np.where(mask == cell_id)
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            crop = img[y_min:y_max+1, x_min:x_max+1, :]

            ch, cw = crop.shape[:2]
            if ch > args.patch_size or cw > args.patch_size:
                cy, cx = ch // 2, cw // 2
                half   = args.patch_size // 2
                crop   = crop[max(0, cy-half):cy+half, max(0, cx-half):cx+half, :]
                ch, cw = crop.shape[:2]

            pad_h = args.patch_size - ch
            pad_w = args.patch_size - cw
            padded = np.pad(crop, ((pad_h//2, pad_h-pad_h//2),
                                   (pad_w//2, pad_w-pad_w//2), (0, 0)))

            cell_mask_crop = (mask[y_min:y_max+1, x_min:x_max+1] == cell_id).astype(np.uint8)
            cell_mask_pad  = np.pad(cell_mask_crop, ((pad_h//2, pad_h-pad_h//2),
                                                     (pad_w//2, pad_w-pad_w//2)))

            label = get_label(int(cell_id), cell_labels, label_map)

            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('mask', data=cell_mask_pad, dtype=np.uint8)
                f.attrs['label']    = label
                f.attrs['image_id'] = img_name
                f.attrs['cell_id']  = int(cell_id)
                for c, marker_name in enumerate(clean_names):
                    f.create_dataset(marker_name, data=padded[:, :, c], dtype=np.uint16)

            n_cells_total += 1

    print(f"\nPatch extraction complete: {n_cells_total:,} cells | {n_skipped} images skipped")
    return patch_dir


# cHL_2_MIBI SUPPORT (config-driven)

def load_chl_metadata(cfg):
    """
    Load cHL_2_MIBI master quantification table, cell-level 5-fold split
    (row-index based, from fold_indices.json), image paths, and segmentation
    paths. Returns meta_df, folds, img_paths, seg_paths, row_map, where
    row_map: {row_idx: (sample_id, cell_id, cell_type)} keyed the same way
    fold_indices.json's train/test row indices are.
    """
    data_root = Path(cfg["data_root"])
    meta_df = pd.read_csv(data_root / cfg["quant_csv"])
    meta_df["sample_id"] = meta_df["sample_id"].astype(str).str.replace(".csv", "", regex=False)
    print(f"cHL: {len(meta_df):,} cells, {meta_df['cell_type'].nunique()} types")

    with open(data_root / cfg["folds_json"]) as f:
        folds = json.load(f)["folds"]
    print(f"cHL: {len(folds)} cell-level folds")

    img_dir   = data_root / cfg["image_dir"]
    img_paths = {p.name.replace("_stacked.ome.tif", ""): p
                 for p in sorted(img_dir.glob("*_stacked.ome.tif"))}
    print(f"cHL: {len(img_paths)} images")

    seg_paths = {}
    for img_id in img_paths:
        d = data_root / cfg["segmentation_dir"] / img_id
        if not d.exists():
            continue
        hits = [f for f in d.rglob("segmentationMap.tif") if not f.name.startswith(".")]
        if hits:
            seg_paths[img_id] = hits[0]
    print(f"cHL: {len(seg_paths)} segmentations found")

    row_map = {idx: (sid, int(cid), ct) for idx, sid, cid, ct in
               zip(meta_df.index, meta_df["sample_id"], meta_df["cell_id"], meta_df["cell_type"])}
    return meta_df, folds, img_paths, seg_paths, row_map


def extract_patches_chl(args, cfg, clean_indices, clean_names, meta_df, img_paths, seg_paths):
    """
    cHL equivalent of extract_patches(). Same .h5 schema (mask + one dataset
    per marker + label/image_id/cell_id attrs), so extract_features_from_patches()
    is reused unmodified.

    Uses scipy.ndimage.find_objects (one O(pixels) pass per image) instead of
    a per-cell np.where(mask==cell_id) full-image scan -- the latter measured
    at ~2 patches/s on cHL's 2048x1536 images (would be ~32h total); this is
    ~290 patches/s. IMMUcan's images are small enough (600x600) that the
    per-cell scan in extract_patches() above was never a practical problem,
    so that function is left as-is.

    Images here are float32 (already-processed intensities, not raw uint16
    counts like IMMUcan). Values are rounded (not truncated) to uint16 for
    h5 storage -- negligible precision loss given the value range (~0-3100),
    and keeps the same marker_max_values=65535 downstream normalisation
    convention used for IMMUcan.
    """
    patch_dir = Path(args.output_dir) / 'patches'
    patch_dir.mkdir(parents=True, exist_ok=True)

    label_lookup = {(sid, cid): ct for sid, cid, ct in
                     zip(meta_df["sample_id"], meta_df["cell_id"].astype(int), meta_df["cell_type"])}

    print(f"\nExtracting cHL patches -> {patch_dir}")
    print(f"  patch_size={args.patch_size} | markers={len(clean_names)}")
    print("=" * 60)

    n_cells_total = n_skipped = 0
    for img_id, img_path in tqdm(sorted(img_paths.items()), desc="Patch extraction (cHL)"):
        if img_id not in seg_paths:
            n_skipped += 1
            continue

        img_raw = tifffile.imread(img_path).astype(np.float32)   # (46, H, W)
        img     = img_raw[clean_indices].transpose(1, 2, 0)       # (H, W, C_clean)
        mask    = tifffile.imread(seg_paths[img_id]).astype(np.int32)

        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids > 0]
        objects  = ndimage.find_objects(mask)

        for cell_id in cell_ids:
            h5_path = patch_dir / f"{img_id}_{int(cell_id):06d}.h5"
            if h5_path.exists():
                continue

            box = objects[int(cell_id) - 1] if int(cell_id) - 1 < len(objects) else None
            if box is None:
                continue
            y_min, y_max = box[0].start, box[0].stop - 1
            x_min, x_max = box[1].start, box[1].stop - 1
            crop = img[y_min:y_max+1, x_min:x_max+1, :]

            ch, cw = crop.shape[:2]
            if ch > args.patch_size or cw > args.patch_size:
                cy, cx = ch // 2, cw // 2
                half   = args.patch_size // 2
                crop   = crop[max(0, cy-half):cy+half, max(0, cx-half):cx+half, :]
                ch, cw = crop.shape[:2]

            pad_h = args.patch_size - ch
            pad_w = args.patch_size - cw
            padded = np.pad(crop, ((pad_h//2, pad_h-pad_h//2),
                                   (pad_w//2, pad_w-pad_w//2), (0, 0)))

            cell_mask_crop = (mask[y_min:y_max+1, x_min:x_max+1] == cell_id).astype(np.uint8)
            cell_mask_pad  = np.pad(cell_mask_crop, ((pad_h//2, pad_h-pad_h//2),
                                                     (pad_w//2, pad_w-pad_w//2)))

            label = label_lookup.get((img_id, int(cell_id)), 'undefined')

            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('mask', data=cell_mask_pad, dtype=np.uint8)
                f.attrs['label']    = label
                f.attrs['image_id'] = img_id
                f.attrs['cell_id']  = int(cell_id)
                for c, marker_name in enumerate(clean_names):
                    f.create_dataset(marker_name,
                                     data=np.round(padded[:, :, c]).astype(np.uint16),
                                     dtype=np.uint16)

            n_cells_total += 1

    print(f"\ncHL patch extraction complete: {n_cells_total:,} cells | {n_skipped} images skipped")
    return patch_dir


# FEATURE EXTRACTION

class PatchDataset(torch.utils.data.Dataset):
    """Replicates the authors' CellPhenotypingDataset normalisation."""
    def __init__(self, patch_dir, marker_info_df, marker_names, max_values=65535.0):
        self.patch_dir    = Path(patch_dir)
        self.patch_list   = sorted(os.listdir(patch_dir))
        self.marker_names = marker_names
        self.max_values   = max_values
        self.marker_meta  = {row['marker_name']: (row['marker_id'], row['marker_mean'], row['marker_std'])
                             for _, row in marker_info_df.iterrows()}

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        patch_path = self.patch_dir / self.patch_list[idx]
        with h5py.File(patch_path, 'r') as f:
            label    = f.attrs['label']
            image_id = f.attrs['image_id']
            cell_id  = int(f.attrs['cell_id'])
            patches, mids = [], []
            for m in self.marker_names:
                mid, mean, std = self.marker_meta[m]
                raw    = f[m][:].astype(np.float32)
                scaled = raw / self.max_values
                normed = (scaled - mean) / (std if std > 0 else 1.0)
                patches.append(torch.tensor(normed, dtype=torch.float32))
                mids.append(int(mid))
        patch_tensor = torch.stack(patches, dim=0)
        return patch_tensor, torch.tensor(mids), label, image_id, cell_id, self.patch_list[idx]


def extract_features_from_patches(args, patch_dir, marker_info_df, clean_names, model):
    feature_dir = Path(args.output_dir) / 'features'
    feature_dir.mkdir(parents=True, exist_ok=True)

    dataset = PatchDataset(patch_dir, marker_info_df, clean_names, max_values=args.marker_max_values)
    loader  = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    print(f"\nExtracting features from {len(dataset):,} patches -> {feature_dir}")
    print("=" * 60)

    metadata_rows = []
    model.eval()
    for patches, mids, labels, image_ids, cell_ids, patch_names in tqdm(loader, desc="Feature extraction"):
        patches = patches.to(args.device)
        with torch.no_grad():
            patch_emb, _, _ = model(patches)
        feats = patch_emb.cpu().numpy()
        for i, patch_name in enumerate(patch_names):
            np.save(feature_dir / patch_name.replace('.h5', '.npy'), feats[i])
            metadata_rows.append({'patch_name': patch_name, 'image_id': image_ids[i],
                                  'cell_id': int(cell_ids[i]), 'label': labels[i]})
        torch.cuda.empty_cache()

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(Path(args.output_dir) / 'feature_metadata.csv', index=False)
    print(f"Feature extraction complete: {len(metadata_df):,} cells")
    return feature_dir, metadata_df


# LOGREG + OPTUNA

def run_logreg_optuna(args, folds, feature_dir, metadata_df):
    import optuna
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    method_name = 'KRONOS_original_supervised'
    out_dir     = Path(args.spceleval_dir or args.output_dir) / method_name / 'level3'
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_lookup = {row['patch_name']: (row['image_id'], int(row['cell_id']), row['label'])
                   for _, row in metadata_df.iterrows()}
    img_to_patches = defaultdict(list)
    for patch_name, (img_id, cell_id, label) in meta_lookup.items():
        img_to_patches[img_id].append((patch_name, cell_id, label))

    print(f"\nLogReg + Optuna | n_trials={args.n_trials} | folds={args.n_folds}")
    print("=" * 60)

    fold_times, train_times, predict_times, fold_reports = [], [], [], {}
    for fold_idx in range(args.n_folds):
        fold_start   = time.time()
        train_images = folds[fold_idx]['train']
        test_images  = folds[fold_idx]['test']
        print(f"\n-- Fold {fold_idx} --")

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

        if args.max_cells_per_type is not None:
            idx_balanced = []
            for cls in np.unique(y_train):
                idx_cls = np.where(y_train == cls)[0]
                chosen  = np.random.RandomState(42).choice(
                    idx_cls, min(len(idx_cls), args.max_cells_per_type), replace=False)
                idx_balanced.extend(chosen)
            X_train = X_train[idx_balanced]
            y_train = y_train[idx_balanced]

        print(f"  Train: {len(X_train):,} cells | {len(train_images)} images")

        # "train" = Optuna hyperparameter search (which itself fits many LogReg
        # models internally) + the final fit with the best C found.
        train_start = time.time()
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        def objective(trial):
            C = trial.suggest_float('C', args.c_low, args.c_high, log=True)
            clf = LogisticRegression(C=C, max_iter=args.max_iter, random_state=42,
                                     class_weight='balanced', solver='lbfgs',
                                     multi_class='multinomial', n_jobs=args.n_jobs)
            n_val   = max(1, int(0.2 * len(X_train)))
            idx_val = np.random.RandomState(fold_idx).choice(len(X_train), n_val, replace=False)
            idx_tr  = np.setdiff1d(np.arange(len(X_train)), idx_val)
            clf.fit(X_train[idx_tr], y_train[idx_tr])
            y_pred = clf.predict(X_train[idx_val])
            return f1_score(y_train[idx_val], y_pred, average='macro', zero_division=0)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
        best_C = study.best_params['C']
        print(f"  Best C={best_C:.2e} (Optuna, {args.n_trials} trials)")

        clf = LogisticRegression(C=best_C, max_iter=args.max_iter, random_state=42,
                                 class_weight='balanced', solver='lbfgs',
                                 multi_class='multinomial', n_jobs=args.n_jobs)
        clf.fit(X_train, y_train)
        train_time = time.time() - train_start

        predict_start = time.time()
        fold_preds = []
        for img_id in test_images:
            for patch_name, cell_id, label in img_to_patches.get(img_id, []):
                feat_path = feature_dir / patch_name.replace('.h5', '.npy')
                if not feat_path.exists():
                    continue
                feat   = scaler.transform(np.load(feat_path).reshape(1, -1))
                y_pred = clf.predict(feat)[0]
                y_prob = clf.predict_proba(feat)[0].max()
                fold_preds.append({'image_id': img_id, 'cell_id': cell_id, 'fold': fold_idx,
                                   'true_phenotype': label, 'predicted_phenotype': y_pred,
                                   'confidence': float(y_prob)})
        predict_time = time.time() - predict_start

        fold_df = pd.DataFrame(fold_preds)
        fold_df.to_csv(out_dir / f"predictions_{fold_idx}.csv", index=False)

        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        train_times.append(train_time)
        predict_times.append(predict_time)

        known  = fold_df[fold_df['true_phenotype'] != 'Unknown']
        report = classification_report(known['true_phenotype'], known['predicted_phenotype'],
                                       output_dict=True, zero_division=0)
        fold_reports[fold_idx] = report
        print(f"  Test:     {len(fold_df):,} cells | {len(test_images)} images")
        print(f"  Accuracy: {report['accuracy']:.3f} | Macro F1: {report['macro avg']['f1-score']:.3f}")
        print(f"  Time:     {fold_time:.1f}s (train={train_time:.1f}s, predict={predict_time:.1f}s)")

    with open(out_dir / 'fold_times.txt', 'w') as f:
        for i, t in enumerate(fold_times):
            f.write(f"fold_{i}: {t:.2f}s (train={train_times[i]:.2f}s, predict={predict_times[i]:.2f}s)\n")
        f.write(f"total: {sum(fold_times):.2f}s\n")
        f.write(f"total_train: {sum(train_times):.2f}s\n")
        f.write(f"total_predict: {sum(predict_times):.2f}s\n")
        f.write(f"mean:  {np.mean(fold_times):.2f}s\n")

    accs = [fold_reports[i]['accuracy']                 for i in range(args.n_folds)]
    f1s  = [fold_reports[i]['macro avg']['f1-score']    for i in range(args.n_folds)]
    wf1s = [fold_reports[i]['weighted avg']['f1-score'] for i in range(args.n_folds)]
    print(f"\n{method_name} Results (LogReg + Optuna):")
    print(f"  Accuracy:    {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Macro F1:    {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"  Weighted F1: {np.mean(wf1s):.3f} ± {np.std(wf1s):.3f}")
    print(f"  Output: {out_dir}")


def run_logreg_optuna_chl(args, folds, row_map, feature_dir):
    """
    cHL equivalent of run_logreg_optuna(). Folds are CELL-level (row indices
    into the master quantification table, from fold_indices.json), not
    image-level like IMMUcan. Same LogReg+Optuna/StandardScaler procedure
    and the same train/predict timing split otherwise.
    """
    import optuna
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    method_name = 'KRONOS_original_supervised'
    out_dir     = Path(args.spceleval_dir or args.output_dir) / method_name / 'level3'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLogReg + Optuna (cHL) | n_trials={args.n_trials} | folds={len(folds)}")
    print("=" * 60)

    def feat_path_for(row_idx):
        sid, cid, label = row_map[row_idx]
        return feature_dir / f"{sid}_{cid:06d}.npy", label

    fold_times, train_times, predict_times, fold_reports = [], [], [], {}
    for fold_idx, fold in enumerate(folds):
        fold_start = time.time()
        train_rows = [int(i) for i in fold['train']]
        test_rows  = [int(i) for i in fold['test']]
        print(f"\n-- Fold {fold_idx} --")

        X_train_list, y_train_list = [], []
        for idx in train_rows:
            fpath, label = feat_path_for(idx)
            if label in args.exclude_labels or not fpath.exists():
                continue
            X_train_list.append(np.load(fpath))
            y_train_list.append(label)

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)

        if args.max_cells_per_type is not None:
            idx_balanced = []
            for cls in np.unique(y_train):
                idx_cls = np.where(y_train == cls)[0]
                chosen  = np.random.RandomState(42).choice(
                    idx_cls, min(len(idx_cls), args.max_cells_per_type), replace=False)
                idx_balanced.extend(chosen)
            X_train = X_train[idx_balanced]
            y_train = y_train[idx_balanced]

        print(f"  Train: {len(X_train):,} cells")

        train_start = time.time()
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        def objective(trial):
            C = trial.suggest_float('C', args.c_low, args.c_high, log=True)
            clf = LogisticRegression(C=C, max_iter=args.max_iter, random_state=42,
                                     class_weight='balanced', solver='lbfgs',
                                     multi_class='multinomial', n_jobs=args.n_jobs)
            n_val   = max(1, int(0.2 * len(X_train)))
            idx_val = np.random.RandomState(fold_idx).choice(len(X_train), n_val, replace=False)
            idx_tr  = np.setdiff1d(np.arange(len(X_train)), idx_val)
            clf.fit(X_train[idx_tr], y_train[idx_tr])
            y_pred = clf.predict(X_train[idx_val])
            return f1_score(y_train[idx_val], y_pred, average='macro', zero_division=0)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
        best_C = study.best_params['C']
        print(f"  Best C={best_C:.2e} (Optuna, {args.n_trials} trials)")

        clf = LogisticRegression(C=best_C, max_iter=args.max_iter, random_state=42,
                                 class_weight='balanced', solver='lbfgs',
                                 multi_class='multinomial', n_jobs=args.n_jobs)
        clf.fit(X_train, y_train)
        train_time = time.time() - train_start

        predict_start = time.time()
        fold_preds = []
        for idx in test_rows:
            fpath, label = feat_path_for(idx)
            if label in args.exclude_labels or not fpath.exists():
                continue
            sid, cid, _ = row_map[idx]
            feat   = scaler.transform(np.load(fpath).reshape(1, -1))
            y_pred = clf.predict(feat)[0]
            y_prob = clf.predict_proba(feat)[0].max()
            fold_preds.append({'image_id': sid, 'cell_id': cid, 'row_idx': idx, 'fold': fold_idx,
                               'true_phenotype': label, 'predicted_phenotype': y_pred,
                               'confidence': float(y_prob)})
        predict_time = time.time() - predict_start

        fold_df = pd.DataFrame(fold_preds)
        fold_df.to_csv(out_dir / f"predictions_{fold_idx}.csv", index=False)

        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        train_times.append(train_time)
        predict_times.append(predict_time)

        report = classification_report(fold_df['true_phenotype'], fold_df['predicted_phenotype'],
                                       output_dict=True, zero_division=0)
        fold_reports[fold_idx] = report
        print(f"  Test:     {len(fold_df):,} cells")
        print(f"  Accuracy: {report['accuracy']:.3f} | Macro F1: {report['macro avg']['f1-score']:.3f}")
        print(f"  Time:     {fold_time:.1f}s (train={train_time:.1f}s, predict={predict_time:.1f}s)")

    with open(out_dir / 'fold_times.txt', 'w') as f:
        for i, t in enumerate(fold_times):
            f.write(f"fold_{i}: {t:.2f}s (train={train_times[i]:.2f}s, predict={predict_times[i]:.2f}s)\n")
        f.write(f"total: {sum(fold_times):.2f}s\n")
        f.write(f"total_train: {sum(train_times):.2f}s\n")
        f.write(f"total_predict: {sum(predict_times):.2f}s\n")
        f.write(f"mean:  {np.mean(fold_times):.2f}s\n")

    accs = [fold_reports[i]['accuracy']              for i in range(len(folds))]
    f1s  = [fold_reports[i]['macro avg']['f1-score'] for i in range(len(folds))]
    wf1s = [fold_reports[i]['weighted avg']['f1-score'] for i in range(len(folds))]
    print(f"\n{method_name} Results (cHL, LogReg + Optuna):")
    print(f"  Accuracy:    {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Macro F1:    {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"  Weighted F1: {np.mean(wf1s):.3f} ± {np.std(wf1s):.3f}")
    print(f"  Output: {out_dir}")


# MODEL LOADING

def load_kronos_model(kronos_dir, device):
    sys.path.insert(0, str(kronos_dir))
    from kronos import create_model_from_pretrained
    cache_dir = Path(kronos_dir) / 'model_assets'
    model, precision, embedding_dim = create_model_from_pretrained(
        checkpoint_path='hf_hub:MahmoodLab/kronos', cache_dir=str(cache_dir))
    model = model.to(device)
    model.eval()
    print(f"KRONOS loaded | device={device} | embedding_dim={embedding_dim}")
    return model


# ARGUMENT PARSER

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_kronos_initial.py',
        description='KRONOS config-driven pipeline (authors original approach)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)
    for m in ['extract_patches', 'extract_features', 'classify', 'all']:
        p = sub.add_parser(m, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('--dataset',  required=True, choices=['immucan', 'chl'])
        p.add_argument('--config',   required=True, help='Path to kronos.json')
        p.add_argument('--kronos-dir',  default=None)
        p.add_argument('--marker-meta', default=None)
        p.add_argument('--output-dir',  required=True)
        p.add_argument('--spceleval-dir', default=None)
        # optional overrides (default None -> config)
        p.add_argument('--patch-size',  type=int, default=None)
        p.add_argument('--batch-size',  type=int, default=None)
        p.add_argument('--exclude-markers', nargs='+', default=None)
        p.add_argument('--device',      default=None)
        p.add_argument('--n-folds',     type=int,   default=None)
        p.add_argument('--n-trials',    type=int,   default=None)
        p.add_argument('--c-low',       type=float, default=None)
        p.add_argument('--c-high',      type=float, default=None)
        p.add_argument('--max-iter',    type=int,   default=None)
        p.add_argument('--max-cells-per-type', type=int, default=None)
        p.add_argument('--n-jobs',      type=int,   default=None)
    return parser


# MAIN

def main():
    parser = build_parser()
    args   = parser.parse_args()

    cfg = load_config(args.config, args.dataset)

    def fill(attr, key, default=None):
        if getattr(args, attr) is None:
            setattr(args, attr, cfg.get(key, default))
    fill('patch_size', 'patch_size', 64)
    fill('batch_size', 'batch_size', 16)
    fill('device', 'device', 'cuda:0')
    fill('n_folds', 'n_folds', 5)
    fill('n_trials', 'n_trials', 25)
    fill('c_low', 'c_low', 1e-10)
    fill('c_high', 'c_high', 1e5)
    fill('max_iter', 'max_iter', 10000)
    fill('max_cells_per_type', 'max_cells_per_type', None)
    fill('n_jobs', 'n_jobs', -1)
    args.marker_max_values = cfg.get('marker_max_values', 65535.0)
    if args.exclude_markers is None:
        args.exclude_markers = cfg["exclude_markers"]
    args.exclude_labels = set(cfg.get('exclude_labels', []))

    total_start = time.time()
    print(f"\n{'='*60}")
    print(f"run_kronos_initial.py | mode={args.mode} | dataset={args.dataset}")
    print(f"KRONOS authors' approach: LogReg + Optuna (not RF)")
    print(f"{'='*60}")

    is_chl        = (args.dataset == 'chl')
    data_root     = cfg["data_root"]
    biomarkers    = cfg["biomarkers"]
    name_mappings = cfg.get("name_mappings", {})

    clean_indices, clean_names = get_clean_markers(biomarkers, args.exclude_markers)

    # Separate marker_info file per dataset so an immucan and a chl run under
    # the same --output-dir never read each other's stale marker metadata.
    marker_info_name = 'marker_info_with_metadata_chl.csv' if is_chl else 'marker_info_with_metadata.csv'
    marker_info_path = Path(args.output_dir) / marker_info_name
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.marker_meta and not marker_info_path.exists():
        marker_info_df = build_marker_info_csv(clean_names, name_mappings,
                                               args.marker_meta, marker_info_path)
    elif marker_info_path.exists():
        marker_info_df = pd.read_csv(marker_info_path)
    elif args.mode in ('extract_features', 'all'):
        raise ValueError("--marker-meta required for first run")
    else:
        marker_info_df = None

    # cHL metadata (images/segmentations/folds/row_map) needed by both the
    # extract_patches and classify steps -- load once up front.
    if is_chl:
        chl_meta_df, chl_folds, chl_img_paths, chl_seg_paths, chl_row_map = load_chl_metadata(cfg)

    if args.mode in ('extract_patches', 'all'):
        if is_chl:
            patch_dir = extract_patches_chl(args, cfg, clean_indices, clean_names,
                                            chl_meta_df, chl_img_paths, chl_seg_paths)
        else:
            label_map     = load_label_map(data_root)
            _, all_images = load_folds(data_root, args.n_folds)
            patch_dir     = extract_patches(args, cfg, all_images, clean_indices, clean_names, label_map)
    else:
        patch_dir = Path(args.output_dir) / 'patches'

    if args.mode in ('extract_features', 'all'):
        if not args.kronos_dir:
            raise ValueError("--kronos-dir required for extract_features")
        model = load_kronos_model(args.kronos_dir, args.device)
        feature_dir, metadata_df = extract_features_from_patches(
            args, patch_dir, marker_info_df, clean_names, model)
    else:
        feature_dir = Path(args.output_dir) / 'features'
        metadata_df = pd.read_csv(Path(args.output_dir) / 'feature_metadata.csv') \
                      if (Path(args.output_dir) / 'feature_metadata.csv').exists() else None

    if args.mode in ('classify', 'all'):
        if is_chl:
            run_logreg_optuna_chl(args, chl_folds, chl_row_map, feature_dir)
        else:
            folds, _ = load_folds(data_root, args.n_folds)
            run_logreg_optuna(args, folds, feature_dir, metadata_df)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"run_kronos_initial.py complete | total: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()