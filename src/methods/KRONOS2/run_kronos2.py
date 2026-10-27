#!/usr/bin/env python3
"""
run_kronos2.py — KRONOS2 config-driven pipeline (dataset-agnostic)
====================================================================

KRONOS2 (MahmoodLab/KRONOS2 on HuggingFace) is the authors' next-generation,
marker-aware successor to KRONOS. Same overall shape as run_kronos.py
(per-cell patch -> feature extraction -> classifier head), but the model
itself differs:

  - Loaded via transformers.AutoModel(trust_remote_code=True) -- no local
    model_assets/ dir, fetched from HF on first run.
  - Marker-aware: each channel's name is passed alongside the pixels, and
    model.preprocess() applies a per-marker z-score keyed against its own
    bundled 288-marker vocabulary -- name aliasing (kronos2.json's
    name_mappings) is needed where our spelling doesn't match the vocabulary.
  - ViT-B/16, 768-dim CLS output, fp32 inference (the authors' own
    reproducibility requirement).
  - Classifier head is LogReg with an Optuna search over C per fold (not the
    fixed-C LogReg shared/utils_foundational_models.run_supervised uses for
    the other FM methods) -- this is KRONOS2's own established approach, not
    something to unify away.

Environment note: KRONOS2 pins torch==2.6.0+cu124 in its own requirements.txt,
which ships no kernels for this machine's Blackwell GPU (sm_120). Run in a
dedicated env (not the shared `stellar` env used by KRONOS1/VirTues/Stellar):
    micromamba create -p /tmp/kronos2_env python=3.11
    /tmp/kronos2_env/bin/pip install -r requirements.txt  # from the HF repo
    /tmp/kronos2_env/bin/pip uninstall -y torch torchvision xformers
    /tmp/kronos2_env/bin/pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
    /tmp/kronos2_env/bin/pip install h5py scikit-learn optuna

Usage
-----
python run_kronos2.py all \\
    --dataset-path /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan \\
    --config       src/methods/configs/kronos2.json \\
    --output-dir   /home/juliaoesterle/results/kronos2/IMMUcan \\
    --device       cuda:0
"""

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

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import ndimage

warnings.filterwarnings('ignore')


#  CONFIG LOADING

def _rebase_local_paths(obj, prefix="/home/juliaoesterle", env_var="SPCELLEVAL_ROOT"):
    """kronos2.json hardcodes absolute paths rooted at beauty (/home/juliaoesterle).
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


def get_clean_markers(biomarkers, exclude):
    exclude_set = set(exclude)
    clean_idx   = [i for i, m in enumerate(biomarkers) if m not in exclude_set]
    clean_names = [m for m in biomarkers if m not in exclude_set]
    print(f"Using {len(clean_names)} markers (excluded: {sorted(exclude_set)})")
    return clean_idx, clean_names


#  DATA LOADING + PATCH EXTRACTION

def _file_stem(name):
    for ext in ('.ome.tiff', '.ome.tif', '.tiff', '.tif'):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def find_image(img_dir, img_id):
    files = [p for p in img_dir.iterdir() if not p.name.startswith('.')]
    exact = [p for p in files if _file_stem(p.name) == img_id]
    if exact:
        return exact[0]
    prefixed = [p for p in files if _file_stem(p.name).startswith(f'{img_id}_')]
    return prefixed[0] if len(prefixed) == 1 else None


def find_segmentation(seg_dir, img_id):
    for ext in ('.tiff', '.tif'):
        if (seg_dir / f'{img_id}{ext}').exists():
            return seg_dir / f'{img_id}{ext}'
    if (seg_dir / img_id).is_dir():
        seg_files = sorted(p for p in (seg_dir / img_id).rglob('segmentationMap.tif')
                           if not p.name.startswith('.'))
        if seg_files:
            return seg_files[0]
    return None


def load_data(cfg, data_root, folds_dir, include_validation=False):
    """
    Returns meta_df[image_id, cell_id, label], folds (cell-level: sets of
    (image_id, cell_id) tuples read from the per-fold fold_{i}_train.csv/
    fold_{i}_test.csv files), img_paths, seg_paths.
    """
    quant_csv = data_root / 'quantification' / 'processed' / f'{data_root.name}_quantification.csv'
    raw_df = pd.read_csv(quant_csv, usecols=['sample_id', 'cell_id', 'cell_type'])
    raw_df['sample_id'] = raw_df['sample_id'].astype(str).str.replace('.csv', '', regex=False)

    def load_keys(path):
        df = pd.read_csv(path, usecols=['sample_id', 'cell_id'])
        df['sample_id'] = df['sample_id'].astype(str).str.replace('.csv', '', regex=False)
        return set(zip(df['sample_id'], df['cell_id']))

    folds  = []
    fold_i = 1
    while (folds_dir / f'fold_{fold_i}_train.csv').exists():
        train_keys = load_keys(folds_dir / f'fold_{fold_i}_train.csv')
        if include_validation and (folds_dir / f'fold_{fold_i}_validation.csv').exists():
            train_keys |= load_keys(folds_dir / f'fold_{fold_i}_validation.csv')
        folds.append({
            'train': train_keys,
            'test':  load_keys(folds_dir / f'fold_{fold_i}_test.csv'),
        })
        fold_i += 1
    print(f"Loaded {len(folds)} folds (cell-level) from {folds_dir}")

    exclude_labels = set(cfg.get("exclude_labels", []))
    meta_df = raw_df[~raw_df['cell_type'].isin(exclude_labels)].copy()
    meta_df = meta_df.rename(columns={'sample_id': 'image_id', 'cell_type': 'label'})
    meta_df = meta_df[['image_id', 'cell_id', 'label']]
    print(f"Loaded metadata: {len(meta_df):,} cells "
          f"(dropped {len(raw_df) - len(meta_df):,} excluded-label cells)")

    img_dir   = data_root / 'raw_images' / 'multistack_tiffs'
    seg_dir   = data_root / 'segmentation'
    img_paths, seg_paths = {}, {}
    for img_id in sorted(meta_df['image_id'].unique()):
        img_path = find_image(img_dir, img_id)
        if img_path is not None:
            img_paths[img_id] = img_path
        seg_path = find_segmentation(seg_dir, img_id)
        if seg_path is not None:
            seg_paths[img_id] = seg_path
    print(f"Found {len(img_paths)} images, {len(seg_paths)} segmentation maps")

    return meta_df, folds, img_paths, seg_paths


def extract_patches(args, cfg, clean_indices, clean_names, meta_df, img_paths, seg_paths):
    """
    Raw per-marker bbox crops + binary mask + label, saved as one .h5 per
    cell for extract_features_from_patches().

    Uses scipy.ndimage.find_objects (one O(pixels) pass per image) instead of
    a per-cell np.where(mask==cell_id) full-image scan.

    Intensities are rounded (not truncated) to uint16 for h5 storage.
    """
    patch_dir = Path(args.output_dir) / 'patches'
    patch_dir.mkdir(parents=True, exist_ok=True)

    label_lookup = dict(zip(zip(meta_df['image_id'], meta_df['cell_id']), meta_df['label']))

    print(f"\nExtracting patches -> {patch_dir}")
    print(f"  patch_size={args.patch_size} | markers={len(clean_names)}")

    n_cells_total = n_skipped = 0
    for img_id, img_path in tqdm(sorted(img_paths.items()), desc="Patch extraction"):
        if img_id not in seg_paths:
            n_skipped += 1
            continue

        img_raw = tifffile.imread(img_path).astype(np.float32)   # (C, H, W)
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
                y_lo, x_lo = max(0, cy - half), max(0, cx - half)
                crop   = crop[y_lo:cy+half, x_lo:cx+half, :]
                y_min, x_min = y_min + y_lo, x_min + x_lo
                ch, cw = crop.shape[:2]
                y_max, x_max = y_min + ch - 1, x_min + cw - 1

            pad_h = args.patch_size - ch
            pad_w = args.patch_size - cw
            padded = np.pad(crop, ((pad_h//2, pad_h-pad_h//2),
                                   (pad_w//2, pad_w-pad_w//2), (0, 0)))

            cell_mask_crop = (mask[y_min:y_max+1, x_min:x_max+1] == cell_id).astype(np.uint8)
            cell_mask_pad  = np.pad(cell_mask_crop, ((pad_h//2, pad_h-pad_h//2),
                                                     (pad_w//2, pad_w-pad_w//2)))

            # 'Unknown' marks a segmented cell with no matching meta_df row
            # (distinct from the real 'undefined' class). Informational only
            # here -- run_supervised_optuna() filters on meta_df, not this
            # attr, so it has no effect on training or scoring.
            label = label_lookup.get((img_id, int(cell_id)), 'Unknown')

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

    print(f"Patch extraction complete: {n_cells_total:,} cells | {n_skipped} images skipped")
    return patch_dir


#  FEATURE EXTRACTION — KRONOS2-specific
# Reads the same .h5 schema for both datasets, normalises + extracts via
# KRONOS2's own marker-aware model.preprocess()/forward().

UINT16_SCALE = 65535.0  # matches sp_image.py's own _scaling_factor() convention


class Kronos2PatchDataset(torch.utils.data.Dataset):
    """Reads the .h5 patches saved by extract_patches(); returns the raw
    uint16 crop scaled to [0, ~1] (model.preprocess() does the marker-aware
    z-score afterwards, on the collated batch -- it needs the full
    marker_names list to key its lookup, so it can't run per-sample here)."""
    def __init__(self, patch_dir, marker_names, name_mappings):
        self.patch_dir      = Path(patch_dir)
        self.patch_list     = sorted(os.listdir(patch_dir))
        self.marker_names   = marker_names
        self.kronos2_names  = [name_mappings.get(m, m) for m in marker_names]

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        patch_path = self.patch_dir / self.patch_list[idx]
        with h5py.File(patch_path, 'r') as f:
            label    = f.attrs['label']
            image_id = f.attrs['image_id']
            cell_id  = int(f.attrs['cell_id'])
            arrs = [f[m][:].astype(np.float32) / UINT16_SCALE for m in self.marker_names]
        patch = np.stack(arrs, axis=0)  # (n_markers, P, P)
        return torch.from_numpy(patch), label, image_id, cell_id, self.patch_list[idx]


def extract_features_from_patches(args, patch_dir, clean_names, name_mappings, model):
    feature_dir = Path(args.output_dir) / 'features'
    feature_dir.mkdir(parents=True, exist_ok=True)

    dataset = Kronos2PatchDataset(patch_dir, clean_names, name_mappings)
    loader  = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    marker_names = dataset.kronos2_names
    print(f"  KRONOS2 marker names passed to the model: {marker_names}")

    print(f"\nExtracting features from {len(dataset):,} patches -> {feature_dir}")

    metadata_rows = []
    for patches, labels, image_ids, cell_ids, patch_names in tqdm(loader, desc="Feature extraction"):
        patches_np = patches.numpy()  # (B, n_markers, P, P), already /65535 scaled
        normed = model.preprocess(patches_np, marker_names, preferred_dapi=None)
        x = torch.from_numpy(normed).to(args.device)
        feats = model(x, marker_names).cpu().numpy()  # (B, 768), fp32
        for i, patch_name in enumerate(patch_names):
            np.save(feature_dir / patch_name.replace('.h5', '.npy'), feats[i])
            metadata_rows.append({'patch_name': patch_name, 'image_id': image_ids[i],
                                  'cell_id': int(cell_ids[i]), 'label': labels[i]})
        torch.cuda.empty_cache()

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(Path(args.output_dir) / 'feature_metadata.csv', index=False)
    print(f"Feature extraction complete: {len(metadata_df):,} cells")
    return feature_dir, metadata_df


#  SUPERVISED CLASSIFICATION — LogReg + Optuna, unified across fold_type

def run_supervised_optuna(args, folds, fold_type, metadata_df, feature_dir):
    """
    LogReg with an Optuna search over C per fold. fold_type='image':
    folds[i]['train'/'test'] are lists of image ids. fold_type='cell':
    folds[i]['train'/'test'] are sets of (image_id, cell_id) tuples.
    """
    import optuna
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    method_name = 'KRONOS2_supervised'
    out_dir     = Path(args.spceleval_dir or args.output_dir) / method_name / 'level3'
    out_dir.mkdir(parents=True, exist_ok=True)

    known = metadata_df[~metadata_df['label'].isin(args.exclude_labels | {'Unknown'})].copy()
    known['feat_path'] = known['patch_name'].str.replace('.h5', '.npy', regex=False)
    cell_keys = pd.Series(list(zip(known['image_id'], known['cell_id'])), index=known.index)

    print(f"\nLogReg + Optuna | n_trials={args.n_trials} | folds={len(folds)} | fold_type={fold_type}")

    fold_times, train_times, predict_times, fold_reports = [], [], [], {}
    for fold_idx, fold in enumerate(folds):
        fold_start = time.time()
        if fold_type == 'image':
            train_mask = known['image_id'].isin(fold['train'])
            test_mask  = known['image_id'].isin(fold['test'])
        else:
            train_mask = cell_keys.isin(fold['train'])
            test_mask  = cell_keys.isin(fold['test'])
        print(f"\n-- Fold {fold_idx} --")

        train_df = known[train_mask]
        X_train  = np.array([np.load(feature_dir / p) for p in train_df['feat_path']])
        y_train  = train_df['label'].values

        if args.max_cells_per_type is not None:
            idx_balanced = []
            for cls in np.unique(y_train):
                idx_cls = np.where(y_train == cls)[0]
                chosen  = np.random.RandomState(42).choice(
                    idx_cls, min(len(idx_cls), args.max_cells_per_type), replace=False)
                idx_balanced.extend(chosen)
            X_train = X_train[idx_balanced]
            y_train = y_train[idx_balanced]

        print(f"  Train: {len(X_train):,} cells | {train_df['image_id'].nunique()} images")

        train_start = time.time()
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        def objective(trial):
            C = trial.suggest_float('C', args.c_low, args.c_high, log=True)
            clf = LogisticRegression(C=C, max_iter=args.max_iter, random_state=42,
                                     class_weight='balanced', solver='lbfgs',
                                     n_jobs=args.n_jobs)
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
                                 n_jobs=args.n_jobs)
        clf.fit(X_train, y_train)
        train_time = time.time() - train_start

        predict_start = time.time()
        test_df = known[test_mask]
        X_test  = scaler.transform(np.array([np.load(feature_dir / p) for p in test_df['feat_path']]))
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        predict_time = time.time() - predict_start

        fold_df = pd.DataFrame({
            'image_id':            test_df['image_id'].values,
            'cell_id':             test_df['cell_id'].values,
            'fold':                fold_idx,
            'true_phenotype':      test_df['label'].values,
            'predicted_phenotype': y_pred,
            'confidence':          y_proba.max(axis=1),
        })
        fold_df.to_csv(out_dir / f"predictions_{fold_idx}.csv", index=False)

        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        train_times.append(train_time)
        predict_times.append(predict_time)

        report = classification_report(fold_df['true_phenotype'], fold_df['predicted_phenotype'],
                                       output_dict=True, zero_division=0)
        fold_reports[fold_idx] = report
        print(f"  Test:     {len(fold_df):,} cells | {test_df['image_id'].nunique()} images")
        print(f"  Accuracy: {report['accuracy']:.3f} | Macro F1: {report['macro avg']['f1-score']:.3f}")
        print(f"  Time:     {fold_time:.1f}s (train={train_time:.1f}s, predict={predict_time:.1f}s)")

    with open(out_dir / 'fold_times.txt', 'w') as f:
        for i, t in enumerate(fold_times):
            f.write(f"fold_{i}: {t:.2f}s (train={train_times[i]:.2f}s, predict={predict_times[i]:.2f}s)\n")
        f.write(f"total: {sum(fold_times):.2f}s\n")
        f.write(f"total_train: {sum(train_times):.2f}s\n")
        f.write(f"total_predict: {sum(predict_times):.2f}s\n")
        f.write(f"mean: {np.mean(fold_times):.2f}s\n")

    accs = [fold_reports[i]['accuracy']                 for i in range(len(folds))]
    f1s  = [fold_reports[i]['macro avg']['f1-score']    for i in range(len(folds))]
    wf1s = [fold_reports[i]['weighted avg']['f1-score'] for i in range(len(folds))]
    print(f"\n{method_name} Results (LogReg + Optuna):")
    print(f"  Accuracy:    {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"  Macro F1:    {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
    print(f"  Weighted F1: {np.mean(wf1s):.3f} +/- {np.std(wf1s):.3f}")
    print(f"  Output: {out_dir}")


#  MODEL LOADING

def load_kronos2_model(device):
    from transformers import AutoModel
    model = AutoModel.from_pretrained('MahmoodLab/KRONOS2', trust_remote_code=True, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"KRONOS2 loaded | device={device} | embedding_dim=768 | n_params={n_params:,}")
    return model


#  ARGUMENT PARSER

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_kronos2.py',
        description='KRONOS2 Foundation Model — config-driven pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ['extract', 'supervised', 'all']:
        p = sub.add_parser(
            mode,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            help={
                'extract':    'Extract patches + KRONOS2 features (reusable downstream)',
                'supervised': 'Logistic-regression classifier head + Optuna C search + 5-fold CV',
                'all':        'extract -> supervised in sequence',
            }[mode],
        )
        p.add_argument('--dataset-path', required=True,
                       help='Dataset directory (markers.txt, quantification/, raw_images/, segmentation/)')
        p.add_argument('--folds-dir', default=None,
                       help='k-fold directory, absolute or relative to --dataset-path')
        p.add_argument('--include-validation', action='store_true',
                       help='Add fold_{i}_validation.csv cells to the training set')
        p.add_argument('--config',  required=True, help='Path to kronos2.json')
        p.add_argument('--output-dir', required=True)
        p.add_argument('--spceleval-dir', default=None)
        # optional overrides (default None -> config)
        p.add_argument('--device',      default=None)
        p.add_argument('--patch-size',  type=int, default=None)
        p.add_argument('--batch-size',  type=int, default=None)
        p.add_argument('--n-folds',     type=int, default=None)
        p.add_argument('--n-trials',    type=int, default=None)
        p.add_argument('--c-low',       type=float, default=None)
        p.add_argument('--c-high',      type=float, default=None)
        p.add_argument('--max-iter',    type=int, default=None)
        p.add_argument('--max-cells-per-type', type=int, default=None)
        p.add_argument('--n-jobs',      type=int, default=None)
    return parser


#  MAIN

def main():
    parser = build_parser()
    args   = parser.parse_args()
    total_start = time.time()

    data_root    = Path(args.dataset_path)
    args.dataset = data_root.name
    cfg = load_config(args.config, args.dataset)

    def fill(attr, key, default=None):
        if getattr(args, attr) is None:
            setattr(args, attr, cfg.get(key, default))
    fill('device', 'device', 'cuda:0')
    fill('patch_size', 'patch_size', 64)
    fill('batch_size', 'batch_size', 16)
    fill('n_folds', 'n_folds', 5)
    fill('n_trials', 'n_trials', 25)
    fill('c_low', 'c_low', 1e-10)
    fill('c_high', 'c_high', 1e5)
    fill('max_iter', 'max_iter', 10000)
    fill('max_cells_per_type', 'max_cells_per_type', None)
    fill('n_jobs', 'n_jobs', -1)
    fill('folds_dir', 'folds_dir', 'quantification/processed/kfolds_StratifiedGroupKFold_level3')
    args.exclude_labels = set(cfg.get('exclude_labels', []))

    print(f"\n{'='*60}")
    print(f"run_kronos2.py | mode={args.mode} | dataset={args.dataset}")
    print(f"{'='*60}")

    name_mappings = cfg.get("name_mappings", {})
    with open(data_root / 'markers.txt') as f:
        biomarkers = [line.strip() for line in f if line.strip()]
    clean_indices, clean_names = get_clean_markers(biomarkers, cfg.get("exclude_markers", []))

    meta_df, folds, img_paths, seg_paths = load_data(
        cfg, data_root, data_root / args.folds_dir, args.include_validation)

    if args.mode in ('extract', 'all'):
        patch_dir = extract_patches(args, cfg, clean_indices, clean_names,
                                    meta_df, img_paths, seg_paths)
        model = load_kronos2_model(args.device)
        feature_dir, metadata_df = extract_features_from_patches(
            args, patch_dir, clean_names, name_mappings, model)
    else:
        feature_dir = Path(args.output_dir) / 'features'
        meta_path   = Path(args.output_dir) / 'feature_metadata.csv'
        metadata_df = pd.read_csv(meta_path)

    if args.mode in ('supervised', 'all'):
        run_supervised_optuna(args, folds, 'cell', metadata_df, feature_dir)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"run_kronos2.py complete | total: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
