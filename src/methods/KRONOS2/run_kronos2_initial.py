#!/usr/bin/env python3
"""
run_kronos2_initial.py — KRONOS2 Cell Phenotyping: authors' approach
======================================================================

KRONOS2 (MahmoodLab/KRONOS2 on HuggingFace) is the authors' next-generation,
marker-aware successor to KRONOS. Same overall pipeline shape as
run_kronos_initial.py (KRONOS1): .h5 patch extraction per cell (cell-centered
crop, authors' cell-phenotyping approach) -> feature extraction -> Logistic
Regression + Optuna. What's different is the model itself:

  - Loaded via transformers.AutoModel(trust_remote_code=True), not the old
    `kronos` pip package -- no local model_assets/ dir needed, it's fetched
    from HF on first run (~440MB weights + bundled dinov2/ package).
  - Marker-aware: each channel's name is passed alongside the pixels, and the
    model's own model.preprocess() applies a per-marker z-score keyed against
    its bundled 288-marker vocabulary (marker_metadata.csv, shipped with the
    checkpoint) -- so there's no local marker_metadata.csv/marker_id lookup
    like KRONOS1's build_marker_info_csv(); name aliasing (kronos2.json's
    name_mappings) is still needed for markers whose IMMUcan spelling doesn't
    exactly match the vocabulary's spelling.
  - ViT-B/16, 768-dim CLS output (KRONOS1: ViT-S, 384-dim).
  - fp32 inference (the authors' own reproducibility requirement -- do not
    cast to fp16/bf16).

Environment note: KRONOS2 pins torch==2.6.0+cu124 in its own requirements.txt
for bit-exact reproducibility, but that build ships no kernels for this
machine's Blackwell GPU (sm_120) -- "CUDA error: no kernel image is available
for execution on the device". Verified as a hardware/build mismatch, not a
config issue. Run in a dedicated env (not the shared `stellar` env used by
KRONOS1/VirTues/Stellar, to avoid downgrading their already-working torch)
with torch==2.7.0+cu128 instead (matches what already works on this GPU
elsewhere in this project) and xformers omitted -- dinov2's attention module
falls back to a non-xformers path automatically when it's absent, verified
working. Set up via:
    micromamba create -p /tmp/kronos2_env python=3.11
    /tmp/kronos2_env/bin/pip install -r requirements.txt  # from the HF repo
    /tmp/kronos2_env/bin/pip uninstall -y torch torchvision xformers
    /tmp/kronos2_env/bin/pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
    /tmp/kronos2_env/bin/pip install h5py scikit-learn optuna

Usage
-----
python3 run_kronos2_initial.py all \
    --dataset    immucan \
    --config     src/methods/configs/kronos2.json \
    --output-dir /home/juliaoesterle/results/kronos2/immucan \
    --device     cuda:0

# Individual steps:
python3 run_kronos2_initial.py extract_patches  --dataset immucan --config ... --output-dir ...
python3 run_kronos2_initial.py extract_features --dataset immucan --config ... --output-dir ...
python3 run_kronos2_initial.py classify         --dataset immucan --config ... --output-dir ...

Output
------
{output_dir}/
  patches/                            .h5 per cell (same schema as KRONOS1)
  features/                           .npy per cell (768-dim)
  KRONOS2_original_supervised/level3/predictions_{0..4}.csv + fold_times.txt
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

# Third-party
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Shared benchmark utilities
from utils.utils_foundational_models import (
    load_label_map, load_folds, get_label,
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


# PATCH EXTRACTION (config-driven, IMMUcan)
# Identical to run_kronos_initial.py's extract_patches() -- patch extraction
# is model-agnostic (raw per-marker crops + binary mask + label), so nothing
# here needs to change for KRONOS2.

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


# FEATURE EXTRACTION -- KRONOS2-specific
# Reads the same .h5 schema as KRONOS1, but normalises + extracts via
# KRONOS2's own marker-aware model.preprocess()/forward() instead of
# KRONOS1's marker_metadata.csv-based per-marker mean/std.

# uint16 scaling divisor -- matches sp_image.py's own _scaling_factor()
# convention exactly (IMMUcan's raw npz data genuinely is uint16 counts).
UINT16_SCALE = 65535.0


class Kronos2PatchDataset(torch.utils.data.Dataset):
    """Reads the .h5 patches saved by extract_patches(); returns the raw
    uint16 crop scaled to [0, ~1] (KRONOS2's model.preprocess() does the
    marker-aware z-score afterwards, on the collated batch -- not per-sample
    here, since it needs the full marker_names list to key its lookup)."""
    def __init__(self, patch_dir, marker_names, name_mappings):
        self.patch_dir    = Path(patch_dir)
        self.patch_list   = sorted(os.listdir(patch_dir))
        self.marker_names = marker_names
        # the names actually passed to KRONOS2 (aliased where confirmed)
        self.kronos2_names = [name_mappings.get(m, m) for m in marker_names]

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
    print("=" * 60)

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


# LOGREG + OPTUNA
# Identical procedure to run_kronos_initial.py's run_logreg_optuna() -- only
# the feature dimensionality changed (768 vs 384), which this code doesn't
# need to know about. method_name is renamed so KRONOS1/KRONOS2 outputs never
# collide under the same --output-dir.

def run_logreg_optuna(args, folds, feature_dir, metadata_df):
    import optuna
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    method_name = 'KRONOS2_original_supervised'
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


# MODEL LOADING

def load_kronos2_model(device):
    from transformers import AutoModel
    model = AutoModel.from_pretrained('MahmoodLab/KRONOS2', trust_remote_code=True, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"KRONOS2 loaded | device={device} | embedding_dim=768 | n_params={n_params:,}")
    return model


# ARGUMENT PARSER

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_kronos2_initial.py',
        description='KRONOS2 authors\' approach (marker-aware) -- IMMUcan',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)
    for m in ['extract_patches', 'extract_features', 'classify', 'all']:
        p = sub.add_parser(m, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('--dataset',  required=True, choices=['immucan'])
        p.add_argument('--config',   required=True, help='Path to kronos2.json')
        p.add_argument('--output-dir',  required=True)
        p.add_argument('--spceleval-dir', default=None)
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
    if args.exclude_markers is None:
        args.exclude_markers = cfg["exclude_markers"]

    total_start = time.time()
    print(f"\n{'='*60}")
    print(f"run_kronos2_initial.py | mode={args.mode} | dataset={args.dataset}")
    print(f"KRONOS2 authors' approach: marker-aware ViT-B/16, LogReg + Optuna")
    print(f"{'='*60}")

    data_root     = cfg["data_root"]
    biomarkers    = cfg["biomarkers"]
    name_mappings = cfg.get("name_mappings", {})

    clean_indices, clean_names = get_clean_markers(biomarkers, args.exclude_markers)

    if args.mode in ('extract_patches', 'all'):
        label_map     = load_label_map(data_root)
        _, all_images = load_folds(data_root, args.n_folds)
        patch_dir     = extract_patches(args, cfg, all_images, clean_indices, clean_names, label_map)
    else:
        patch_dir = Path(args.output_dir) / 'patches'

    if args.mode in ('extract_features', 'all'):
        model = load_kronos2_model(args.device)
        feature_dir, metadata_df = extract_features_from_patches(
            args, patch_dir, clean_names, name_mappings, model)
    else:
        feature_dir = Path(args.output_dir) / 'features'
        meta_path = Path(args.output_dir) / 'feature_metadata.csv'
        metadata_df = pd.read_csv(meta_path) if meta_path.exists() else None

    if args.mode in ('classify', 'all'):
        folds, _ = load_folds(data_root, args.n_folds)
        run_logreg_optuna(args, folds, feature_dir, metadata_df)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"run_kronos2_initial.py complete | total: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
