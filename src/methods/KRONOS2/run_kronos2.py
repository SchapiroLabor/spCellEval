#!/usr/bin/env python3
"""
run_kronos2.py — KRONOS2 config-driven benchmark pipeline (dataset-agnostic)
=============================================================================

KRONOS2 (MahmoodLab/KRONOS2 on HuggingFace) is the marker-aware successor of
KRONOS. Cells are embedded as in the authors' cell phenotyping (mahmoodlab/CORAL):
a patch_size x patch_size window centred on the cell centroid, zero-padded only at
image borders, every pixel outside the cell set to 0, intensities divided by the
source dtype's scaling factor, normalised by the model's own per-marker statistics
(model.preprocess, with the nuclear stain mapped to DAPI as its nuclear channel)
and embedded with a ViT-B/16 (768-dim CLS token, fp32). --patch-mode bbox instead
crops the cell's bounding box and zero-pads it, without masking. The classifier head is a
logistic regression whose C is tuned per fold with Optuna on a held-out part of
the training cells.

Modes
-----
extract     Embed every labelled cell, one features/<image>.npz per image.
supervised  LogReg + Optuna on the cached features, cross-validated over the folds.
all         extract -> supervised.

Environment: the model's requirements pin torch 2.6 (cu124), which has no kernels
for Blackwell GPUs (sm_120); use torch 2.7 with cu128 there.

Usage
-----
python run_kronos2.py all \\
    --dataset-path <data_root>/IMMUcan \\
    --config       src/methods/configs/kronos2.json \\
    --output-dir   <results>/kronos2/IMMUcan
"""

import time
START = time.perf_counter()  # fold_times.csv: the run is timed from here, package imports included

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from tqdm import tqdm
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from benchmark_io import FoldTimer, fill_from_config, load_config, read_markers, load_benchmark_data, read_segmentation
from utils_foundational_models import check_cache_params, check_fold_coverage


#  FEATURE EXTRACTION

def scaling_factor(dtype):
    """Intensity divisor by source dtype, as in KRONOS2's own image loader (sp_image.py)."""
    if dtype == np.uint8:
        return 255.0
    if np.issubdtype(dtype, np.unsignedinteger):
        return 65535.0
    if np.issubdtype(dtype, np.floating):
        return 400.0
    raise ValueError(f"No KRONOS2 scaling factor for dtype {dtype}")


def cell_patch(img, box, patch_size):
    """Bounding-box crop of one cell, centre-cropped if larger than patch_size, zero-padded to it."""
    crop = img[:, box[0], box[1]]
    ch, cw = crop.shape[1:]
    if ch > patch_size or cw > patch_size:
        half = patch_size // 2
        y0, x0 = max(0, ch // 2 - half), max(0, cw // 2 - half)
        crop = crop[:, y0:y0 + patch_size, x0:x0 + patch_size]
        ch, cw = crop.shape[1:]
    pad_h, pad_w = patch_size - ch, patch_size - cw
    return np.pad(crop, ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)))


def centroid_patch(img, mask, cell_id, centroid, patch_size):
    """Window centred on the cell centroid, zero-padded at image borders, pixels outside the cell set to 0."""
    c, h, w = img.shape
    y0, x0 = centroid[0] - patch_size // 2, centroid[1] - patch_size // 2
    ys, xs = slice(max(y0, 0), min(y0 + patch_size, h)), slice(max(x0, 0), min(x0 + patch_size, w))
    patch = np.zeros((c, patch_size, patch_size), dtype=img.dtype)
    patch[:, ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = img[:, ys, xs] * (mask[ys, xs] == cell_id)
    return patch


def check_markers(model, model_markers):
    """model.preprocess silently uses default statistics for a marker outside the KRONOS2 vocabulary."""
    kronos2 = sys.modules[type(model).__module__]
    index = kronos2.build_marker_index(model._marker_stats)
    unknown = [m for m in model_markers if kronos2.marker_match_key(m) not in index]
    if unknown:
        raise ValueError(f"Not in the KRONOS2 marker vocabulary: {unknown}; "
                         f"add them to marker_name_map or exclude_markers")


def extract_features(args, model, marker_indices, model_markers, meta_df, img_paths, seg_paths):
    feature_dir = Path(args.output_dir) / 'features'
    dapi = 'DAPI' if 'DAPI' in model_markers else None
    check_cache_params(feature_dir, {'model': 'MahmoodLab/KRONOS2', 'patch_size': args.patch_size,
                                     'patch_mode': args.patch_mode, 'markers': model_markers,
                                     'preferred_dapi': dapi})
    check_markers(model, model_markers)

    cells_by_image = meta_df.groupby('image_id')['cell_id'].apply(list)
    print(f"\nExtracting KRONOS2 features | patch_mode={args.patch_mode} | patch_size={args.patch_size} | "
          f"markers={len(model_markers)} | nuclear stain={dapi}")
    for img_id, cell_ids in tqdm(cells_by_image.items(), total=len(cells_by_image), desc="Images"):
        out_path = feature_dir / f'{img_id}.npz'
        if out_path.exists() or img_id not in img_paths or img_id not in seg_paths:
            continue
        raw = tifffile.imread(str(img_paths[img_id]))
        img = raw[marker_indices].astype(np.float32) / scaling_factor(raw.dtype)
        mask = read_segmentation(seg_paths[img_id])
        boxes = ndimage.find_objects(mask)

        kept = [c for c in cell_ids if c - 1 < len(boxes) and boxes[c - 1] is not None]
        centroids = None
        if args.patch_mode == 'centroid':
            centroids = {c: (int(round(y)), int(round(x))) for c, (y, x) in
                         zip(kept, ndimage.center_of_mass(np.ones_like(mask, dtype=np.float32), mask, kept))}
        feats = []
        for start in range(0, len(kept), args.batch_size):
            batch = kept[start:start + args.batch_size]
            patches = np.stack([cell_patch(img, boxes[c - 1], args.patch_size) if centroids is None
                                else centroid_patch(img, mask, c, centroids[c], args.patch_size) for c in batch])
            x = torch.from_numpy(model.preprocess(patches, model_markers, preferred_dapi=dapi)).to(args.device)
            with torch.inference_mode():
                feats.append(model(x, model_markers).float().cpu().numpy())
        np.savez(out_path, features=np.concatenate(feats) if feats else np.zeros((0, 768), np.float32),
                 cell_ids=np.asarray(kept, dtype=np.int64))
    return feature_dir


def load_features(feature_dir, meta_df):
    """Features aligned to meta_df rows that have one; returns (features, meta subset)."""
    keyed = {}
    for path in sorted(Path(feature_dir).glob('*.npz')):
        with np.load(path) as z:
            for cid, f in zip(z['cell_ids'], z['features']):
                keyed[(path.stem, int(cid))] = f
    rows = [i for i, k in enumerate(zip(meta_df['image_id'], meta_df['cell_id'])) if k in keyed]
    meta = meta_df.iloc[rows].reset_index(drop=True)
    feats = np.stack([keyed[k] for k in zip(meta['image_id'], meta['cell_id'])])
    print(f"Loaded features: {feats.shape} from {feature_dir}")
    return feats, meta


#  SUPERVISED: LogReg + Optuna

def run_supervised_optuna(args, folds, feats, meta, start):
    import optuna
    from optuna.samplers import TPESampler
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir = Path(args.spceleval_dir or args.output_dir) / 'KRONOS2_supervised' / 'level3'
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = pd.Series(list(zip(meta['image_id'], meta['cell_id'])))
    check_fold_coverage(folds, keys)
    labels = meta['label'].values
    feat_cols = [f'emb_{i}' for i in range(feats.shape[1])]

    def logreg(C):
        return LogisticRegression(C=C, max_iter=args.max_iter, random_state=args.seed,
                                  class_weight='balanced', solver='lbfgs')

    print(f"\nLogReg + Optuna | n_trials={args.n_trials} | folds={len(folds)}")
    timer, f1s = FoldTimer(start), []
    for fold_idx, fold in enumerate(folds):
        with timer.fold(fold_idx):
            train_idx = np.where(keys.isin(fold['train']).values)[0]
            test_idx = np.where(keys.isin(fold['test']).values)[0]
            rng = np.random.RandomState(args.seed + fold_idx)
            if args.max_cells_per_type is not None:
                train_idx = np.concatenate([
                    rng.choice(idx, min(len(idx), args.max_cells_per_type), replace=False)
                    for idx in (train_idx[labels[train_idx] == c] for c in np.unique(labels[train_idx]))])
            print(f"\n-- Fold {fold_idx} --\n  Train: {len(train_idx):,} cells | "
                  f"{len(set(meta['image_id'].values[train_idx]))} images")

            held_out = rng.choice(len(train_idx), max(1, int(0.2 * len(train_idx))), replace=False)
            tune_tr = np.setdiff1d(np.arange(len(train_idx)), held_out)
            tune_scaler = StandardScaler().fit(feats[train_idx[tune_tr]])
            X_tr, y_tr = tune_scaler.transform(feats[train_idx[tune_tr]]), labels[train_idx[tune_tr]]
            X_val, y_val = tune_scaler.transform(feats[train_idx[held_out]]), labels[train_idx[held_out]]

            def objective(trial):
                clf = logreg(trial.suggest_float('C', args.c_low, args.c_high, log=True)).fit(X_tr, y_tr)
                return f1_score(y_val, clf.predict(X_val), average='macro', zero_division=0)

            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=args.seed))
            study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
            best_C = study.best_params['C']
            print(f"  Best C={best_C:.2e} (Optuna, {args.n_trials} trials)")

            scaler = StandardScaler().fit(feats[train_idx])
            clf = logreg(best_C).fit(scaler.transform(feats[train_idx]), labels[train_idx])

            X_test = scaler.transform(feats[test_idx])
            y_pred, y_proba = clf.predict(X_test), clf.predict_proba(X_test)

            fold_df = pd.DataFrame({
                'image_id': meta['image_id'].values[test_idx],
                'cell_id': meta['cell_id'].values[test_idx],
                'true_phenotype': labels[test_idx],
                'fold': fold_idx,
                'predicted_phenotype': y_pred,
                'confidence': y_proba.max(axis=1),
            })
            fold_df = pd.concat([fold_df, pd.DataFrame(feats[test_idx], columns=feat_cols)], axis=1)
            fold_df.to_csv(out_dir / f'predictions_{fold_idx}.csv', index=False)

        report = classification_report(labels[test_idx], y_pred, output_dict=True, zero_division=0)
        f1s.append(report['macro avg']['f1-score'])
        print(f"  Test: {len(test_idx):,} cells | Accuracy {report['accuracy']:.3f} | "
              f"Macro F1 {f1s[-1]:.3f} | {timer.folds[fold_idx]:.1f}s")

    timer.write(out_dir)

    print(f"\nKRONOS2_supervised: Macro F1 {np.mean(f1s):.3f} +/- {np.std(f1s):.3f} | {out_dir}")


#  MAIN

def load_kronos2_model(device):
    from transformers import AutoModel
    model = AutoModel.from_pretrained('MahmoodLab/KRONOS2', trust_remote_code=True, device=device)
    model.eval()
    print(f"KRONOS2 loaded on {device}")
    return model


def build_parser():
    parser = argparse.ArgumentParser(prog='run_kronos2.py', description='KRONOS2 benchmark pipeline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode, help_text in [('extract', 'Embed every labelled cell'),
                            ('supervised', 'LogReg + Optuna on cached features'),
                            ('all', 'extract -> supervised')]:
        p = sub.add_parser(mode, help=help_text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('--dataset-path', required=True,
                       help='Dataset directory (markers.txt, quantification/, raw_images/, segmentation/)')
        p.add_argument('--config', required=True, help='Path to kronos2.json')
        p.add_argument('--output-dir', required=True)
        p.add_argument('--spceleval-dir', default=None)
        p.add_argument('--folds-dir', default=None, help='k-fold directory, absolute or relative to --dataset-path')
        p.add_argument('--include-validation', action=argparse.BooleanOptionalAction, default=None)
        p.add_argument('--patch-mode', choices=['centroid', 'bbox'], default=None,
                       help='centroid: masked window centred on the cell (authors); '
                            'bbox: zero-padded bounding-box crop, unmasked')
        for name, typ in [('device', str), ('patch-size', int), ('batch-size', int), ('n-trials', int),
                          ('c-low', float), ('c-high', float), ('max-iter', int),
                          ('max-cells-per-type', int), ('seed', int)]:
            p.add_argument(f'--{name}', type=typ, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    data_root = Path(args.dataset_path)
    cfg = load_config(args.config, data_root.name)
    fill_from_config(args, cfg)
    if args.patch_size % 16:
        raise ValueError(f"patch_size must be a multiple of 16 (ViT-B/16), got {args.patch_size}")

    print(f"run_kronos2.py | mode={args.mode} | dataset={data_root.name}")
    markers = read_markers(data_root)
    exclude = set(cfg.get('exclude_markers', []))
    marker_indices = [i for i, m in enumerate(markers) if m not in exclude]
    name_map = cfg.get('marker_name_map', {})
    model_markers = [name_map.get(markers[i], markers[i]) for i in marker_indices]
    print(f"Using {len(marker_indices)} of {len(markers)} markers (excluded: {sorted(exclude)})")

    meta_df, folds, img_paths, seg_paths = load_benchmark_data(
        data_root, data_root / args.folds_dir, cfg.get('exclude_labels', []), args.include_validation)

    feature_dir = Path(args.output_dir) / 'features'
    if args.mode in ('extract', 'all'):
        model = load_kronos2_model(args.device)
        extract_features(args, model, marker_indices, model_markers, meta_df, img_paths, seg_paths)
    if args.mode in ('supervised', 'all'):
        feats, meta = load_features(feature_dir, meta_df)
        run_supervised_optuna(args, folds, feats, meta, START)

    print(f"run_kronos2.py complete | {(time.perf_counter() - START) / 3600:.2f}h")


if __name__ == '__main__':
    main()
