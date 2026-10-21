#!/usr/bin/env python3
"""
run_kronos.py — KRONOS Foundation Model: config-driven benchmark CLI
======================================================================

Dataset-agnostic CLI for KRONOS (adaptive bounding-box patches, see
run_kronos_initial.py for the original prototype this replaces). All
dataset-specific constants (paths, markers, exclusions, name maps, fold
layout) live in src/methods/configs/kronos.json; this script only
implements model loading and feature extraction. The supervised
classifier, Leiden clustering, greedy F1 and output saving are shared
with the other foundation-model methods via utils_foundational_models.py.

Modes
-----
extract     Run KRONOS on every image, cache per-cell embeddings.
supervised  Logistic-regression classifier head on frozen embeddings,
            5-fold CV (the benchmark's main reported number).
leiden      Unsupervised Leiden clustering + greedy majority-label F1.
all         extract -> supervised -> leiden.

Usage
-----
python run_kronos.py all \\
    --dataset     immucan \\
    --config      src/methods/configs/kronos.json \\
    --kronos-dir  /home/juliaoesterle/KRONOS_mine \\
    --marker-meta /home/juliaoesterle/KRONOS_mine/model_assets/marker_metadata.csv \\
    --output-dir  /home/juliaoesterle/results/kronos/immucan \\
    --device      cuda:0

python run_kronos.py all \\
    --dataset     cHL_2_MIBI \\
    --config      src/methods/configs/kronos.json \\
    --kronos-dir  /home/juliaoesterle/KRONOS_mine \\
    --marker-meta /home/juliaoesterle/KRONOS_mine/model_assets/marker_metadata.csv \\
    --output-dir  /home/juliaoesterle/results/kronos/chl \\
    --device      cuda:0
"""

import os
import sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)

# CUDA env vars — must be set before torch import
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.3')
os.environ.setdefault('LD_LIBRARY_PATH',
                      '/usr/local/cuda-12.3/lib64:' +
                      os.environ.get('LD_LIBRARY_PATH', ''))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from tqdm import tqdm

from utils.utils_foundational_models import (
    load_label_map, load_folds, get_label, save_embeddings,
    load_embeddings, rebuild_img_feature_store, run_supervised, run_leiden,
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


#  CONFIG LOADING

def _rebase_local_paths(obj, prefix="/home/juliaoesterle", env_var="SPCELLEVAL_ROOT"):
    """kronos.json hardcodes absolute paths rooted at beauty (/home/juliaoesterle).
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
    if dataset not in cfg["datasets"]:
        raise ValueError(f"Dataset '{dataset}' not in {config_path}. "
                         f"Available: {list(cfg['datasets'])}")
    merged = dict(cfg.get("defaults", {}))
    merged.update(cfg["datasets"][dataset])
    merged["dataset"] = dataset
    return _rebase_local_paths(merged)


#  Marker helpers

def get_clean_markers(biomarkers, exclude):
    exclude_set = set(exclude)
    clean_idx   = [i for i, m in enumerate(biomarkers) if m not in exclude_set]
    clean_names = [m for m in biomarkers if m not in exclude_set]
    print(f"Using {len(clean_names)} markers (excluded: {sorted(exclude_set)})")
    return clean_idx, clean_names


def build_normalisation_params(clean_names, marker_meta_path, name_mappings, no_equivalent):
    """
    Build per-marker mean and std arrays for KRONOS normalisation.

    KRONOS was trained on absolute marker intensities normalised by
    per-marker mean/std from marker_metadata.csv. For markers with no
    KRONOS equivalent, mean=0 and std=1 is used (identity normalisation).
    """
    df = pd.read_csv(marker_meta_path)
    lookup = {
        row['marker_name'].upper(): (row['marker_id'], row['marker_mean'], row['marker_std'])
        for _, row in df.iterrows()
    }
    no_equivalent_set = set(no_equivalent)

    means, stds, marker_ids = [], [], []
    print("\nKRONOS marker normalisation params:")

    for m in clean_names:
        kronos_name = name_mappings.get(m, m)
        key         = kronos_name.upper()

        if key in lookup:
            mid, mean, std = lookup[key]
            means.append(mean)
            stds.append(std if std > 0 else 1.0)
            marker_ids.append(int(mid))
            status = f"FOUND   -> {kronos_name:20s} id={mid} mean={mean:.3f} std={std:.3f}"
        elif m in no_equivalent_set:
            means.append(0.0)
            stds.append(1.0)
            marker_ids.append(0)
            status = "FALLBACK -> mean=0 std=1 (no KRONOS equivalent)"
        else:
            means.append(0.0)
            stds.append(1.0)
            marker_ids.append(0)
            status = "MISSING  -> mean=0 std=1 (not in KRONOS training data)"

        print(f"  {m:25s} {status}")

    means = np.array(means, dtype=np.float32)
    stds  = np.array(stds,  dtype=np.float32)

    n_found   = sum(1 for mid in marker_ids if mid > 0)
    n_missing = sum(1 for mid in marker_ids if mid == 0)
    print(f"\nNormalisation: {n_found} markers with KRONOS stats | "
          f"{n_missing} markers using identity (mean=0, std=1)")

    return means, stds, marker_ids


#  KRONOS model loading

def load_kronos_model(kronos_dir, device):
    """
    Load KRONOS model from HuggingFace checkpoint.
    kronos package is imported here because kronos_dir must be on sys.path first.
    Requires HuggingFace authentication: huggingface-cli auth login
    Model: MahmoodLab/kronos (gated repo — access request required)
    """
    sys.path.insert(0, str(kronos_dir))
    from kronos import create_model_from_pretrained

    cache_dir = Path(kronos_dir) / 'model_assets'
    cache_dir.mkdir(parents=True, exist_ok=True)

    model, precision, embedding_dim = create_model_from_pretrained(
        checkpoint_path='hf_hub:MahmoodLab/kronos',
        cache_dir=str(cache_dir),
    )
    model = model.to(device)
    model.eval()
    print(f"KRONOS loaded on: {device}")
    print(f"  precision:     {precision}")
    print(f"  embedding_dim: {embedding_dim}")
    return model, embedding_dim, precision


#  KRONOS-specific patch extraction

def get_bbox_patch(img, ys, xs, patch_size=64):
    """
    Adaptive bounding box: crop exact cell bbox from segmentation mask
    pixels -> symmetrically zero-pad to patch_size x patch_size.
    Mean IMMUcan cell: 9.5x10.6px bbox — well within 64x64.
    """
    y_min, y_max   = int(ys.min()), int(ys.max())
    x_min, x_max   = int(xs.min()), int(xs.max())
    crop           = img[y_min:y_max+1, x_min:x_max+1, :]
    crop_h, crop_w = crop.shape[:2]

    if crop_h > patch_size or crop_w > patch_size:
        cy, cx = crop_h // 2, crop_w // 2
        half   = patch_size // 2
        crop   = crop[max(0, cy-half):cy+half, max(0, cx-half):cx+half, :]
        crop_h, crop_w = crop.shape[:2]

    pad_h = patch_size - crop_h
    pad_w = patch_size - crop_w
    patch = np.pad(crop, ((pad_h // 2, pad_h - pad_h // 2),
                          (pad_w // 2, pad_w - pad_w // 2),
                          (0, 0)))
    return patch


def extract_features_batch(patches, model, means, stds, device):
    """
    Batched KRONOS forward pass on patch_size x patch_size patches.
    KRONOS outputs three embeddings — we use patch_embeddings
    (single summary vector per patch, equivalent to Eva's CLS token).
    """
    batch = np.stack(patches)                            # (N, H, W, C)
    batch = torch.from_numpy(batch).to(device)
    batch = batch.permute(0, 3, 1, 2).float()            # (N, C, H, W)

    mean_t = torch.tensor(means, dtype=torch.float32).to(device)
    std_t  = torch.tensor(stds,  dtype=torch.float32).to(device)
    batch  = (batch - mean_t[None, :, None, None]) / std_t[None, :, None, None]

    with torch.no_grad():
        patch_embeddings, marker_embeddings, token_embeddings = model(batch)

    return patch_embeddings.cpu().numpy()                # (N, embedding_dim)


#  Feature extraction

def extract_features(args, cfg, model, all_images, clean_indices, clean_names,
                     means, stds, label_map):
    """
    Extract KRONOS cell embeddings for all images using adaptive bounding box.
    Results are cached per image for crash-safe resumption.
    """
    data_root = Path(cfg["data_root"])
    cache_dir = Path(args.output_dir) / 'embeddings' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    NPZ_DIR    = data_root / cfg["image_dir"]
    MASK_DIR   = data_root / cfg["segmentation_dir"]
    LABELS_DIR = data_root / cfg["label_dir"]

    img_feature_store = {}

    print(f"\nExtracting KRONOS features | patch_size={args.patch_size} | "
          f"batch_size={args.batch_size} | device={args.device}")
    print("=" * 60)

    for img_name in tqdm(all_images, desc="KRONOS (bbox)"):
        cache_feat = cache_dir / f"{img_name}_feat.npy"
        cache_meta = cache_dir / f"{img_name}_meta.csv"

        if cache_feat.exists() and cache_meta.exists():
            feats = np.load(cache_feat)
            meta  = pd.read_csv(cache_meta)
            img_feature_store[img_name] = (
                feats, meta['label'].tolist(), meta['cell_id'].tolist()
            )
            continue

        npz_file   = NPZ_DIR    / f"{img_name}.npz"
        mask_file  = MASK_DIR   / f"{img_name}.tiff"
        label_file = LABELS_DIR / f"{img_name}.txt"
        if not all(f.exists() for f in [npz_file, mask_file, label_file]):
            print(f"  Skipping {img_name} — missing files")
            continue

        img = np.load(npz_file)['data'].astype(np.float32)
        img = img[clean_indices]
        img = img / (img.max() + 1e-8)     # scale to [0,1] before KRONOS normalisation
        img = img.transpose(1, 2, 0)       # (H, W, C)

        mask = tifffile.imread(mask_file)
        with open(label_file) as f:
            cell_labels = [int(line.strip()) for line in f.readlines()]

        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids > 0]

        patches_buf, labels_buf, cell_ids_buf = [], [], []
        for cell_id in cell_ids:
            label  = get_label(int(cell_id), cell_labels, label_map)
            ys, xs = np.where(mask == cell_id)
            patch  = get_bbox_patch(img, ys, xs, args.patch_size)
            patches_buf.append(patch)
            labels_buf.append(label)
            cell_ids_buf.append(int(cell_id))

        all_feats = []
        for i in range(0, len(patches_buf), args.batch_size):
            batch_feats = extract_features_batch(
                patches_buf[i:i + args.batch_size],
                model, means, stds, args.device
            )
            all_feats.extend(batch_feats)
            torch.cuda.empty_cache()

        feats_arr = np.array(all_feats)

        np.save(cache_feat, feats_arr)
        pd.DataFrame({
            'image_id': [img_name] * len(cell_ids_buf),
            'cell_id':  cell_ids_buf,
            'label':    labels_buf,
        }).to_csv(cache_meta, index=False)

        img_feature_store[img_name] = (feats_arr, labels_buf, cell_ids_buf)

    all_feats_list, all_labels_list, all_ids_list, all_imgs_list = [], [], [], []
    for img_name, (feats, labels, cell_ids) in img_feature_store.items():
        all_feats_list.extend(feats)
        all_labels_list.extend(labels)
        all_ids_list.extend(cell_ids)
        all_imgs_list.extend([img_name] * len(cell_ids))

    all_feats_arr = np.array(all_feats_list)
    metadata_all  = pd.DataFrame({
        'image_id': all_imgs_list,
        'cell_id':  all_ids_list,
        'label':    all_labels_list,
    })

    save_embeddings(args.output_dir, all_feats_arr, metadata_all)

    print(f"\nExtraction complete: {len(all_feats_arr):,} cells | "
          f"embedding_dim={all_feats_arr.shape[1]}")
    print(pd.Series(all_labels_list).value_counts().to_string())

    return img_feature_store, all_feats_arr, metadata_all


#  cHL data loading + feature extraction

def load_chl_data(cfg, data_root):
    """
    Returns meta_df[image_id, cell_id, label], folds (cell-level: sets of
    (image_id, cell_id) tuples read from the per-fold fold_{i}_train.csv/
    fold_{i}_test.csv files, not fold_indices.json -- see kronos.json's
    _comment_folds_dir), img_paths, seg_paths.
    """
    raw_df = pd.read_csv(data_root / cfg["quant_csv"])
    raw_df['sample_id'] = raw_df['sample_id'].astype(str).str.replace('.csv', '', regex=False)

    def load_keys(path):
        df = pd.read_csv(path, usecols=['sample_id', 'cell_id'])
        df['sample_id'] = df['sample_id'].astype(str).str.replace('.csv', '', regex=False)
        return set(zip(df['sample_id'], df['cell_id']))

    folds_dir = data_root / cfg["folds_dir"]
    folds  = []
    fold_i = 1
    while (folds_dir / f'fold_{fold_i}_train.csv').exists():
        folds.append({
            'train': load_keys(folds_dir / f'fold_{fold_i}_train.csv'),
            'test':  load_keys(folds_dir / f'fold_{fold_i}_test.csv'),
        })
        fold_i += 1
    print(f"Loaded {len(folds)} folds (cell-level)")

    exclude_labels = set(cfg.get("exclude_labels", []))
    meta_df = raw_df[~raw_df['cell_type'].isin(exclude_labels)].copy()
    meta_df = meta_df.rename(columns={'sample_id': 'image_id', 'cell_type': 'label'})
    meta_df = meta_df[['image_id', 'cell_id', 'label']]
    print(f"Loaded cHL metadata: {len(meta_df):,} cells "
          f"(dropped {len(raw_df) - len(meta_df):,} excluded-label cells)")

    img_dir   = data_root / cfg["image_dir"]
    img_paths = {}
    for tif in sorted(img_dir.glob('*_stacked.ome.tif')):
        if tif.name.startswith('.'):
            continue
        img_paths[tif.name.replace('_stacked.ome.tif', '')] = tif

    seg_dir   = data_root / cfg["segmentation_dir"]
    seg_paths = {}
    for img_id in img_paths:
        seg_files = [p for p in (seg_dir / img_id).rglob('segmentationMap.tif')
                     if not p.name.startswith('.')]
        if seg_files:
            seg_paths[img_id] = seg_files[0]
    print(f"Found {len(img_paths)} images, {len(seg_paths)} segmentation maps")

    return meta_df, folds, img_paths, seg_paths


def extract_features_chl(args, model, all_images, clean_indices, clean_names,
                         means, stds, meta_df, img_paths, seg_paths):
    """
    cHL equivalent of extract_features(). Same bbox-crop + normalisation
    logic; only image/mask reading and ground-truth lookup differ (ome_tiff
    images, nested per-image segmentation maps, labels from meta_df instead
    of cells2labels/*.txt + label_map).
    """
    cache_dir = Path(args.output_dir) / 'embeddings' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    img_feature_store = {}

    print(f"\nExtracting KRONOS features (cHL) | patch_size={args.patch_size} | "
          f"batch_size={args.batch_size} | device={args.device}")
    print("=" * 60)

    for img_id in tqdm(all_images, desc="KRONOS (bbox, cHL)"):
        cache_feat = cache_dir / f"{img_id}_feat.npy"
        cache_meta = cache_dir / f"{img_id}_meta.csv"

        if cache_feat.exists() and cache_meta.exists():
            feats = np.load(cache_feat)
            meta  = pd.read_csv(cache_meta)
            img_feature_store[img_id] = (
                feats, meta['label'].tolist(), meta['cell_id'].tolist()
            )
            continue

        if img_id not in img_paths or img_id not in seg_paths:
            print(f"  Skipping {img_id} — missing image or segmentation")
            continue

        meta_img = meta_df[meta_df['image_id'] == img_id]
        if meta_img.empty:
            print(f"  Skipping {img_id} — no metadata rows")
            continue
        label_lookup = dict(zip(meta_img['cell_id'], meta_img['label']))

        img = tifffile.imread(img_paths[img_id]).astype(np.float32)
        img = img[clean_indices]
        img = img / (img.max() + 1e-8)     # scale to [0,1] before KRONOS normalisation
        img = img.transpose(1, 2, 0)       # (H, W, C)

        mask = tifffile.imread(seg_paths[img_id]).astype(np.int32)

        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids > 0]

        patches_buf, labels_buf, cell_ids_buf = [], [], []
        for cell_id in cell_ids:
            cell_id = int(cell_id)
            if cell_id not in label_lookup:
                continue
            ys, xs = np.where(mask == cell_id)
            patch  = get_bbox_patch(img, ys, xs, args.patch_size)
            patches_buf.append(patch)
            labels_buf.append(label_lookup[cell_id])
            cell_ids_buf.append(cell_id)

        if not patches_buf:
            print(f"  Skipping {img_id} — no ground-truth cells extracted")
            continue

        all_feats = []
        for i in range(0, len(patches_buf), args.batch_size):
            batch_feats = extract_features_batch(
                patches_buf[i:i + args.batch_size],
                model, means, stds, args.device
            )
            all_feats.extend(batch_feats)
            torch.cuda.empty_cache()

        feats_arr = np.array(all_feats)

        np.save(cache_feat, feats_arr)
        pd.DataFrame({
            'image_id': [img_id] * len(cell_ids_buf),
            'cell_id':  cell_ids_buf,
            'label':    labels_buf,
        }).to_csv(cache_meta, index=False)

        img_feature_store[img_id] = (feats_arr, labels_buf, cell_ids_buf)

    all_feats_list, all_labels_list, all_ids_list, all_imgs_list = [], [], [], []
    for img_id, (feats, labels, cell_ids) in img_feature_store.items():
        all_feats_list.extend(feats)
        all_labels_list.extend(labels)
        all_ids_list.extend(cell_ids)
        all_imgs_list.extend([img_id] * len(cell_ids))

    all_feats_arr = np.array(all_feats_list)
    metadata_all  = pd.DataFrame({
        'image_id': all_imgs_list,
        'cell_id':  all_ids_list,
        'label':    all_labels_list,
    })

    save_embeddings(args.output_dir, all_feats_arr, metadata_all)

    print(f"\nExtraction complete: {len(all_feats_arr):,} cells | "
          f"embedding_dim={all_feats_arr.shape[1]}")
    print(pd.Series(all_labels_list).value_counts().to_string())

    return img_feature_store, all_feats_arr, metadata_all


#  Argument parser

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_kronos.py',
        description='KRONOS Foundation Model — config-driven pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ['extract', 'supervised', 'leiden', 'all']:
        p = sub.add_parser(
            mode,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            help={
                'extract':    'Extract KRONOS embeddings only (reusable downstream)',
                'supervised': 'Logistic-regression classifier head + 5-fold CV',
                'leiden':     'Leiden clustering + greedy F1',
                'all':        'extract -> supervised -> leiden in sequence',
            }[mode],
        )
        p.add_argument('--dataset', choices=['immucan', 'cHL_2_MIBI'], required=True)
        p.add_argument('--config',  required=True, help='Path to kronos.json')
        p.add_argument('--kronos-dir', required=True,
                       help='KRONOS repo directory (contains kronos/ package)')
        p.add_argument('--marker-meta', required=True,
                       help='Path to marker_metadata.csv from MahmoodLab/kronos')
        p.add_argument('--output-dir', required=True)
        p.add_argument('--spceleval-dir', default=None)
        # optional overrides (default None -> config)
        p.add_argument('--device',     default=None)
        p.add_argument('--patch-size', type=int, default=None)
        p.add_argument('--batch-size', type=int, default=None)
        p.add_argument('--n-folds',    type=int, default=None)
        p.add_argument('--logreg-c',   type=float, default=None)
        p.add_argument('--n-jobs',     type=int, default=None)
        p.add_argument('--leiden-resolution',  type=float, default=None)
        p.add_argument('--leiden-n-neighbors', type=int,   default=None)
        p.add_argument('--leiden-subsample',   type=int,   default=None)
    return parser


#  Main

def main():
    parser = build_parser()
    args   = parser.parse_args()
    total_start = time.time()

    cfg = load_config(args.config, args.dataset)

    def fill(attr, key, default=None):
        if getattr(args, attr) is None:
            setattr(args, attr, cfg.get(key, default))
    fill('device', 'device', 'cuda:0')
    fill('patch_size', 'patch_size', 64)
    fill('batch_size', 'batch_size', 16)
    fill('n_folds', 'n_folds', 5)
    fill('logreg_c', 'logreg_c', 1.0)
    fill('n_jobs', 'n_jobs', -1)
    fill('leiden_resolution', 'leiden_resolution', 2.0)
    fill('leiden_n_neighbors', 'leiden_n_neighbors', 15)
    fill('leiden_subsample', 'leiden_subsample', 50000)

    print(f"\n{'='*60}")
    print(f"run_kronos.py | mode={args.mode} | dataset={args.dataset}")
    print(f"{'='*60}")

    data_root     = Path(cfg["data_root"])
    fold_type     = cfg["fold_type"]
    clean_indices, clean_names = get_clean_markers(cfg["biomarkers"], cfg["exclude_markers"])
    means, stds, marker_ids = build_normalisation_params(
        clean_names, args.marker_meta, cfg.get("name_mappings", {}), cfg.get("no_equivalent", [])
    )

    if args.dataset == 'immucan':
        label_map          = load_label_map(str(data_root))
        folds, all_images  = load_folds(str(data_root), args.n_folds)
    else:
        meta_df, folds, img_paths, seg_paths = load_chl_data(cfg, data_root)
        all_images = sorted(img_paths.keys())

    supervised_name = 'KRONOS_supervised'
    leiden_name     = 'KRONOS_leiden'

    all_feats_arr, metadata_all = load_embeddings(args.output_dir)
    if all_feats_arr is not None and args.mode in ('supervised', 'leiden'):
        img_feature_store = rebuild_img_feature_store(args.output_dir, all_images)
    else:
        model, embedding_dim, precision = load_kronos_model(args.kronos_dir, args.device)
        if args.dataset == 'immucan':
            img_feature_store, all_feats_arr, metadata_all = extract_features(
                args, cfg, model, all_images, clean_indices, clean_names, means, stds, label_map
            )
        else:
            img_feature_store, all_feats_arr, metadata_all = extract_features_chl(
                args, model, all_images, clean_indices, clean_names, means, stds,
                meta_df, img_paths, seg_paths
            )

    if args.mode in ('supervised', 'all'):
        run_supervised(args, folds, img_feature_store, all_feats_arr, metadata_all,
                       method_name=supervised_name, fold_type=fold_type)

    if args.mode in ('leiden', 'all'):
        run_leiden(args, folds, img_feature_store, all_feats_arr, metadata_all,
                   method_name=leiden_name, fold_type=fold_type)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"run_kronos.py complete | total: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
