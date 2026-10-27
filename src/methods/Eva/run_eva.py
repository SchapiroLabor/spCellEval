#!/usr/bin/env python3
"""
run_eva.py — Eva Foundation Model: config-driven unified benchmark CLI
======================================================================

Dataset-agnostic CLI for Eva v3 (the authors' spatial-token-map / full-tile
approach -- see eva/project/notebooks/Eva-pipeline-v3.ipynb). All
dataset-specific constants (paths, markers, exclusions, name maps, fold
layout) live in src/methods/configs/eva.json; this script only implements
model loading and feature extraction. The supervised classifier, Leiden
clustering, greedy F1 and output saving are shared with the other
foundation-model methods via utils_foundational_models.py.

Modes
-----
extract     Run Eva on every image, cache per-cell embeddings.
supervised  Logistic-regression classifier head on frozen embeddings,
            5-fold CV (the benchmark's main reported number).
leiden      Unsupervised Leiden clustering + greedy majority-label F1.
all         extract -> supervised -> leiden.

Usage
-----
python run_eva.py all \\
    --dataset-path /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan \\
    --config       src/methods/configs/eva.json \\
    --eva-dir      /home/juliaoesterle/eva/project \\
    --output-dir   /home/juliaoesterle/results/eva/IMMUcan \\
    --device       cuda:1

python run_eva.py extract \\
    --dataset-path /home/juliaoesterle/data/phenotyping_benchmark/cHL_2_MIBI \\
    --config       src/methods/configs/eva.json \\
    --eva-dir      /home/juliaoesterle/eva/project \\
    --output-dir   /home/juliaoesterle/results/eva/cHL_2_MIBI \\
    --device       cuda:1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch


#  CONFIG LOADING

def _rebase_local_paths(obj, prefix="/home/juliaoesterle", env_var="SPCELLEVAL_ROOT"):
    """eva.json hardcodes absolute paths rooted at beauty (/home/juliaoesterle).
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


#  EVA MODEL LOADING

def load_eva_model(eva_dir, device):
    eva_dir = Path(eva_dir)
    sys.path.insert(0, str(eva_dir))
    os.chdir(str(eva_dir))

    from omegaconf import OmegaConf
    from Eva.utils import load_from_hf

    conf  = OmegaConf.load(eva_dir / 'config.yaml')
    model = load_from_hf(repo_id='yandrewl/Eva', conf=conf, device=device)
    model.eval()
    print(f"Eva loaded on {next(model.parameters()).device}")
    return model


#  MARKER HELPERS

def get_markers(cfg, data_root):
    """Read channel order from markers.txt; exclude + map names from config."""
    with open(data_root / 'markers.txt') as f:
        all_markers = [line.strip() for line in f if line.strip()]

    exclude_set   = set(cfg.get("exclude_markers", []))
    name_map      = cfg.get("marker_name_map", {})
    clean_indices = [i for i, m in enumerate(all_markers) if m not in exclude_set]
    biomarkers    = [name_map.get(all_markers[i], all_markers[i]) for i in clean_indices]

    print(f"Markers: {len(all_markers)} total, "
          f"{len(clean_indices)} kept (excluded: {sorted(exclude_set)})")
    return clean_indices, biomarkers


#  DATA LOADING

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
    (image_id, cell_id) tuples), img_paths, seg_paths.

    Ground truth comes directly from the quantification table's own
    'cell_type' column - not 'cell_labels' (a different column some of the
    original data releases carry, which is the authors' own annotation, not
    ours).

    """
    quant_csv = data_root / 'quantification' / 'processed' / f'{data_root.name}_quantification.csv'
    raw_df = pd.read_csv(quant_csv, usecols=['sample_id', 'cell_id', 'cell_type'])
    raw_df['sample_id'] = raw_df['sample_id'].astype(str).str.replace('.csv', '', regex=False)

    def load_keys(path):
        df = pd.read_csv(path, usecols=['sample_id', 'cell_id'])
        df['sample_id'] = df['sample_id'].astype(str).str.replace('.csv', '', regex=False)
        return set(zip(df['sample_id'], df['cell_id']))

    folds   = []
    fold_i  = 1
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


#  IMAGE / SEGMENTATION READING

def read_image(img_path, clean_indices):
    return tifffile.imread(str(img_path)).astype(np.float32)[clean_indices]


def read_segmentation(seg_path):
    return tifffile.imread(str(seg_path)).astype(np.int32)


#  FEATURE EXTRACTION  (v3: spatial token map, see docstring of extract_token_map)

def extract_token_map(img, model, biomarkers, device, tile_size, stride):
    """
    Overlapping tile_size x tile_size tiles cover the whole image -> one
    HxWx768 spatial token map, tokens averaged where tiles overlap.
    A cell's feature is the mean token vector over its own segmentation-mask
    pixels (computed by the caller). img: (H, W, C) float32.
    """
    H, W, C  = img.shape
    feat_dim = 768

    feat_accum  = np.zeros((H, W, feat_dim), dtype=np.float32)
    count_accum = np.zeros((H, W), dtype=np.float32)

    y_starts = list(range(0, max(1, H - tile_size + 1), stride))
    x_starts = list(range(0, max(1, W - tile_size + 1), stride))
    if y_starts[-1] + tile_size < H: y_starts.append(H - tile_size)
    if x_starts[-1] + tile_size < W: x_starts.append(W - tile_size)

    for y0 in y_starts:
        for x0 in x_starts:
            tile = img[y0:y0 + tile_size, x0:x0 + tile_size, :]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                pad_tile = np.zeros((tile_size, tile_size, C), dtype=np.float32)
                pad_tile[:tile.shape[0], :tile.shape[1], :] = tile
                tile = pad_tile

            t = torch.from_numpy(tile).unsqueeze(0).to(device)
            with torch.no_grad():
                token_out, _ = model.model.forward_encoder(
                    t.permute(0, 3, 1, 2).contiguous(), [biomarkers]
                )

            # Drop CLS, mean over marker channels -> (n_patches, 768) -> (grid, grid, 768)
            spatial    = token_out[0, :, 1:, :].mean(dim=0).cpu().numpy()
            grid_size  = int(np.sqrt(spatial.shape[0]))
            token_grid = spatial.reshape(grid_size, grid_size, feat_dim)
            token_size = tile_size // grid_size

            for gy in range(grid_size):
                for gx in range(grid_size):
                    py0 = y0 + gy * token_size
                    px0 = x0 + gx * token_size
                    py1 = min(py0 + token_size, H)
                    px1 = min(px0 + token_size, W)
                    feat_accum[py0:py1, px0:px1, :] += token_grid[gy, gx, :]
                    count_accum[py0:py1, px0:px1]   += 1.0

    count_accum = np.maximum(count_accum, 1.0)
    return feat_accum / count_accum[:, :, np.newaxis]


def extract_features(args, model, all_images, clean_indices, biomarkers,
                     meta_df, img_paths, seg_paths):
    """
    Runs extract_token_map per image, then pools each cell's feature from
    its own mask pixels. Per-image results are cached to disk in the format
    utils_foundational_models.rebuild_img_feature_store()/load_embeddings()
    already expect, so a crashed/resumed run or a later supervised/leiden-only
    invocation reuses them without re-running Eva.
    """
    from utils_foundational_models import save_embeddings

    cache_dir = Path(args.output_dir) / 'embeddings' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    img_feature_store = {}
    all_feats_list, meta_records = [], []
    t0_total = time.time()

    for img_id in sorted(all_images):
        cache_feat = cache_dir / f"{img_id}_feat.npy"
        cache_meta = cache_dir / f"{img_id}_meta.csv"
        if cache_feat.exists() and cache_meta.exists():
            feats = np.load(cache_feat)
            meta  = pd.read_csv(cache_meta, dtype={'image_id': str})
            img_feature_store[img_id] = (feats, meta['label'].tolist(), meta['cell_id'].tolist())
            all_feats_list.append(feats)
            meta_records.append(meta)
            continue

        if img_id not in img_paths or img_id not in seg_paths:
            print(f"  [skip] {img_id}: missing image or segmentation")
            continue

        t0 = time.time()
        img_chw = read_image(img_paths[img_id], clean_indices)           # (C, H, W)
        img_hwc = np.ascontiguousarray(img_chw.transpose(1, 2, 0))       # (H, W, C)
        seg     = read_segmentation(seg_paths[img_id])

        meta_img = meta_df[meta_df['image_id'] == img_id]
        if meta_img.empty:
            print(f"  [skip] {img_id}: no metadata rows")
            continue

        token_map = extract_token_map(img_hwc, model, biomarkers, args.device,
                                      tile_size=args.tile_size, stride=args.tile_stride)

        feats, labels, cell_ids = [], [], []
        for _, row in meta_img.iterrows():
            cell_pixels = seg == row['cell_id']
            if not cell_pixels.any():
                continue
            feats.append(token_map[cell_pixels].mean(axis=0))
            labels.append(row['label'])
            cell_ids.append(row['cell_id'])

        if not feats:
            print(f"  [skip] {img_id}: no cells extracted")
            continue

        feats_arr = np.array(feats, dtype=np.float32)
        np.save(cache_feat, feats_arr)
        meta_out = pd.DataFrame({'image_id': img_id, 'cell_id': cell_ids, 'label': labels})
        meta_out.to_csv(cache_meta, index=False)

        img_feature_store[img_id] = (feats_arr, labels, cell_ids)
        all_feats_list.append(feats_arr)
        meta_records.append(meta_out)
        print(f"  {img_id}: {len(feats_arr)} cells | {time.time() - t0:.1f}s")

    all_feats_arr = np.concatenate(all_feats_list, axis=0)
    metadata_all  = pd.concat(meta_records, ignore_index=True)
    print(f"\nExtracted {len(all_feats_arr):,} cells total in {time.time() - t0_total:.1f}s")

    save_embeddings(args.output_dir, all_feats_arr, metadata_all)
    return img_feature_store, all_feats_arr, metadata_all


#  ARGUMENT PARSER

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_eva.py',
        description='Eva v3 config-driven pipeline (dataset-agnostic)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ['extract', 'supervised', 'leiden', 'all']:
        p = sub.add_parser(mode, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('--dataset-path', required=True,
                       help='Dataset directory (markers.txt, quantification/, raw_images/, segmentation/)')
        p.add_argument('--folds-dir', default=None,
                       help='k-fold directory, absolute or relative to --dataset-path')
        p.add_argument('--include-validation', action=argparse.BooleanOptionalAction, default=None,
                       help='Add fold_{i}_validation.csv cells to the training set')
        p.add_argument('--config',  required=True, help='Path to eva.json')
        p.add_argument('--eva-dir', required=True, help='Eva project root')
        p.add_argument('--output-dir', required=True)
        p.add_argument('--spceleval-dir', default=None)
        # optional overrides (default None -> config)
        p.add_argument('--device',       default=None)
        p.add_argument('--tile-size',    type=int, default=None)
        p.add_argument('--tile-stride',  type=int, default=None)
        p.add_argument('--n-folds',      type=int, default=None)
        p.add_argument('--logreg-c',     type=float, default=None)
        p.add_argument('--n-jobs',       type=int, default=-1)
        p.add_argument('--leiden-resolution',  type=float, default=None)
        p.add_argument('--leiden-n-neighbors', type=int,   default=None)
        p.add_argument('--leiden-subsample',   type=int,   default=None)
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
    fill('device', 'device', 'cuda:1')
    fill('tile_size', 'tile_size', 224)
    fill('tile_stride', 'tile_stride', 112)
    fill('n_folds', 'n_folds', 5)
    fill('logreg_c', 'logreg_c', 1.0)
    fill('leiden_resolution', 'leiden_resolution', 2.0)
    fill('leiden_n_neighbors', 'leiden_n_neighbors', 15)
    fill('leiden_subsample', 'leiden_subsample', 50000)
    fill('folds_dir', 'folds_dir', 'quantification/processed/kfolds_StratifiedGroupKFold_level3')
    fill('include_validation', 'include_validation', False)

    print(f"\n{'='*60}")
    print(f"run_eva.py | mode={args.mode} | dataset={args.dataset}")
    print(f"{'='*60}")

    utils_dir = (Path(args.spceleval_dir) / 'src' / 'methods' / 'utils' if args.spceleval_dir
                else Path(__file__).resolve().parent.parent / 'utils')
    sys.path.insert(0, str(utils_dir))
    from utils_foundational_models import (
        load_embeddings, rebuild_img_feature_store, run_supervised, run_leiden,
    )

    meta_df, folds, img_paths, seg_paths = load_data(
        cfg, data_root, data_root / args.folds_dir, args.include_validation)
    clean_indices, biomarkers = get_markers(cfg, data_root)

    all_images = sorted(img_paths.keys())

    supervised_name = f"EVA_supervised_{args.dataset}"
    leiden_name     = f"EVA_leiden_{args.dataset}"

    all_feats_arr, metadata_all = load_embeddings(args.output_dir)
    if all_feats_arr is not None and args.mode in ('supervised', 'leiden'):
        img_feature_store = rebuild_img_feature_store(args.output_dir, all_images)
    else:
        model = load_eva_model(args.eva_dir, args.device)
        img_feature_store, all_feats_arr, metadata_all = extract_features(
            args, model, all_images, clean_indices, biomarkers,
            meta_df, img_paths, seg_paths,
        )

    if args.mode in ('supervised', 'all'):
        run_supervised(args, folds, img_feature_store, all_feats_arr, metadata_all,
                       method_name=supervised_name, fold_type='cell')
    if args.mode in ('leiden', 'all'):
        run_leiden(args, folds, img_feature_store, all_feats_arr, metadata_all,
                   method_name=leiden_name, fold_type='cell')

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"run_eva.py complete | dataset={args.dataset} | total: {total_time:.1f}s")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
