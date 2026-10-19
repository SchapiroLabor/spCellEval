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
    --dataset    immucan \\
    --config     src/methods/configs/eva.json \\
    --eva-dir    /home/juliaoesterle/eva/project \\
    --output-dir /home/juliaoesterle/results/eva/immucan \\
    --device     cuda:1

python run_eva.py extract \\
    --dataset    cHL_2_MIBI \\
    --config     src/methods/configs/eva.json \\
    --eva-dir    /home/juliaoesterle/eva/project \\
    --output-dir /home/juliaoesterle/results/eva/chl \\
    --device     cuda:1
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
    if dataset not in cfg["datasets"]:
        raise ValueError(f"Dataset '{dataset}' not in {config_path}. "
                         f"Available: {list(cfg['datasets'])}")
    merged = dict(cfg.get("defaults", {}))
    merged.update(cfg["datasets"][dataset])
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

def get_immucan_markers(cfg, data_root):
    """Read IMMUcan markers from channels.txt; exclude + map names from config."""
    channels_path = data_root / cfg["channels_txt"]
    with open(channels_path) as f:
        all_markers = [line.strip() for line in f if line.strip()]

    exclude_set   = set(cfg["exclude_markers"])
    name_map      = cfg.get("marker_name_map", {})
    clean_indices = [i for i, m in enumerate(all_markers) if m not in exclude_set]
    biomarkers    = [name_map.get(all_markers[i], all_markers[i]) for i in clean_indices]

    print(f"IMMUcan markers: {len(all_markers)} total, "
          f"{len(clean_indices)} kept (excluded: {sorted(exclude_set)})")
    return clean_indices, biomarkers


def get_chl_markers(cfg):
    """Build channel indices + Eva-compatible names for cHL from config."""
    channels    = cfg["channels"]
    exclude_set = set(cfg["exclude_markers"])
    name_map    = cfg.get("marker_name_map", {})

    clean_indices = [i for i, c in enumerate(channels) if c not in exclude_set]
    biomarkers    = [name_map.get(c, c) for c in channels if c not in exclude_set]

    print(f"cHL markers: {len(channels)} total, "
          f"{len(clean_indices)} kept (excluded: {sorted(exclude_set)})")
    return clean_indices, biomarkers


#  DATA LOADING  (config-driven paths, dataset-specific ground truth sourcing)

def load_immucan_data(cfg, data_root):
    """
    Returns meta_df[image_id, cell_id, label], folds (image-level: lists of
    image ids), img_paths, seg_paths.

    Ground truth comes from CellTypes/cells2labels/{img}.txt (one integer
    per cell, 1-indexed, -1 = no annotation) resolved against
    CellTypes/labels_cell_type.csv (int -> phenotype name) - this is the
    same table spCellEval's own quantification CSV is built from, verified
    cell-for-cell against it.
    """
    img_dir = data_root / cfg["image_dir"]
    c2l_dir = data_root / cfg["label_dir"]
    seg_dir = data_root / cfg["segmentation_dir"]

    lbl_csv  = pd.read_csv(data_root / cfg["labels_csv"])
    int2type = dict(zip(lbl_csv['label'], lbl_csv['phenotype']))

    img_paths, seg_paths, records = {}, {}, []
    for npz_path in sorted(img_dir.glob('*.npz')):
        if npz_path.name.startswith('.'):
            continue
        img_id = npz_path.stem
        img_paths[img_id] = npz_path

        seg_path = seg_dir / f'{img_id}.tiff'
        if seg_path.exists():
            seg_paths[img_id] = seg_path

        c2l_path = c2l_dir / f'{img_id}.txt'
        if not c2l_path.exists():
            continue
        with open(c2l_path) as f:
            label_ints = [int(line.strip()) for line in f if line.strip()]
        for cell_id, lbl_int in enumerate(label_ints, start=1):
            if lbl_int == -1:
                continue  # genuinely unannotated, not part of the dataset
            if lbl_int not in int2type:
                raise ValueError(
                    f"{c2l_path}: label {lbl_int} for cell {cell_id} has no "
                    f"entry in {cfg['labels_csv']}."
                )
            records.append({'image_id': img_id, 'cell_id': cell_id,
                            'label': int2type[lbl_int]})

    meta_df = pd.DataFrame(records)
    print(f"Loaded IMMUcan metadata: {len(meta_df):,} cells, "
          f"{len(img_paths)} images, {len(seg_paths)} segmentation masks")

    with open(data_root / cfg["folds_json"]) as f:
        raw_folds = json.load(f)
    fold_indices = sorted({int(k.split('_')[1]) for k in raw_folds if k.startswith('fold_')})
    folds = [{'train': raw_folds[f'fold_{i}_train_set'],
              'test':  raw_folds[f'fold_{i}_test_set']} for i in fold_indices]
    print(f"Loaded {len(folds)} folds (image-level)")

    return meta_df, folds, img_paths, seg_paths


def load_chl_data(cfg, data_root):
    """
    Returns meta_df[image_id, cell_id, label], folds (cell-level: sets of
    (image_id, cell_id) tuples), img_paths, seg_paths.

    Ground truth comes directly from the quantification table's own
    'cell_type' column - not 'cell_labels' (a different column some of the
    original data releases carry, which is the authors' own annotation, not
    ours).

    """
    raw_df = pd.read_csv(data_root / cfg["quant_csv"])
    raw_df['sample_id'] = raw_df['sample_id'].astype(str).str.replace('.csv', '', regex=False)

    def load_keys(path):
        df = pd.read_csv(path, usecols=['sample_id', 'cell_id'])
        df['sample_id'] = df['sample_id'].astype(str).str.replace('.csv', '', regex=False)
        return set(zip(df['sample_id'], df['cell_id']))

    folds_dir = data_root / cfg["folds_dir"]
    folds   = []
    fold_i  = 1
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


#  IMAGE / SEGMENTATION READING

def read_immucan_image(img_path, clean_indices):
    arr = np.load(img_path, allow_pickle=True)
    return arr['data'].astype(np.float32)[clean_indices]


def read_chl_image(img_path, clean_indices):
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
                     meta_df, img_paths, seg_paths, read_img_fn):
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
            meta  = pd.read_csv(cache_meta)
            img_feature_store[img_id] = (feats, meta['label'].tolist(), meta['cell_id'].tolist())
            all_feats_list.append(feats)
            meta_records.append(meta)
            continue

        if img_id not in img_paths or img_id not in seg_paths:
            print(f"  [skip] {img_id}: missing image or segmentation")
            continue

        t0 = time.time()
        img_chw = read_img_fn(img_paths[img_id], clean_indices)          # (C, H, W)
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
        description='Eva v3 config-driven pipeline — IMMUcan + cHL_2_MIBI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ['extract', 'supervised', 'leiden', 'all']:
        p = sub.add_parser(mode, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('--dataset', choices=['immucan', 'cHL_2_MIBI'], required=True)
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

    print(f"\n{'='*60}")
    print(f"run_eva.py | mode={args.mode} | dataset={args.dataset}")
    print(f"{'='*60}")

    utils_dir = (Path(args.spceleval_dir) / 'src' / 'methods' / 'utils' if args.spceleval_dir
                else Path(__file__).resolve().parent.parent / 'utils')
    sys.path.insert(0, str(utils_dir))
    from utils_foundational_models import (
        load_embeddings, rebuild_img_feature_store, run_supervised, run_leiden,
    )

    data_root = Path(cfg["data_root"])
    fold_type = cfg["fold_type"]
    if args.dataset == 'immucan':
        meta_df, folds, img_paths, seg_paths = load_immucan_data(cfg, data_root)
        clean_indices, biomarkers = get_immucan_markers(cfg, data_root)
        read_img_fn = read_immucan_image
    else:
        meta_df, folds, img_paths, seg_paths = load_chl_data(cfg, data_root)
        clean_indices, biomarkers = get_chl_markers(cfg)
        read_img_fn = read_chl_image

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
            meta_df, img_paths, seg_paths, read_img_fn,
        )

    if args.mode in ('supervised', 'all'):
        run_supervised(args, folds, img_feature_store, all_feats_arr, metadata_all,
                       method_name=supervised_name, fold_type=fold_type)
    if args.mode in ('leiden', 'all'):
        run_leiden(args, folds, img_feature_store, all_feats_arr, metadata_all,
                   method_name=leiden_name, fold_type=fold_type)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"run_eva.py complete | dataset={args.dataset} | total: {total_time:.1f}s")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
