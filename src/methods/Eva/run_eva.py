#!/usr/bin/env python3
"""
run_eva.py — Eva Foundation Model: config-driven unified benchmark CLI
======================================================================

Dataset-agnostic CLI for Eva v3 (the authors' spatial-token-map / full-tile
approach, see eva/project/notebooks/Eva-pipeline-v3.ipynb). Dataset-specific
settings (marker exclusions, marker name map, fold directory) live in
src/methods/configs/eva.json; data access is shared via benchmark_io.py and
the classifier head, Leiden clustering and output writing via
utils_foundational_models.py.

Modes
-----
extract     Run Eva on every image, cache per-cell embeddings.
supervised  Logistic-regression classifier head on frozen embeddings.
leiden      Unsupervised Leiden clustering + greedy majority-label F1.
all         extract -> supervised -> leiden.

Usage
-----
python run_eva.py all \\
    --dataset-path <data_dir>/IMMUcan \\
    --config       src/methods/configs/eva.json \\
    --eva-dir      <eva_repo>/project \\
    --output-dir   <results_dir>/eva/IMMUcan
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from benchmark_io import load_benchmark_data, load_config, read_image, read_markers, read_segmentation
from utils_foundational_models import (
    check_fold_coverage, load_embeddings, rebuild_img_feature_store, run_leiden,
    run_supervised, save_embeddings,
)

EVA_REPO_ID = 'yandrewl/Eva'


def load_eva_model(eva_dir, device):
    eva_dir = Path(eva_dir)
    sys.path.insert(0, str(eva_dir))
    os.chdir(str(eva_dir))

    from omegaconf import OmegaConf
    from Eva.utils import load_from_hf

    conf = OmegaConf.load(eva_dir / 'config.yaml')
    model = load_from_hf(repo_id=EVA_REPO_ID, conf=conf, device=device)
    model.eval()
    print(f"Eva loaded on {next(model.parameters()).device}")
    return model


def get_markers(cfg, data_root):
    """Channel order from markers.txt; exclusions and Eva name mapping from the config."""
    all_markers = read_markers(data_root)
    exclude_set = set(cfg.get('exclude_markers', []))
    name_map = cfg.get('marker_name_map', {})
    clean_indices = [i for i, m in enumerate(all_markers) if m not in exclude_set]
    biomarkers = [name_map.get(all_markers[i], all_markers[i]) for i in clean_indices]
    print(f"Markers: {len(all_markers)} total, "
          f"{len(clean_indices)} kept (excluded: {sorted(exclude_set)})")
    return clean_indices, biomarkers


def check_cache_params(output_dir, params):
    """Refuse to mix embeddings extracted with different settings in one output directory."""
    emb_dir = Path(output_dir) / 'embeddings'
    params_file = emb_dir / 'params.json'
    if params_file.exists():
        with open(params_file) as f:
            cached = json.load(f)
        if cached != params:
            diff = {k: (cached.get(k), params.get(k)) for k in set(cached) | set(params)
                    if cached.get(k) != params.get(k)}
            raise ValueError(f"{emb_dir} was extracted with different settings "
                             f"(cached, requested): {diff}. Use a new --output-dir.")
        return
    cache_dir = emb_dir / 'cache'
    if (emb_dir / 'all_cell_features.npy').exists() or (cache_dir.exists() and any(cache_dir.iterdir())):
        raise ValueError(f"{emb_dir} contains embeddings without params.json; "
                         f"their extraction settings are unknown. Use a new --output-dir.")
    emb_dir.mkdir(parents=True, exist_ok=True)
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)


#  FEATURE EXTRACTION  (v3: spatial token map averaged over each cell's pixels)

def _tile_starts(size, tile_size, stride):
    starts = list(range(0, max(1, size - tile_size + 1), stride))
    if starts[-1] + tile_size < size:
        starts.append(size - tile_size)
    return starts


def _encode_tile(model, tile, biomarkers, device, tile_size):
    """Eva tokens of one tile as a (grid, grid, D) array (CLS dropped, mean over marker channels)."""
    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
        padded = np.zeros((tile_size, tile_size, tile.shape[2]), dtype=np.float32)
        padded[:tile.shape[0], :tile.shape[1]] = tile
        tile = padded
    t = torch.from_numpy(tile).unsqueeze(0).to(device)
    with torch.no_grad():
        token_out, _ = model.model.forward_encoder(t.permute(0, 3, 1, 2).contiguous(), [biomarkers])
    spatial = token_out[0, :, 1:, :].mean(dim=0).cpu().numpy()
    grid = int(np.sqrt(spatial.shape[0]))
    return spatial.reshape(grid, grid, -1)


def extract_cell_features(img, seg, model, biomarkers, device, tile_size, stride):
    """
    Mean Eva token vector over each cell's pixels, where a pixel's token vector is the
    average over all overlapping tiles covering it (zero if no token block covers it).
    Accumulated per cell and tile, so the H x W x D token map is never materialised.
    img: (H, W, C) float32, seg: (H, W) int labels. Returns {cell_id: (D,) float32}.
    """
    H, W, _ = img.shape
    ys, xs = _tile_starts(H, tile_size, stride), _tile_starts(W, tile_size, stride)
    tiles = [(y0, x0) for y0 in ys for x0 in xs]

    first = _encode_tile(model, img[:tile_size, :tile_size], biomarkers, device, tile_size)
    grid, feat_dim = first.shape[0], first.shape[2]
    token_size = tile_size // grid
    covered = grid * token_size

    coverage = np.zeros((H, W), dtype=np.float32)
    for y0, x0 in tiles:
        coverage[y0:min(y0 + covered, H), x0:min(x0 + covered, W)] += 1.0
    pixel_weight = 1.0 / np.maximum(coverage, 1.0)

    seg_flat = seg.ravel()
    n_labels = int(seg_flat.max()) + 1
    sums = np.zeros((n_labels, feat_dim), dtype=np.float64)

    for i, (y0, x0) in enumerate(tiles):
        tokens = first if i == 0 else _encode_tile(
            model, img[y0:y0 + tile_size, x0:x0 + tile_size], biomarkers, device, tile_size)
        h, w = min(covered, H - y0), min(covered, W - x0)
        labels = seg[y0:y0 + h, x0:x0 + w].ravel()
        weights = pixel_weight[y0:y0 + h, x0:x0 + w].ravel()
        token_idx = ((np.arange(h) // token_size)[:, None] * grid
                     + (np.arange(w) // token_size)[None, :]).ravel()
        cells, rows = np.unique(labels, return_inverse=True)
        mixing = coo_matrix((weights, (rows, token_idx)), shape=(len(cells), grid * grid)).tocsr()
        sums[cells] += mixing @ tokens.reshape(grid * grid, feat_dim).astype(np.float64)

    n_pixels = np.bincount(seg_flat, minlength=n_labels)
    return {cell_id: (sums[cell_id] / n_pixels[cell_id]).astype(np.float32)
            for cell_id in np.flatnonzero(n_pixels) if cell_id != 0}


def extract_features(args, model, all_images, clean_indices, biomarkers,
                     meta_df, img_paths, seg_paths):
    """Per-image extraction with a per-image cache, then the combined embedding files."""
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
            meta = pd.read_csv(cache_meta, dtype={'image_id': str})
            img_feature_store[img_id] = (feats, meta['label'].tolist(), meta['cell_id'].tolist())
            all_feats_list.append(feats)
            meta_records.append(meta)
            continue
        if img_id not in seg_paths:
            print(f"  [skip] {img_id}: no segmentation")
            continue

        t0 = time.time()
        img = np.ascontiguousarray(read_image(img_paths[img_id], clean_indices).transpose(1, 2, 0))
        seg = read_segmentation(seg_paths[img_id])
        cell_feats = extract_cell_features(img, seg, model, biomarkers, args.device,
                                           args.tile_size, args.tile_stride)

        meta_img = meta_df[(meta_df['image_id'] == img_id) & meta_df['cell_id'].isin(list(cell_feats))]
        if meta_img.empty:
            print(f"  [skip] {img_id}: no cells extracted")
            continue
        feats_arr = np.stack([cell_feats[c] for c in meta_img['cell_id']])
        meta_out = meta_img[['image_id', 'cell_id', 'label']].reset_index(drop=True)
        np.save(cache_feat, feats_arr)
        meta_out.to_csv(cache_meta, index=False)

        img_feature_store[img_id] = (feats_arr, meta_out['label'].tolist(), meta_out['cell_id'].tolist())
        all_feats_list.append(feats_arr)
        meta_records.append(meta_out)
        print(f"  {img_id}: {len(feats_arr)} cells | {time.time() - t0:.1f}s")

    all_feats_arr = np.concatenate(all_feats_list, axis=0)
    metadata_all = pd.concat(meta_records, ignore_index=True)
    print(f"\nExtracted {len(all_feats_arr):,} cells total in {time.time() - t0_total:.1f}s")

    save_embeddings(args.output_dir, all_feats_arr, metadata_all)
    return img_feature_store, all_feats_arr, metadata_all


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
        p.add_argument('--config', required=True, help='Path to eva.json')
        p.add_argument('--eva-dir', required=True, help='Eva project root')
        p.add_argument('--output-dir', required=True)
        p.add_argument('--device', default=None)
        p.add_argument('--tile-size', type=int, default=None)
        p.add_argument('--tile-stride', type=int, default=None)
        p.add_argument('--logreg-c', type=float, default=None)
        p.add_argument('--n-jobs', type=int, default=-1, help='Threads for Leiden KNN cluster assignment')
        p.add_argument('--leiden-resolution', type=float, default=None)
        p.add_argument('--leiden-n-neighbors', type=int, default=None)
        p.add_argument('--leiden-subsample', type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    total_start = time.time()

    # Eva's loader changes the working directory, so resolve user paths first.
    data_root = Path(args.dataset_path).resolve()
    args.output_dir = str(Path(args.output_dir).resolve())
    args.dataset = data_root.name
    cfg = load_config(args.config, args.dataset)

    def fill(attr, key, default):
        if getattr(args, attr) is None:
            setattr(args, attr, cfg.get(key, default))
    fill('device', 'device', 'cuda:0')
    fill('tile_size', 'tile_size', 224)
    fill('tile_stride', 'tile_stride', 112)
    fill('logreg_c', 'logreg_c', 1.0)
    fill('leiden_resolution', 'leiden_resolution', 2.0)
    fill('leiden_n_neighbors', 'leiden_n_neighbors', 15)
    fill('leiden_subsample', 'leiden_subsample', 50000)
    fill('folds_dir', 'folds_dir', 'quantification/processed/kfolds_StratifiedGroupKFold_level3')
    fill('include_validation', 'include_validation', True)

    print(f"\n{'=' * 60}\nrun_eva.py | mode={args.mode} | dataset={args.dataset}\n{'=' * 60}")

    meta_df, folds, img_paths, seg_paths = load_benchmark_data(
        data_root, data_root / args.folds_dir, cfg.get('exclude_labels', []), args.include_validation)
    clean_indices, biomarkers = get_markers(cfg, data_root)
    all_images = sorted(img_paths)

    check_cache_params(args.output_dir, {
        'model': EVA_REPO_ID, 'tile_size': args.tile_size, 'tile_stride': args.tile_stride,
        'markers': biomarkers,
    })

    all_feats_arr, metadata_all = load_embeddings(args.output_dir)
    if all_feats_arr is not None and args.mode in ('supervised', 'leiden'):
        img_feature_store = rebuild_img_feature_store(args.output_dir, all_images)
    else:
        model = load_eva_model(args.eva_dir, args.device)
        img_feature_store, all_feats_arr, metadata_all = extract_features(
            args, model, all_images, clean_indices, biomarkers, meta_df, img_paths, seg_paths)
    check_fold_coverage(folds, zip(metadata_all['image_id'], metadata_all['cell_id']))

    if args.mode in ('supervised', 'all'):
        run_supervised(args, folds, img_feature_store, all_feats_arr, metadata_all,
                       method_name='EVA_supervised', fold_type='cell')
    if args.mode in ('leiden', 'all'):
        run_leiden(args, folds, img_feature_store, all_feats_arr, metadata_all,
                   method_name='EVA_leiden', fold_type='cell')

    print(f"\n{'=' * 60}\nrun_eva.py complete | dataset={args.dataset} | "
          f"total: {time.time() - total_start:.1f}s\n{'=' * 60}\n")


if __name__ == '__main__':
    main()
