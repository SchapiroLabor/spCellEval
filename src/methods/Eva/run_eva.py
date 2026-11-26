#!/usr/bin/env python3
"""
run_eva.py — Eva Foundation Model: config-driven unified benchmark CLI
======================================================================

Dataset-agnostic CLI for Eva. By default each cell is embedded as in the Eva
preprint's cell phenotyping: a 32x32 block around the cell centroid with every
pixel outside the cell set to 0, placed in the middle of an empty 224x224 patch,
embedded as the mean of all tokens. Each channel is clipped at its 99th percentile
and scaled to [0, 1] (the preprint uses histogram-threshold bounds without
specifying them). --input bbox instead places the cell's whole bounding box,
unmasked, in the empty patch (adaptive bbox, see
eva/project/notebooks/Eva-pipeline-v7.ipynb). --input tile runs Eva on overlapping
full-image tiles and averages the token map over each cell's pixels (our own
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

import time
START = time.perf_counter()  # fold_times.csv: the run is timed from here, package imports included

import argparse
import os
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from benchmark_io import (
    fill_from_config, load_benchmark_data, load_config, read_image, read_markers, read_segmentation,
)
from utils_foundational_models import (
    check_cache_params, load_embeddings, run_leiden, run_supervised, save_embeddings,
)

EVA_REPO_ID = 'yandrewl/Eva'
EVA_PATCH = 224   # Eva's fixed input size
CELL_BLOCK = 32   # block around the centroid in the preprint's cell phenotyping


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


#  PREPROCESSING

def scale_channels(img):
    """Clip each channel of a (C, H, W) image at its 99th percentile and scale it to [0, 1]."""
    high = np.percentile(img.reshape(len(img), -1), 99, axis=1)[:, None, None]
    return np.clip(img / np.maximum(high, 1e-8), 0, 1).astype(np.float32)


#  FEATURE EXTRACTION  (--input cell / bbox: one patch per cell)

def get_cell_patch(img, seg, cell_id, centroid):
    """CELL_BLOCK x CELL_BLOCK block around the centroid, pixels outside the cell set to 0,
    centred in an empty EVA_PATCH x EVA_PATCH patch. img: (H, W, C)."""
    h, w = seg.shape
    y0, x0 = centroid[0] - CELL_BLOCK // 2, centroid[1] - CELL_BLOCK // 2
    ys, xs = slice(max(y0, 0), min(y0 + CELL_BLOCK, h)), slice(max(x0, 0), min(x0 + CELL_BLOCK, w))
    off = (EVA_PATCH - CELL_BLOCK) // 2
    patch = np.zeros((EVA_PATCH, EVA_PATCH, img.shape[2]), dtype=np.float32)
    patch[off + ys.start - y0:off + ys.stop - y0, off + xs.start - x0:off + xs.stop - x0] = \
        img[ys, xs] * (seg[ys, xs] == cell_id)[..., None]
    return patch


def get_bbox_patch(img, bbox):
    """The cell's bounding box, unmasked, centre-cropped to EVA_PATCH if larger and centred
    in an empty EVA_PATCH x EVA_PATCH patch. img: (H, W, C)."""
    crop = img[bbox]
    h, w = crop.shape[:2]
    y, x = max(0, h // 2 - EVA_PATCH // 2), max(0, w // 2 - EVA_PATCH // 2)
    crop = crop[y:y + EVA_PATCH, x:x + EVA_PATCH]
    h, w = crop.shape[:2]
    oy, ox = (EVA_PATCH - h) // 2, (EVA_PATCH - w) // 2
    patch = np.zeros((EVA_PATCH, EVA_PATCH, img.shape[2]), dtype=np.float32)
    patch[oy:oy + h, ox:ox + w] = crop
    return patch


def extract_patch_features(img, seg, cell_ids, model, biomarkers, device, batch_size, bbox=False):
    """Mean of all Eva tokens per cell patch, as Eva.utils.extract_features(cls=False); the patch
    is get_bbox_patch if bbox else get_cell_patch.
    img: (H, W, C) float32, seg: (H, W) int labels. Returns {cell_id: (D,) float32}."""
    if bbox:
        boxes = ndimage.find_objects(seg)
        patches = (get_bbox_patch(img, boxes[c - 1]) for c in cell_ids)
    else:
        centroids = ndimage.center_of_mass(np.ones_like(seg, dtype=np.float32), seg, cell_ids)
        patches = (get_cell_patch(img, seg, c, (int(round(cy)), int(round(cx))))
                   for c, (cy, cx) in zip(cell_ids, centroids))
    feats = []
    while batch := list(islice(patches, batch_size)):
        t = torch.from_numpy(np.stack(batch)).to(device).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            token_out, _ = model.model.forward_encoder(t, [biomarkers] * len(batch))
        feats.append(token_out[:, :, 1:, :].mean(2).reshape(len(batch), -1).cpu().numpy())
    return dict(zip(cell_ids, np.concatenate(feats))) if feats else {}


#  FEATURE EXTRACTION  (--input tile: spatial token map averaged over each cell's pixels)

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

    all_feats_list, meta_records = [], []
    t0_total = time.time()

    for img_id in sorted(all_images):
        cache_feat = cache_dir / f"{img_id}_feat.npy"
        cache_meta = cache_dir / f"{img_id}_meta.csv"
        if cache_feat.exists() and cache_meta.exists():
            feats = np.load(cache_feat)
            meta = pd.read_csv(cache_meta, dtype={'image_id': str})
            all_feats_list.append(feats)
            meta_records.append(meta)
            continue
        if img_id not in seg_paths:
            print(f"  [skip] {img_id}: no segmentation")
            continue

        t0 = time.time()
        img = scale_channels(read_image(img_paths[img_id], clean_indices))
        img = np.ascontiguousarray(img.transpose(1, 2, 0))
        seg = read_segmentation(seg_paths[img_id])
        if args.input == 'tile':
            cell_feats = extract_cell_features(img, seg, model, biomarkers, args.device,
                                               args.tile_size, args.tile_stride)
        else:
            labelled = set(meta_df.loc[meta_df['image_id'] == img_id, 'cell_id'])
            cell_ids = sorted(labelled & set(np.unique(seg).tolist()) - {0})
            cell_feats = extract_patch_features(img, seg, cell_ids, model, biomarkers, args.device,
                                                args.batch_size, bbox=args.input == 'bbox')

        meta_img = meta_df[(meta_df['image_id'] == img_id) & meta_df['cell_id'].isin(list(cell_feats))]
        if meta_img.empty:
            print(f"  [skip] {img_id}: no cells extracted")
            continue
        feats_arr = np.stack([cell_feats[c] for c in meta_img['cell_id']])
        meta_out = meta_img[['image_id', 'cell_id', 'label']].reset_index(drop=True)
        np.save(cache_feat, feats_arr)
        meta_out.to_csv(cache_meta, index=False)

        all_feats_list.append(feats_arr)
        meta_records.append(meta_out)
        print(f"  {img_id}: {len(feats_arr)} cells | {time.time() - t0:.1f}s")

    all_feats_arr = np.concatenate(all_feats_list, axis=0)
    metadata_all = pd.concat(meta_records, ignore_index=True)
    print(f"\nExtracted {len(all_feats_arr):,} cells total in {time.time() - t0_total:.1f}s")

    save_embeddings(args.output_dir, all_feats_arr, metadata_all)
    return all_feats_arr, metadata_all


def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_eva.py',
        description='Eva config-driven pipeline (dataset-agnostic)',
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
        p.add_argument('--input', choices=['cell', 'bbox', 'tile'], default=None,
                       help='cell: masked 32x32 cell block in an empty 224x224 patch (preprint); '
                            'bbox: unmasked bounding box of the cell in an empty 224x224 patch; '
                            'tile: token map of overlapping full-image tiles')
        p.add_argument('--batch-size', type=int, default=None,
                       help='Cell patches per forward pass (--input cell / bbox)')
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

    # Eva's loader changes the working directory, so resolve user paths first.
    data_root = Path(args.dataset_path).resolve()
    args.output_dir = str(Path(args.output_dir).resolve())
    args.dataset = data_root.name
    cfg = load_config(args.config, args.dataset)
    fill_from_config(args, cfg)

    print(f"\n{'=' * 60}\nrun_eva.py | mode={args.mode} | dataset={args.dataset}\n{'=' * 60}")

    meta_df, folds, img_paths, seg_paths = load_benchmark_data(
        data_root, data_root / args.folds_dir, cfg.get('exclude_labels', []), args.include_validation)
    clean_indices, biomarkers = get_markers(cfg, data_root)
    all_images = sorted(img_paths)

    check_cache_params(Path(args.output_dir) / 'embeddings', {
        'model': EVA_REPO_ID, 'input': args.input, 'tile_size': args.tile_size, 'tile_stride': args.tile_stride,
        'normalisation': 'per-channel clip at the 99th percentile, scaled to [0, 1]', 'markers': biomarkers,
    })

    all_feats_arr, metadata_all = load_embeddings(args.output_dir)
    if all_feats_arr is None or args.mode in ('extract', 'all'):
        model = load_eva_model(args.eva_dir, args.device)
        all_feats_arr, metadata_all = extract_features(
            args, model, all_images, clean_indices, biomarkers, meta_df, img_paths, seg_paths)

    head_start = time.perf_counter()
    if args.mode in ('supervised', 'all'):
        run_supervised(args, folds, all_feats_arr, metadata_all, method_name='EVA_supervised', start=START)
    if args.mode in ('leiden', 'all'):  # timed without the supervised head before it
        run_leiden(args, folds, all_feats_arr, metadata_all, method_name='EVA_leiden',
                   start=START + time.perf_counter() - head_start)

    print(f"\n{'=' * 60}\nrun_eva.py complete | dataset={args.dataset} | "
          f"total: {time.perf_counter() - START:.1f}s\n{'=' * 60}\n")


if __name__ == '__main__':
    main()
