#!/usr/bin/env python3
"""
run_kronos.py — KRONOS foundation model: config-driven benchmark CLI
====================================================================

Cells are embedded with KRONOS (MahmoodLab/kronos, ViT-S/16) as in the
authors' cell-phenotyping tutorial: a patch_size x patch_size window centred on
the cell centroid, standardised per marker, every pixel outside the cell mask set
to 0, overlapping tokens (stride 8), and the per-marker features flattened
(n_markers x 384) as the embedding. The only deviation: each image is scaled to
[0, 1] by its maximum instead of by the dtype maximum (65535), because the
tutorial's divisor assumes 16-bit data and our images do not use that range.
--patch-mode bbox instead crops the cell's bounding box and zero-pads it,
without masking. The supervised head, Leiden clustering, greedy F1 and output
layout are shared with the other foundation-model methods
(utils/utils_foundational_models.py, utils/benchmark_io.py), except that the
predictions CSVs have no emb_* columns: with n_markers x 384 dimensions they
would be ~36x larger, so the embeddings are kept once, in
embeddings/all_cell_features.npy. Dataset-specific
settings (marker exclusions, KRONOS name mappings) live in configs/kronos.json,
keyed by the dataset directory name.

Modes
-----
extract     Embed every labelled cell and cache the embeddings.
supervised  Logistic-regression head on frozen embeddings, one per fold.
leiden      Leiden clustering + greedy majority-label F1.
all         extract -> supervised -> leiden.

Usage
-----
python run_kronos.py all \\
    --dataset-path <data>/IMMUcan \\
    --config       src/methods/configs/kronos.json \\
    --kronos-dir   <KRONOS repo> \\
    --marker-meta  <KRONOS repo>/model_assets/marker_metadata.csv \\
    --output-dir   <results>/IMMUcan/kronos \\
    --device       cuda:0
"""

import time
START = time.perf_counter()  # fold_times.csv: the run is timed from here, package imports included

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from benchmark_io import (
    fill_from_config, load_config, read_markers, read_image, read_segmentation, load_benchmark_data,
)
from utils_foundational_models import (
    save_embeddings, load_embeddings, check_cache_params, run_supervised, run_leiden,
)

VIT_PATCH = 16
CHECKPOINT = 'hf_hub:MahmoodLab/kronos'


#  Markers

def select_markers(markers, exclude):
    exclude = set(exclude)
    unknown = exclude - set(markers)
    if unknown:
        raise ValueError(f"exclude_markers not found in markers.txt: {sorted(unknown)}")
    keep = [i for i, m in enumerate(markers) if m not in exclude]
    names = [markers[i] for i in keep]
    print(f"Using {len(names)} markers (excluded: {sorted(exclude)})")
    return keep, names


def build_marker_params(names, marker_meta_path, marker_name_map):
    """Per-marker KRONOS marker_id, mean and std from KRONOS' marker_metadata.csv."""
    df = pd.read_csv(marker_meta_path)
    lookup = {name.upper(): (mid, mean, std) for name, mid, mean, std
              in zip(df['marker_name'], df['marker_id'], df['marker_mean'], df['marker_std'])}
    kronos_names = [marker_name_map.get(m, m) for m in names]
    missing = [m for m, k in zip(names, kronos_names) if k.upper() not in lookup]
    if missing:
        raise ValueError(f"No KRONOS marker for {missing}; add them to marker_name_map or exclude_markers")

    ids, means, stds = zip(*(lookup[k.upper()] for k in kronos_names))
    print("\nKRONOS markers:")
    for m, k, mid in zip(names, kronos_names, ids):
        print(f"  {m:25s} -> {k:20s} id={mid}")
    return (np.array(ids, dtype=np.int64), np.array(means, dtype=np.float32),
            np.array([s if s > 0 else 1.0 for s in stds], dtype=np.float32))


#  KRONOS model

def load_kronos_model(kronos_dir, device, token_overlap):
    """MahmoodLab/kronos is a gated HuggingFace repo (requires `huggingface-cli login`)."""
    sys.path.insert(0, str(kronos_dir))
    from kronos import create_model_from_pretrained

    cache_dir = Path(kronos_dir) / 'model_assets'
    cache_dir.mkdir(parents=True, exist_ok=True)
    model, _, _ = create_model_from_pretrained(
        checkpoint_path=CHECKPOINT, cache_dir=str(cache_dir),
        cfg={'model_type': 'vits16', 'token_overlap': token_overlap})
    model = model.to(device).eval()
    print(f"KRONOS loaded on: {device}")
    return model


#  Patch extraction

def get_bbox_patch(img, bbox, patch_size):
    """Crop a cell's bounding box (centre-cropped if larger) and zero-pad it to patch_size."""
    crop = img[bbox[0], bbox[1], :]
    crop_h, crop_w = crop.shape[:2]
    if crop_h > patch_size or crop_w > patch_size:
        cy, cx = crop_h // 2, crop_w // 2
        half = patch_size // 2
        crop = crop[max(0, cy - half):cy + half, max(0, cx - half):cx + half, :]
        crop_h, crop_w = crop.shape[:2]
    pad_h, pad_w = patch_size - crop_h, patch_size - crop_w
    return np.pad(crop, ((pad_h // 2, pad_h - pad_h // 2),
                         (pad_w // 2, pad_w - pad_w // 2),
                         (0, 0)))


def get_centroid_patch(img, mask, cell_id, centroid, patch_size):
    """Window of patch_size centred on the cell centroid, zero-padded only at image borders,
    plus the binary mask of that cell within the window."""
    h, w = mask.shape
    y0, x0 = centroid[0] - patch_size // 2, centroid[1] - patch_size // 2
    ys, xs = slice(max(y0, 0), min(y0 + patch_size, h)), slice(max(x0, 0), min(x0 + patch_size, w))
    inner = (slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0))
    patch = np.zeros((patch_size, patch_size, img.shape[2]), dtype=img.dtype)
    cell_mask = np.zeros((patch_size, patch_size), dtype=bool)
    patch[inner] = img[ys, xs]
    cell_mask[inner] = mask[ys, xs] == cell_id
    return patch, cell_mask


def embed_patches(patches, cell_masks, model, ids_t, mean_t, std_t, device):
    """Standardise per marker, zero everything outside the cell (if masks are given) and
    return the flattened marker features (n_cells, n_markers * 384)."""
    batch = torch.from_numpy(np.stack(patches)).to(device).permute(0, 3, 1, 2).float()
    batch = (batch - mean_t[None, :, None, None]) / std_t[None, :, None, None]
    if cell_masks is not None:
        batch = batch * torch.from_numpy(np.stack(cell_masks)).to(device)[:, None]
    with torch.no_grad():
        _, marker_features, _ = model(batch, marker_ids=ids_t.expand(len(batch), -1))
    return marker_features.reshape(len(batch), -1).cpu().numpy()


def extract_features(args, model, all_images, keep_idx, ids, means, stds, meta_df, img_paths, seg_paths):
    """Embed every labelled cell, caching per image for crash-safe resumption."""
    cache_dir = Path(args.output_dir) / 'embeddings' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids_t = torch.tensor(ids, dtype=torch.int64, device=args.device)
    mean_t = torch.tensor(means, dtype=torch.float32, device=args.device)
    std_t = torch.tensor(stds, dtype=torch.float32, device=args.device)

    print(f"\nExtracting KRONOS features | patch_mode={args.patch_mode} | patch_size={args.patch_size} | "
          f"token_overlap={args.token_overlap} | batch_size={args.batch_size} | device={args.device}")
    print("=" * 60)

    per_image = {}
    for img_id in tqdm(all_images, desc=f"KRONOS ({args.patch_mode})"):
        cache_feat = cache_dir / f"{img_id}_feat.npy"
        cache_meta = cache_dir / f"{img_id}_meta.csv"
        if cache_feat.exists() and cache_meta.exists():
            meta = pd.read_csv(cache_meta, dtype={'image_id': str})
            per_image[img_id] = (np.load(cache_feat), meta['label'].tolist(),
                                         meta['cell_id'].tolist())
            continue
        if img_id not in img_paths or img_id not in seg_paths:
            print(f"  Skipping {img_id}: missing image or segmentation")
            continue

        meta_img = meta_df[meta_df['image_id'] == img_id]
        label_lookup = dict(zip(meta_img['cell_id'], meta_img['label']))

        img = read_image(img_paths[img_id], keep_idx)
        img = (img / (img.max() + 1e-8)).transpose(1, 2, 0)
        mask = read_segmentation(seg_paths[img_id])
        cell_ids = [int(c) for c in np.unique(mask) if c != 0 and int(c) in label_lookup]
        if not cell_ids:
            print(f"  Skipping {img_id}: no labelled cells in the segmentation")
            continue
        labels = [label_lookup[c] for c in cell_ids]

        if args.patch_mode == 'centroid':
            centroids = ndimage.center_of_mass(np.ones_like(mask, dtype=np.float32), mask, cell_ids)
            patches, cell_masks = zip(*(
                get_centroid_patch(img, mask, c, (int(round(cy)), int(round(cx))), args.patch_size)
                for c, (cy, cx) in zip(cell_ids, centroids)))
        else:
            bboxes = ndimage.find_objects(mask)
            patches = [get_bbox_patch(img, bboxes[c - 1], args.patch_size) for c in cell_ids]
            cell_masks = None

        feats = np.concatenate([
            embed_patches(patches[i:i + args.batch_size],
                          None if cell_masks is None else cell_masks[i:i + args.batch_size],
                          model, ids_t, mean_t, std_t, args.device)
            for i in range(0, len(patches), args.batch_size)
        ])
        np.save(cache_feat, feats)
        pd.DataFrame({'image_id': img_id, 'cell_id': cell_ids, 'label': labels}).to_csv(
            cache_meta, index=False)
        per_image[img_id] = (feats, labels, cell_ids)

    all_feats_arr = np.concatenate([feats for feats, _, _ in per_image.values()])
    metadata_all = pd.DataFrame({
        'image_id': [img_id for img_id, (_, _, ids) in per_image.items() for _ in ids],
        'cell_id':  [c for _, _, ids in per_image.values() for c in ids],
        'label':    [l for _, labels, _ in per_image.values() for l in labels],
    })
    save_embeddings(args.output_dir, all_feats_arr, metadata_all)
    print(f"\nExtraction complete: {len(all_feats_arr):,} cells | embedding_dim={all_feats_arr.shape[1]}")
    print(metadata_all['label'].value_counts().to_string())
    return all_feats_arr, metadata_all


#  CLI

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_kronos.py',
        description='KRONOS foundation model — config-driven pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode, help_text in [('extract', 'Extract KRONOS embeddings only'),
                            ('supervised', 'Logistic-regression head per fold'),
                            ('leiden', 'Leiden clustering + greedy F1'),
                            ('all', 'extract -> supervised -> leiden')]:
        p = sub.add_parser(mode, help=help_text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('--dataset-path', required=True,
                       help='Dataset directory (markers.txt, quantification/, raw_images/, segmentation/)')
        p.add_argument('--config', required=True, help='Path to kronos.json')
        p.add_argument('--kronos-dir', required=True, help='KRONOS repo directory (contains kronos/)')
        p.add_argument('--marker-meta', required=True, help='marker_metadata.csv from MahmoodLab/kronos')
        p.add_argument('--output-dir', required=True)
        p.add_argument('--folds-dir', default=None,
                       help='k-fold directory, absolute or relative to --dataset-path')
        p.add_argument('--include-validation', action=argparse.BooleanOptionalAction, default=None,
                       help='Add fold_{i}_validation.csv cells to the training set')
        p.add_argument('--spceleval-dir', default=None)
        p.add_argument('--device', default=None)
        p.add_argument('--patch-size', type=int, default=None)
        p.add_argument('--patch-mode', choices=['centroid', 'bbox'], default=None,
                       help='centroid: masked window centred on the cell (authors); '
                            'bbox: zero-padded bounding-box crop, unmasked')
        p.add_argument('--token-overlap', action=argparse.BooleanOptionalAction, default=None,
                       help='Overlapping tokens (stride 8), as in the authors\' tutorial')
        p.add_argument('--batch-size', type=int, default=None)
        p.add_argument('--logreg-c', type=float, default=None)
        p.add_argument('--leiden-resolution', type=float, default=None)
        p.add_argument('--leiden-n-neighbors', type=int, default=None)
        p.add_argument('--leiden-subsample', type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()

    data_root = Path(args.dataset_path)
    args.dataset = data_root.name
    cfg = load_config(args.config, args.dataset)
    fill_from_config(args, cfg)
    if args.patch_size % VIT_PATCH:
        raise ValueError(f"patch_size must be a multiple of {VIT_PATCH} (KRONOS is ViT-S/{VIT_PATCH}), "
                         f"got {args.patch_size}")

    print(f"\n{'=' * 60}\nrun_kronos.py | mode={args.mode} | dataset={args.dataset}\n{'=' * 60}")

    keep_idx, names = select_markers(read_markers(data_root), cfg.get('exclude_markers', []))
    ids, means, stds = build_marker_params(names, args.marker_meta, cfg.get('marker_name_map', {}))
    meta_df, folds, img_paths, seg_paths = load_benchmark_data(
        data_root, data_root / args.folds_dir, cfg.get('exclude_labels', []), args.include_validation)
    all_images = sorted(img_paths)

    check_cache_params(Path(args.output_dir) / 'embeddings', {
        'checkpoint': CHECKPOINT,
        'patch_size': args.patch_size,
        'patch_mode': args.patch_mode,
        'token_overlap': args.token_overlap,
        'embedding': 'marker_features',
        'markers': names,
        'marker_ids': ids.tolist(),
        'normalisation': 'image / image.max(), then per-marker (x - mean) / std',
        'marker_mean': means.tolist(),
        'marker_std': stds.tolist(),
    })

    all_feats_arr, metadata_all = load_embeddings(args.output_dir)
    if all_feats_arr is None or args.mode in ('extract', 'all'):
        model = load_kronos_model(args.kronos_dir, args.device, args.token_overlap)
        all_feats_arr, metadata_all = extract_features(
            args, model, all_images, keep_idx, ids, means, stds, meta_df, img_paths, seg_paths)

    head_start = time.perf_counter()
    if args.mode in ('supervised', 'all'):
        run_supervised(args, folds, all_feats_arr, metadata_all, method_name='KRONOS_supervised', start=START,
                       embedding_columns=False)
    if args.mode in ('leiden', 'all'):  # timed without the supervised head before it
        run_leiden(args, folds, all_feats_arr, metadata_all, method_name='KRONOS_leiden',
                   start=START + time.perf_counter() - head_start,
                   embedding_columns=False)

    total_time = time.perf_counter() - START
    print(f"\n{'=' * 60}\nrun_kronos.py complete | total: {total_time:.1f}s ({total_time / 3600:.2f}h)\n{'=' * 60}\n")


if __name__ == '__main__':
    main()
