#!/usr/bin/env python3
"""
run_kronos.py — KRONOS foundation model: config-driven benchmark CLI
====================================================================

Each cell's segmentation bounding box is cropped, zero-padded to
patch_size x patch_size and embedded with KRONOS (MahmoodLab/kronos,
ViT-S/16). The supervised head, Leiden clustering, greedy F1 and output
layout are shared with the other foundation-model methods
(utils/utils_foundational_models.py, utils/benchmark_io.py). Dataset-specific
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

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from benchmark_io import load_config, read_markers, read_image, read_segmentation, load_benchmark_data
from utils_foundational_models import (
    save_embeddings, load_embeddings, rebuild_img_feature_store, check_fold_coverage,
    run_supervised, run_leiden,
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='xFormers is not available')

VIT_PATCH = 16
CHECKPOINT = 'hf_hub:MahmoodLab/kronos'


#  Markers + normalisation

def select_markers(markers, exclude):
    exclude = set(exclude)
    unknown = exclude - set(markers)
    if unknown:
        raise ValueError(f"exclude_markers not found in markers.txt: {sorted(unknown)}")
    keep = [i for i, m in enumerate(markers) if m not in exclude]
    names = [markers[i] for i in keep]
    print(f"Using {len(names)} markers (excluded: {sorted(exclude)})")
    return keep, names


def build_normalisation_params(names, marker_meta_path, marker_name_map, no_equivalent):
    """
    Per-marker mean/std from KRONOS' marker_metadata.csv. Markers without a
    KRONOS equivalent are left unnormalised (mean=0, std=1).
    """
    df = pd.read_csv(marker_meta_path)
    lookup = {name.upper(): (mean, std) for name, mean, std
              in zip(df['marker_name'], df['marker_mean'], df['marker_std'])}
    no_equivalent = set(no_equivalent)

    means, stds = [], []
    print("\nKRONOS marker normalisation params:")
    for m in names:
        kronos_name = marker_name_map.get(m, m)
        if kronos_name.upper() in lookup:
            mean, std = lookup[kronos_name.upper()]
            std = std if std > 0 else 1.0
            status = f"FOUND    -> {kronos_name:20s} mean={mean:.3f} std={std:.3f}"
        else:
            mean, std = 0.0, 1.0
            status = ("FALLBACK -> no KRONOS equivalent" if m in no_equivalent
                      else "MISSING  -> not in KRONOS training data")
        means.append(mean)
        stds.append(std)
        print(f"  {m:25s} {status}")

    n_found = sum(1 for m in names if marker_name_map.get(m, m).upper() in lookup)
    print(f"\nNormalisation: {n_found} markers with KRONOS stats | "
          f"{len(names) - n_found} markers unnormalised")
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


#  Embedding cache

def check_cache_params(output_dir, params):
    """Refuse to reuse cached embeddings written with different extraction parameters."""
    emb_dir = Path(output_dir) / 'embeddings'
    path = emb_dir / 'params.json'
    params = json.loads(json.dumps(params))
    if path.exists():
        with open(path) as f:
            cached = json.load(f)
        if cached != params:
            changed = sorted(k for k in set(cached) | set(params) if cached.get(k) != params.get(k))
            raise ValueError(f"{emb_dir} was extracted with different parameters ({changed}); "
                             f"use a new --output-dir")
        return
    if (emb_dir / 'cache').exists() and any((emb_dir / 'cache').iterdir()):
        raise ValueError(f"{emb_dir} holds cached embeddings without params.json; "
                         f"use a new --output-dir")
    emb_dir.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(params, f, indent=1)


#  KRONOS model

def load_kronos_model(kronos_dir, device):
    """MahmoodLab/kronos is a gated HuggingFace repo (requires `huggingface-cli login`)."""
    sys.path.insert(0, str(kronos_dir))
    from kronos import create_model_from_pretrained

    cache_dir = Path(kronos_dir) / 'model_assets'
    cache_dir.mkdir(parents=True, exist_ok=True)
    model, _, _ = create_model_from_pretrained(checkpoint_path=CHECKPOINT, cache_dir=str(cache_dir))
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


def embed_patches(patches, model, mean_t, std_t, device):
    batch = torch.from_numpy(np.stack(patches)).to(device).permute(0, 3, 1, 2).float()
    batch = (batch - mean_t[None, :, None, None]) / std_t[None, :, None, None]
    with torch.no_grad():
        patch_embeddings, _, _ = model(batch)
    return patch_embeddings.cpu().numpy()


def extract_features(args, model, all_images, keep_idx, means, stds, meta_df, img_paths, seg_paths):
    """Embed every labelled cell, caching per image for crash-safe resumption."""
    cache_dir = Path(args.output_dir) / 'embeddings' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    mean_t = torch.tensor(means, dtype=torch.float32, device=args.device)
    std_t = torch.tensor(stds, dtype=torch.float32, device=args.device)

    print(f"\nExtracting KRONOS features | patch_size={args.patch_size} | "
          f"batch_size={args.batch_size} | device={args.device}")
    print("=" * 60)

    img_feature_store = {}
    for img_id in tqdm(all_images, desc="KRONOS (bbox)"):
        cache_feat = cache_dir / f"{img_id}_feat.npy"
        cache_meta = cache_dir / f"{img_id}_meta.csv"
        if cache_feat.exists() and cache_meta.exists():
            meta = pd.read_csv(cache_meta, dtype={'image_id': str})
            img_feature_store[img_id] = (np.load(cache_feat), meta['label'].tolist(),
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
        bboxes = ndimage.find_objects(mask)

        patches, labels, cell_ids = [], [], []
        for cell_id in np.unique(mask):
            cell_id = int(cell_id)
            if cell_id == 0 or cell_id not in label_lookup:
                continue
            patches.append(get_bbox_patch(img, bboxes[cell_id - 1], args.patch_size))
            labels.append(label_lookup[cell_id])
            cell_ids.append(cell_id)
        if not patches:
            print(f"  Skipping {img_id}: no labelled cells in the segmentation")
            continue

        feats = np.concatenate([
            embed_patches(patches[i:i + args.batch_size], model, mean_t, std_t, args.device)
            for i in range(0, len(patches), args.batch_size)
        ])
        np.save(cache_feat, feats)
        pd.DataFrame({'image_id': img_id, 'cell_id': cell_ids, 'label': labels}).to_csv(
            cache_meta, index=False)
        img_feature_store[img_id] = (feats, labels, cell_ids)

    all_feats_arr = np.concatenate([feats for feats, _, _ in img_feature_store.values()])
    metadata_all = pd.DataFrame({
        'image_id': [img_id for img_id, (_, _, ids) in img_feature_store.items() for _ in ids],
        'cell_id':  [c for _, _, ids in img_feature_store.values() for c in ids],
        'label':    [l for _, labels, _ in img_feature_store.values() for l in labels],
    })
    save_embeddings(args.output_dir, all_feats_arr, metadata_all)
    print(f"\nExtraction complete: {len(all_feats_arr):,} cells | embedding_dim={all_feats_arr.shape[1]}")
    print(metadata_all['label'].value_counts().to_string())
    return img_feature_store, all_feats_arr, metadata_all


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
        p.add_argument('--batch-size', type=int, default=None)
        p.add_argument('--logreg-c', type=float, default=None)
        p.add_argument('--leiden-resolution', type=float, default=None)
        p.add_argument('--leiden-n-neighbors', type=int, default=None)
        p.add_argument('--leiden-subsample', type=int, default=None)
    return parser


DEFAULTS = {
    'device': 'cuda:0',
    'patch_size': 64,
    'batch_size': 16,
    'logreg_c': 1.0,
    'leiden_resolution': 2.0,
    'leiden_n_neighbors': 15,
    'leiden_subsample': 50000,
    'folds_dir': 'quantification/processed/kfolds_StratifiedGroupKFold_level3',
    'include_validation': False,
}


def main():
    args = build_parser().parse_args()
    total_start = time.time()

    data_root = Path(args.dataset_path)
    args.dataset = data_root.name
    cfg = load_config(args.config, args.dataset)
    for attr, default in DEFAULTS.items():
        if getattr(args, attr) is None:
            setattr(args, attr, cfg.get(attr, default))
    if args.patch_size % VIT_PATCH:
        raise ValueError(f"patch_size must be a multiple of {VIT_PATCH} (KRONOS is ViT-S/{VIT_PATCH}), "
                         f"got {args.patch_size}")

    print(f"\n{'=' * 60}\nrun_kronos.py | mode={args.mode} | dataset={args.dataset}\n{'=' * 60}")

    keep_idx, names = select_markers(read_markers(data_root), cfg.get('exclude_markers', []))
    means, stds = build_normalisation_params(
        names, args.marker_meta, cfg.get('marker_name_map', {}), cfg.get('no_equivalent', []))
    meta_df, folds, img_paths, seg_paths = load_benchmark_data(
        data_root, data_root / args.folds_dir, cfg.get('exclude_labels', []), args.include_validation)
    all_images = sorted(img_paths)

    check_cache_params(args.output_dir, {
        'checkpoint': CHECKPOINT,
        'patch_size': args.patch_size,
        'markers': names,
        'normalisation': 'image / image.max(), then per-marker (x - mean) / std',
        'marker_mean': means.tolist(),
        'marker_std': stds.tolist(),
    })

    all_feats_arr, metadata_all = load_embeddings(args.output_dir)
    if all_feats_arr is not None and args.mode in ('supervised', 'leiden'):
        img_feature_store = rebuild_img_feature_store(args.output_dir, all_images)
    else:
        model = load_kronos_model(args.kronos_dir, args.device)
        img_feature_store, all_feats_arr, metadata_all = extract_features(
            args, model, all_images, keep_idx, means, stds, meta_df, img_paths, seg_paths)

    if args.mode != 'extract':
        check_fold_coverage(folds, zip(metadata_all['image_id'], metadata_all['cell_id']))
    if args.mode in ('supervised', 'all'):
        run_supervised(args, folds, img_feature_store, all_feats_arr, metadata_all,
                       method_name='KRONOS_supervised', fold_type='cell')
    if args.mode in ('leiden', 'all'):
        run_leiden(args, folds, img_feature_store, all_feats_arr, metadata_all,
                   method_name='KRONOS_leiden', fold_type='cell')

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}\nrun_kronos.py complete | total: {total_time:.1f}s ({total_time / 3600:.2f}h)\n{'=' * 60}\n")


if __name__ == '__main__':
    main()
