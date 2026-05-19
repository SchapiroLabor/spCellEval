#!/usr/bin/env python3
"""
run_kronos.py — KRONOS Foundation Model: IMMUcan Benchmark CLI

KRONOS-specific script: model loading + feature extraction only.
Supervised RF, Leiden clustering, greedy F1, output saving
are all handled by utils_foundational_models.py (shared with run_eva.py etc.)

Usage examples
--------------
# Extract embeddings only
python run_kronos.py extract \\
    --data-dir        /path/to/IMMUcan \\
    --kronos-dir      /home/juliaoesterle/KRONOS_mine \\
    --marker-meta     /home/juliaoesterle/KRONOS_mine/model_assets/marker_metadata.csv \\
    --output-dir      /path/to/output \\
    --device          cuda:0

# Supervised Random Forest
python run_kronos.py supervised \\
    --data-dir        /path/to/IMMUcan \\
    --kronos-dir      /home/juliaoesterle/KRONOS_mine \\
    --marker-meta     /home/juliaoesterle/KRONOS_mine/model_assets/marker_metadata.csv \\
    --output-dir      /path/to/output \\
    --n-estimators    200 \\
    --n-jobs          -1

# Leiden + greedy F1
python run_kronos.py leiden \\
    --data-dir            /path/to/IMMUcan \\
    --kronos-dir          /home/juliaoesterle/KRONOS_mine \\
    --marker-meta         /home/juliaoesterle/KRONOS_mine/model_assets/marker_metadata.csv \\
    --output-dir          /path/to/output \\
    --leiden-resolution   2.0

# Run all three steps in sequence
python run_kronos.py all \\
    --data-dir        /path/to/IMMUcan \\
    --kronos-dir      /home/juliaoesterle/KRONOS_mine \\
    --marker-meta     /home/juliaoesterle/KRONOS_mine/model_assets/marker_metadata.csv \\
    --output-dir      /path/to/output \\
    --device          cuda:0 \\
    --n-jobs          -1

# For Julia
python run_kronos.py all \\
    --data-dir    /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan/ \\
    --kronos-dir  /home/juliaoesterle/KRONOS_mine \\
    --marker-meta /home/juliaoesterle/KRONOS_mine/model_assets/marker_metadata.csv \\
    --output-dir  /home/juliaoesterle/results/kronos \\
    --device      cuda:0

Key differences from Eva
------------------------
- Input: (B, C, H, W) with per-marker normalisation using mean/std
  from marker_metadata.csv (KRONOS was trained on absolute intensities)
- Patch size: 64×64 (flexible, not hard-coded like Eva's 224)
- Output: patch_embeddings (used here), marker_embeddings, token_embeddings
- Marker handling: fuzzy matching via MarkerMetadata class + manual overrides
- Feature: patch_embeddings — single vector per patch (equivalent to CLS token)
- No .contiguous() fix needed (different architecture from Eva)

IMMUcan marker mapping (verified against marker_metadata.csv)
-------------------------------------------------------------
Found directly (26/37):
  MPO, CD16, CD38, CD27, CD15, CD45RA, CD163, B2M, CD20, CD68,
  CD3, LAG3, CD11c, PD1, CD7, PDL1, CD45RO, FOXP3, ICOS, Ki67,
  VISTA, CD40, CD4, CD14, CD206, CD33

Manual mappings (4/37):
  GrzB          → GZMB          (same gene, different antibody name)
  HLADR         → HLA_DR        (same protein)
  CD8a          → CD8           (same protein)
  cleavedPARP   → CLEAVED_CASP3 (apoptosis proxy)

No KRONOS equivalent — mean=0 std=1 fallback (7/37):
  SMA, PDGFRb, TCF7, CarbonicAnhydrase, Ecad, CD33, CD303
  (these markers were not in KRONOS training data)

Excluded from pipeline (3/40):
  DNA1, DNA2, HistoneH3 (structural stains, no biological equivalent)

Output structure
----------------
{output_dir}/
├── embeddings/
│   ├── cache/                              ← per-image cache (crash-safe)
│   ├── all_cell_features.npy               ← (N_cells × embedding_dim)
│   ├── all_cell_metadata.csv               ← image_id, cell_id, label
│   └── leiden_clusters_res{X}.csv          ← Leiden + UMAP (leiden mode)
├── KRONOS_supervised/
│   └── level3/
│       ├── predictions_0.csv ... predictions_4.csv
│       └── fold_times.txt
└── KRONOS_leiden/
    └── level3/
        ├── predictions_0.csv ... predictions_4.csv
        └── fold_times.txt
"""

# Standard library 
import os
import sys
import time
import argparse
import warnings
from pathlib import Path

# CUDA env vars — must be set before torch import 
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.3')
os.environ.setdefault('LD_LIBRARY_PATH',
                      '/usr/local/cuda-12.3/lib64:' +
                      os.environ.get('LD_LIBRARY_PATH', ''))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Third-party 
import numpy as np
import pandas as pd
import tifffile
import torch
from tqdm import tqdm

# Shared benchmark utilities 
# utils_foundational_models.py must be in the same directory as run_kronos.py
sys.path.insert(0, str(Path(__file__).parent))
from utils_foundational_models import (
    load_label_map,
    load_folds,
    get_label,
    save_embeddings,
    load_embeddings,
    rebuild_img_feature_store,
    run_supervised,
    run_leiden,
    add_shared_args,
)

# KRONOS model utilities 
# Imported inside load_kronos_model() because kronos-dir must be on sys.path first.

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# Marker definitions

ALL_BIOMARKERS = [
    "MPO", "HistoneH3", "SMA", "CD16", "CD38", "HLADR", "CD27", "CD15",
    "CD45RA", "CD163", "B2M", "CD20", "CD68", "Ido1", "CD3", "LAG3",
    "CD11c", "PD1", "PDGFRb", "CD7", "GrzB", "PDL1", "TCF7", "CD45RO",
    "FOXP3", "ICOS", "CD8a", "CarbonicAnhydrase", "CD33", "Ki67",
    "VISTA", "CD40", "CD4", "CD14", "Ecad", "CD303", "CD206",
    "cleavedPARP", "DNA1", "DNA2"
]

DEFAULT_EXCLUDE = ['DNA1', 'DNA2', 'HistoneH3']

# Manual name mappings: IMMUcan antibody name → KRONOS marker_name
# Verified against marker_metadata.csv
KRONOS_NAME_MAPPINGS = {
    'GrzB':        'GZMB',           # same gene, different antibody name
    'HLADR':       'HLA_DR',         # same protein
    'CD8a':        'CD8',            # same protein
    'cleavedPARP': 'CLEAVED_CASP3',  # apoptosis proxy — closest available
}

# Markers with no KRONOS equivalent — will use mean=0, std=1 (no normalisation)
KRONOS_NO_EQUIVALENT = {
    'SMA', 'PDGFRb', 'TCF7', 'CarbonicAnhydrase', 'Ecad', 'CD33', 'CD303'
}


def get_clean_markers(exclude):
    exclude_set = set(exclude)
    clean_idx   = [i for i, m in enumerate(ALL_BIOMARKERS) if m not in exclude_set]
    clean_names = [m for m in ALL_BIOMARKERS if m not in exclude_set]
    print(f"Using {len(clean_names)} markers (excluded: {exclude_set})")
    return clean_idx, clean_names


# KRONOS marker normalisation 

def build_normalisation_params(clean_names, marker_meta_path):
    """
    Build per-marker mean and std arrays for KRONOS normalisation.

    KRONOS was trained on absolute marker intensities normalised by
    per-marker mean/std from marker_metadata.csv. For markers not in
    KRONOS training data, mean=0 and std=1 is used (identity normalisation).

    Parameters
    ----------
    clean_names      : list of IMMUcan marker names (after DNA exclusion)
    marker_meta_path : path to marker_metadata.csv

    Returns
    -------
    means : np.ndarray (C,)
    stds  : np.ndarray (C,)
    marker_ids : list of int — KRONOS marker IDs (0 for unknown)
    """
    df = pd.read_csv(marker_meta_path)
    # Build lookup: uppercase name → (marker_id, mean, std)
    lookup = {
        row['marker_name'].upper(): (row['marker_id'], row['marker_mean'], row['marker_std'])
        for _, row in df.iterrows()
    }

    means, stds, marker_ids = [], [], []
    print("\nKRONOS marker normalisation params:")

    for m in clean_names:
        # Apply manual mapping if available
        kronos_name = KRONOS_NAME_MAPPINGS.get(m, m)
        key         = kronos_name.upper()

        if key in lookup:
            mid, mean, std = lookup[key]
            means.append(mean)
            stds.append(std if std > 0 else 1.0)
            marker_ids.append(int(mid))
            status = f"FOUND   → {kronos_name:20s} id={mid} mean={mean:.3f} std={std:.3f}"
        elif m in KRONOS_NO_EQUIVALENT:
            means.append(0.0)
            stds.append(1.0)
            marker_ids.append(0)
            status = f"FALLBACK → mean=0 std=1 (no KRONOS equivalent)"
        else:
            means.append(0.0)
            stds.append(1.0)
            marker_ids.append(0)
            status = f"MISSING  → mean=0 std=1 (not in KRONOS training data)"

        print(f"  {m:25s} {status}")

    means = np.array(means, dtype=np.float32)
    stds  = np.array(stds,  dtype=np.float32)

    n_found   = sum(1 for mid in marker_ids if mid > 0)
    n_missing = sum(1 for mid in marker_ids if mid == 0)
    print(f"\nNormalisation: {n_found} markers with KRONOS stats | "
          f"{n_missing} markers using identity (mean=0, std=1)")

    return means, stds, marker_ids


# KRONOS model loading 

def load_kronos_model(kronos_dir, device):
    """
    Load KRONOS model from HuggingFace checkpoint.
    kronos package is imported here because kronos_dir must be on sys.path first.
    Requires HuggingFace authentication: huggingface-cli auth login
    Model: MahmoodLab/kronos (gated repo — access request required)

    Returns
    -------
    model         : KRONOS model in eval mode
    embedding_dim : int — dimension of patch_embeddings output
    precision     : str — model precision (e.g. 'fp32')
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


# KRONOS-specific patch extraction 

def get_bbox_patch(img, ys, xs, patch_size=64):
    """
    Adaptive bounding box: crop exact cell bbox from segmentation mask
    pixels → symmetrically zero-pad to patch_size × patch_size.

    KRONOS uses 64×64 patches (flexible, unlike Eva's hard-coded 224).
    Mean IMMUcan cell: 9.5×10.6px bbox — well within 64×64.

    Returns: (patch, bbox_h, bbox_w)
    """
    y_min, y_max   = int(ys.min()), int(ys.max())
    x_min, x_max   = int(xs.min()), int(xs.max())
    crop           = img[y_min:y_max+1, x_min:x_max+1, :]
    crop_h, crop_w = crop.shape[:2]

    # Safety: centre-crop if bbox exceeds patch_size (very rare edge case)
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
    return patch, crop_h, crop_w


def extract_features_batch(patches, model, means, stds, device):
    """
    Batched KRONOS forward pass on patch_size × patch_size patches.
    Returns patch_embeddings as cell feature: (N, embedding_dim).

    KRONOS input format: (B, C, H, W) float32
    Normalisation: (x - mean) / std applied per marker channel.

    Unlike Eva, no .contiguous() fix is needed here.
    KRONOS outputs three embeddings — we use patch_embeddings
    (single summary vector per patch, equivalent to Eva's CLS token).
    """
    batch = np.stack(patches)                            # (N, H, W, C)
    batch = torch.from_numpy(batch).to(device)
    batch = batch.permute(0, 3, 1, 2).float()            # (N, C, H, W)

    # Per-marker normalisation using KRONOS training stats
    mean_t = torch.tensor(means, dtype=torch.float32).to(device)
    std_t  = torch.tensor(stds,  dtype=torch.float32).to(device)
    batch  = (batch - mean_t[None, :, None, None]) / std_t[None, :, None, None]

    with torch.no_grad():
        patch_embeddings, marker_embeddings, token_embeddings = model(batch)

    return patch_embeddings.cpu().numpy()                # (N, embedding_dim)


# Feature extraction 

def extract_features(args, model, embedding_dim, all_images,
                     clean_indices, clean_names, means, stds, label_map):
    """
    Extract KRONOS cell embeddings for all images using adaptive bounding box.
    Results are cached per image for crash-safe resumption.
    """
    data_dir  = Path(args.data_dir)
    cache_dir = Path(args.output_dir) / 'embeddings' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    NPZ_DIR    = data_dir / 'CellTypes' / 'data' / 'images'
    MASK_DIR   = data_dir / 'segmentation'
    LABELS_DIR = data_dir / 'CellTypes' / 'cells2labels'

    img_feature_store = {}

    print(f"\nExtracting KRONOS features | patch_size={args.patch_size} | "
          f"batch_size={args.batch_size} | device={args.device}")
    print("=" * 60)

    for img_name in tqdm(all_images, desc="KRONOS (bbox)"):
        cache_feat = cache_dir / f"{img_name}_feat.npy"
        cache_meta = cache_dir / f"{img_name}_meta.csv"

        # Load from cache — skip KRONOS forward passes for this image
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

        # Load image — KRONOS expects raw intensities (normalised internally)
        # select clean channels and scale to [0, 1] before normalisation
        img = np.load(npz_file)['data'].astype(np.float32)
        img = img[clean_indices]
        img = img / (img.max() + 1e-8)     # scale to [0,1] before KRONOS normalisation
        img = img.transpose(1, 2, 0)       # (H, W, C)

        # Load mask and ground truth labels
        mask = tifffile.imread(mask_file)
        with open(label_file) as f:
            cell_labels = [int(line.strip()) for line in f.readlines()]

        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids > 0]

        # Adaptive bounding box per cell
        patches_buf, labels_buf, cell_ids_buf = [], [], []
        for cell_id in cell_ids:
            label  = get_label(int(cell_id), cell_labels, label_map)
            ys, xs = np.where(mask == cell_id)
            patch, _, _ = get_bbox_patch(img, ys, xs, args.patch_size)
            patches_buf.append(patch)
            labels_buf.append(label)
            cell_ids_buf.append(int(cell_id))

        # Batched KRONOS forward passes
        all_feats = []
        for i in range(0, len(patches_buf), args.batch_size):
            batch_feats = extract_features_batch(
                patches_buf[i:i + args.batch_size],
                model, means, stds, args.device
            )
            all_feats.extend(batch_feats)
            torch.cuda.empty_cache()

        feats_arr = np.array(all_feats)

        # Save per-image cache
        np.save(cache_feat, feats_arr)
        pd.DataFrame({
            'image_id': [img_name] * len(cell_ids_buf),
            'cell_id':  cell_ids_buf,
            'label':    labels_buf,
        }).to_csv(cache_meta, index=False)

        img_feature_store[img_name] = (feats_arr, labels_buf, cell_ids_buf)

    # Combine all images into single arrays
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


# Argument Parser 

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_kronos.py',
        description='KRONOS Foundation Model — IMMUcan Benchmark CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(dest='mode', required=True,
                                help='Pipeline mode')
    for m in ['extract', 'supervised', 'leiden', 'all']:
        p = sub.add_parser(
            m,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            help={
                'extract':    'Extract KRONOS embeddings only (reusable downstream)',
                'supervised': 'Supervised Random Forest + 5-fold CV',
                'leiden':     'Leiden clustering + greedy F1',
                'all':        'extract -> supervised -> leiden in sequence',
            }[m]
        )

        # KRONOS-specific arguments
        p.add_argument('--kronos-dir', required=True,
                       help='KRONOS repo directory (contains kronos/ package)')
        p.add_argument('--marker-meta', required=True,
                       help='Path to marker_metadata.csv from MahmoodLab/kronos')
        p.add_argument('--patch-size', type=int, default=64,
                       help='Patch size for cell crops (KRONOS default: 64)')
        p.add_argument('--batch-size', type=int, default=16,
                       help='Cells per KRONOS forward pass')
        p.add_argument('--exclude-markers', nargs='+',
                       default=DEFAULT_EXCLUDE,
                       help='Markers to exclude (no biological equivalent)')
        p.add_argument('--device', default='cuda:0',
                       help='PyTorch device (cuda:0, cuda:1, cpu)')

        # Shared arguments from utils_foundational_models
        add_shared_args(p)

    return parser


# Main 

def main():
    parser = build_parser()
    args   = parser.parse_args()

    total_start = time.time()
    print(f"\n{'='*60}")
    print(f"run_kronos.py | mode={args.mode}")
    print(f"{'='*60}")

    # Setup
    clean_indices, clean_names = get_clean_markers(args.exclude_markers)
    label_map                  = load_label_map(args.data_dir)
    folds, all_images          = load_folds(args.data_dir, args.n_folds)
    means, stds, marker_ids    = build_normalisation_params(
                                     clean_names, args.marker_meta)

    # Output folder names for spCellEval structure
    supervised_name = 'KRONOS_supervised'
    leiden_name     = 'KRONOS_leiden'

    # Try loading precomputed embeddings — skip extraction if available
    all_feats_arr, metadata_all = load_embeddings(args.output_dir)

    if all_feats_arr is not None and args.mode in ('supervised', 'leiden'):
        img_feature_store = rebuild_img_feature_store(args.output_dir, all_images)
    else:
        # Load KRONOS model and extract features
        model, embedding_dim, precision = load_kronos_model(
            args.kronos_dir, args.device
        )
        img_feature_store, all_feats_arr, metadata_all = extract_features(
            args, model, embedding_dim, all_images,
            clean_indices, clean_names, means, stds, label_map
        )

    # Run requested mode(s)
    if args.mode in ('supervised', 'all'):
        run_supervised(
            args, folds, img_feature_store,
            all_feats_arr, metadata_all,
            method_name=supervised_name
        )

    if args.mode in ('leiden', 'all'):
        run_leiden(
            args, folds, img_feature_store,
            all_feats_arr, metadata_all,
            method_name=leiden_name
        )

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"run_kronos.py complete | total: {total_time:.1f}s "
          f"({total_time/3600:.2f}h)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()