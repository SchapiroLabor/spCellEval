#!/usr/bin/env python3
"""
run_eva.py — Eva Foundation Model: IMMUcan Benchmark CLI

Eva-specific script: model loading + feature extraction only.
Supervised RF, Leiden clustering, greedy F1, output saving
are all handled by utils_benchmark.py (shared with run_kronos.py etc.)

Usage examples
--------------
# Extract embeddings only (reusable for supervised + leiden steps)
python run_eva.py extract \\
    --data-dir      /path/to/IMMUcan \\
    --eva-dir       /path/to/eva/project \\
    --output-dir    /path/to/output \\
    --embedding-mode bbox \\
    --device        cuda:1

# Supervised Random Forest (loads precomputed embeddings if available)
python run_eva.py supervised \\
    --data-dir      /path/to/IMMUcan \\
    --eva-dir       /path/to/eva/project \\
    --output-dir    /path/to/output \\
    --embedding-mode bbox \\
    --n-estimators  200 \\
    --n-jobs        -1

# Leiden + greedy F1 (loads precomputed embeddings if available)
python run_eva.py leiden \\
    --data-dir          /path/to/IMMUcan \\
    --eva-dir           /path/to/eva/project \\
    --output-dir        /path/to/output \\
    --embedding-mode    bbox \\
    --leiden-resolution 2.0

# Run all three steps in sequence
python run_eva.py all \\
    --data-dir      /path/to/IMMUcan \\
    --eva-dir       /path/to/eva/project \\
    --output-dir    /path/to/output \\
    --embedding-mode bbox \\
    --device        cuda:1 \\
    --n-jobs        -1

# For Julia
python run_eva.py all \\
    --data-dir   /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan/ \\
    --eva-dir    /home/juliaoesterle/eva/project \\
    --output-dir /home/juliaoesterle/results/eva_bbox \\
    --embedding-mode bbox \\
    --device cuda:1

Embedding modes (--embedding-mode)
-----------------------------------
bbox  Adaptive bounding box from segmentation mask -> zero-pad to 224x224.
      Each cell gets exactly its own pixels. CLS token as feature.
      Most IMMUcan cells: mean bbox 9.5x10.6px, mean area 78.7px²

tile  224x224 overlapping tissue tiles -> full spatial token map ->
      mean spatial tokens over cell mask pixels as feature.
      ~9 Eva forward passes per image (~18 min total for 179 images)

Output structure
----------------
{output_dir}/
├── embeddings/
│   ├── cache/                             <- per-image cache (crash-safe)
│   ├── all_cell_features.npy              <- (N_cells x 768)
│   ├── all_cell_metadata.csv              <- image_id, cell_id, label
│   └── leiden_clusters_res{X}.csv         <- Leiden + UMAP (leiden mode)
├── EVA_supervised_{embedding}/
│   └── level3/
│       ├── predictions_0.csv ... predictions_4.csv
│       └── fold_times.txt
└── EVA_leiden_{embedding}/
    └── level3/
        ├── predictions_0.csv ... predictions_4.csv
        └── fold_times.txt

Notes
-----
- Eva requires 224x224 input (hard-coded assertion in patch_embed layer)
- .contiguous() is required after .permute() before Eva forward pass
- CLS token (index 0) used as cell feature in bbox mode
- Mean spatial tokens used as cell feature in tile mode
- Per-image cache enables crash-safe resumption of long runs
- Embeddings are reused across supervised/leiden runs automatically
"""

# Standard library
import os
import sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)
import time
import argparse
import warnings
from pathlib import Path

#  CUDA env vars
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.3')
os.environ.setdefault('LD_LIBRARY_PATH',
                      '/usr/local/cuda-12.3/lib64:' +
                      os.environ.get('LD_LIBRARY_PATH', ''))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

#  Third-party
import numpy as np
import pandas as pd
import tifffile
import torch
from tqdm import tqdm

#  Shared benchmark utilities
from utils.utils_foundational_models import (
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

#  Eva model utilities 
# OmegaConf and Eva.utils are imported inside load_eva_model() because
# Eva must first be added to sys.path using the runtime --eva-dir argument.
# Importing them here would raise ModuleNotFoundError before --eva-dir is parsed.

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


def get_clean_markers(exclude):
    exclude_set = set(exclude)
    clean_idx   = [i for i, m in enumerate(ALL_BIOMARKERS) if m not in exclude_set]
    clean_names = [m for m in ALL_BIOMARKERS if m not in exclude_set]
    print(f"Using {len(clean_names)} markers (excluded: {exclude_set})")
    return clean_idx, clean_names


#  Eva model loading 

def load_eva_model(eva_dir, device):
    """
    Load Eva model from HuggingFace checkpoint.
    OmegaConf and Eva.utils are imported here because Eva must first be added
    to sys.path using the runtime --eva-dir argument.
    Requires HuggingFace authentication: huggingface-cli auth login
    Model: yandrewl/Eva (public gated repo — access request required)
    """
    sys.path.insert(0, str(eva_dir))
    from omegaconf import OmegaConf       # requires Eva venv
    from Eva.utils import load_from_hf    # requires eva_dir on sys.path

    os.chdir(str(eva_dir))
    conf  = OmegaConf.load(Path(eva_dir) / 'config.yaml')
    model = load_from_hf(repo_id='yandrewl/Eva', conf=conf, device=device)
    model.eval()
    print(f"Eva loaded on: {next(model.parameters()).device}")
    return model


#  Eva-specific patch extraction 

def get_bbox_patch(img, ys, xs, patch_size=224):
    """
    Adaptive bounding box: crop exact cell bbox from segmentation mask
    pixels -> symmetrically zero-pad to patch_size x patch_size.

    Unlike fixed crops, captures the full cell regardless of shape/size.
    Mean IMMUcan cell: 9.5x10.6px bbox, 78.7px² area.

    Returns: (patch, bbox_h, bbox_w)
    """
    y_min, y_max   = int(ys.min()), int(ys.max())
    x_min, x_max   = int(xs.min()), int(xs.max())
    crop           = img[y_min:y_max+1, x_min:x_max+1, :]
    crop_h, crop_w = crop.shape[:2]

    # Safety: centre-crop if bbox exceeds patch_size (I imagine this to be a rare edge case)
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


def extract_features_batch(patches, model, biomarkers, device):
    """
    Bbox mode: batched Eva forward pass on 224x224 patches.
    Returns CLS token (index 0) as cell feature: (N, 768).

    CRITICAL: .contiguous() after .permute() is required.
    Eva's patch_embed uses .view() which needs contiguous memory.
    Without this: RuntimeError: view size is not compatible.
    """
    batch       = np.stack(patches)
    batch_t     = torch.from_numpy(batch).to(device)
    batch_input = batch_t.permute(0, 3, 1, 2).contiguous()  # (N, C, 224, 224)
    batch_bms   = [biomarkers] * len(patches)
    with torch.no_grad():
        token_out, _ = model.model.forward_encoder(batch_input, batch_bms)
    return token_out[:, 0, 0, :].contiguous().cpu().numpy()  # (N, 768)


def extract_token_map(img, model, biomarkers, device, patch_size):
    """
    Tile mode (v3): overlapping 224x224 patches -> HxWx768 token map.
    Tokens are averaged where patches overlap.
    Per-cell feature = mean of token vectors over cell mask pixels.
    ~9 forward passes per 600x600 image.
    """
    H, W, C  = img.shape
    feat_dim = 768
    stride   = patch_size // 2

    feat_accum  = np.zeros((H, W, feat_dim), dtype=np.float32)
    count_accum = np.zeros((H, W), dtype=np.float32)

    y_starts = list(range(0, max(1, H - patch_size + 1), stride))
    x_starts = list(range(0, max(1, W - patch_size + 1), stride))
    if y_starts[-1] + patch_size < H: y_starts.append(H - patch_size)
    if x_starts[-1] + patch_size < W: x_starts.append(W - patch_size)

    for y0 in y_starts:
        for x0 in x_starts:
            tile = img[y0:y0+patch_size, x0:x0+patch_size, :]
            if tile.shape[0] < patch_size or tile.shape[1] < patch_size:
                pad_tile = np.zeros((patch_size, patch_size, C), dtype=np.float32)
                pad_tile[:tile.shape[0], :tile.shape[1], :] = tile
                tile = pad_tile

            t = torch.from_numpy(tile).unsqueeze(0).to(device)
            with torch.no_grad():
                token_out, _ = model.model.forward_encoder(
                    t.permute(0, 3, 1, 2).contiguous(), [biomarkers]
                )

            # Drop CLS, mean over marker channels -> (784, 768) -> (28, 28, 768)
            spatial    = token_out[0, :, 1:, :].mean(dim=0).cpu().numpy()
            grid_size  = int(np.sqrt(spatial.shape[0]))
            token_grid = spatial.reshape(grid_size, grid_size, feat_dim)
            token_size = patch_size // grid_size

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


#  Feature extraction 

def extract_features(args, model, all_images, clean_indices, biomarkers, label_map):
    """
    Extract Eva cell embeddings for all images.
    Supports both 'bbox' (v7) and 'tile' (v3) embedding modes.
    Results are cached per image for crash-safe resumption.
    """
    data_dir  = Path(args.data_dir)
    cache_dir = Path(args.output_dir) / 'embeddings' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    NPZ_DIR    = data_dir / 'CellTypes' / 'data' / 'images'
    MASK_DIR   = data_dir / 'segmentation'
    LABELS_DIR = data_dir / 'CellTypes' / 'cells2labels'

    img_feature_store = {}

    print(f"\nExtracting Eva features | mode={args.embedding_mode} | "
          f"patch_size={args.patch_size} | device={args.device}")
    if args.embedding_mode == 'bbox':
        print(f"  batch_size={args.batch_size} cells per forward pass")
    print("=" * 60)

    for img_name in tqdm(all_images, desc=f"Eva ({args.embedding_mode})"):
        cache_feat = cache_dir / f"{img_name}_feat.npy"
        cache_meta = cache_dir / f"{img_name}_meta.csv"

        # Load from cache
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

        # Load and normalise image
        img = np.load(npz_file)['data'].astype(np.float32)
        img = img[clean_indices]
        img = img / (img.max() + 1e-8)
        img = img.transpose(1, 2, 0)  # (H, W, C)

        # Load mask and ground truth labels
        mask = tifffile.imread(mask_file)
        with open(label_file) as f:
            cell_labels = [int(line.strip()) for line in f.readlines()]

        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids > 0]

        if args.embedding_mode == 'bbox':
            # Adaptive bounding box
            patches_buf, labels_buf, cell_ids_buf = [], [], []
            for cell_id in cell_ids:
                label  = get_label(int(cell_id), cell_labels, label_map)
                ys, xs = np.where(mask == cell_id)
                patch, _, _ = get_bbox_patch(img, ys, xs, args.patch_size)
                patches_buf.append(patch)
                labels_buf.append(label)
                cell_ids_buf.append(int(cell_id))

            all_feats = []
            for i in range(0, len(patches_buf), args.batch_size):
                batch_feats = extract_features_batch(
                    patches_buf[i:i + args.batch_size],
                    model, biomarkers, args.device
                )
                all_feats.extend(batch_feats)
                torch.cuda.empty_cache()

        else:
            # Tile-based spatial token map
            token_map = extract_token_map(
                img, model, biomarkers, args.device, args.patch_size
            )
            all_feats, labels_buf, cell_ids_buf = [], [], []
            for cell_id in cell_ids:
                label  = get_label(int(cell_id), cell_labels, label_map)
                ys, xs = np.where(mask == cell_id)
                feat   = token_map[ys, xs, :].mean(axis=0)
                all_feats.append(feat)
                labels_buf.append(label)
                cell_ids_buf.append(int(cell_id))

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

    print(f"\nExtraction complete: {len(all_feats_arr):,} cells")
    print(pd.Series(all_labels_list).value_counts().to_string())

    return img_feature_store, all_feats_arr, metadata_all


#  Argument Parser 

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_eva.py',
        description='Eva Foundation Model — IMMUcan Benchmark CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(dest='mode', required=True,
                                help='Pipeline mode')
    for m in ['extract', 'supervised', 'leiden', 'all']:
        p = sub.add_parser(
            m,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            help={
                'extract':    'Extract Eva embeddings only (reusable downstream)',
                'supervised': 'Supervised Random Forest + 5-fold CV',
                'leiden':     'Leiden clustering + greedy F1',
                'all':        'extract -> supervised -> leiden in sequence',
            }[m]
        )

        # Eva-specific arguments
        p.add_argument('--eva-dir', required=True,
                       help='Eva project directory (contains config.yaml, Eva/)')
        p.add_argument('--embedding-mode', choices=['bbox', 'tile'], default='bbox',
                       help='bbox: adaptive bounding box (v7) | '
                            'tile: spatial token map (v3)')
        p.add_argument('--patch-size', type=int, default=224,
                       help='Eva input size — do not change (hard-coded at 224)')
        p.add_argument('--batch-size', type=int, default=16,
                       help='[bbox only] Cells per Eva forward pass')
        p.add_argument('--exclude-markers', nargs='+',
                       default=DEFAULT_EXCLUDE,
                       help='Markers to exclude (no GenePT equivalent)')
        p.add_argument('--device', default='cuda:1',
                       help='PyTorch device (cuda:0, cuda:1, cpu)')

        # Shared arguments from utils_benchmark
        add_shared_args(p)

    return parser


#  Main 

def main():
    parser = build_parser()
    args   = parser.parse_args()

    total_start = time.time()
    print(f"\n{'='*60}")
    print(f"run_eva.py | mode={args.mode} | embedding={args.embedding_mode}")
    print(f"{'='*60}")

    # Setup
    clean_indices, biomarkers = get_clean_markers(args.exclude_markers)
    label_map                 = load_label_map(args.data_dir)
    folds, all_images         = load_folds(args.data_dir, args.n_folds)

    # Output folder names for spCellEval structure
    supervised_name = f"EVA_supervised_{args.embedding_mode}"
    leiden_name     = f"EVA_leiden_{args.embedding_mode}"

    # Try loading precomputed embeddings — skip extraction if available
    all_feats_arr, metadata_all = load_embeddings(args.output_dir)

    if all_feats_arr is not None and args.mode in ('supervised', 'leiden'):
        img_feature_store = rebuild_img_feature_store(args.output_dir, all_images)
    else:
        # Load Eva model and extract features
        model = load_eva_model(args.eva_dir, args.device)
        img_feature_store, all_feats_arr, metadata_all = extract_features(
            args, model, all_images, clean_indices, biomarkers, label_map
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
    print(f"run_eva.py complete | total: {total_time:.1f}s "
          f"({total_time/3600:.2f}h)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()