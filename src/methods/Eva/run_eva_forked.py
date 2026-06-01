#!/usr/bin/env python3
"""
run_eva.py — Eva Foundation Model: Unified Benchmark CLI
=========================================================
Author: Julia Oesterle
Date:   June 2026

Supports both the IMMUcan and cHL_2_MIBI datasets via --dataset flag.
Eva-specific: model loading + feature extraction only.
Supervised RF, Leiden clustering, greedy F1, output saving
are all handled by utils_benchmark.py (shared with run_kronos.py etc.)

Usage examples
--------------
# IMMUcan — extract embeddings only
python run_eva.py extract \\
    --dataset    immucan \\
    --data-dir   /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan/ \\
    --eva-dir    /home/juliaoesterle/eva/project \\
    --output-dir /scratch/juliaoesterle/results/eva_immucan/bbox \\
    --device     cuda:1

# cHL — extract embeddings only
python run_eva.py extract \\
    --dataset    chl \\
    --data-dir   /home/juliaoesterle/data/phenotyping_benchmark/cHL_2_MIBI/ \\
    --eva-dir    /home/juliaoesterle/eva/project \\
    --output-dir /scratch/juliaoesterle/results/eva_chl/bbox \\
    --device     cuda:1 \\
    --batch-size 4

# IMMUcan — full pipeline
python run_eva.py all \\
    --dataset    immucan \\
    --data-dir   /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan/ \\
    --eva-dir    /home/juliaoesterle/eva/project \\
    --output-dir /scratch/juliaoesterle/results/eva_immucan/bbox \\
    --device     cuda:1 \\
    --n-jobs     -1

# cHL — full pipeline
python run_eva.py all \\
    --dataset    chl \\
    --data-dir   /home/juliaoesterle/data/phenotyping_benchmark/cHL_2_MIBI/ \\
    --eva-dir    /home/juliaoesterle/eva/project \\
    --output-dir /scratch/juliaoesterle/results/eva_chl/bbox \\
    --device     cuda:1 \\
    --batch-size 4 \\
    --n-jobs     -1
"""

# stdlib 
import argparse
import json
import os
import sys
import time
from pathlib import Path

# third-party 
import numpy as np
import pandas as pd
import tifffile
import torch


# IMMUcan — markers to exclude (structural / non-biological)
IMMUCAN_DEFAULT_EXCLUDE = ['DNA1', 'DNA2', 'Histone H3']

# cHL_2_MIBI — channel names in multistack order (from markers.txt)
CHL_CHANNELS = [
    'CD4', 'CD20', 'PD-1', 'FoxP3', 'CD14', 'CD161', 'Tbet', 'CD25',
    'CD68', 'CD3', 'CD57', 'Pax-5', 'GATA3', 'TCRgd', 'RORgT', 'CD45RO',
    'Na-K ATPase', 'CD56', 'Lag3', 'CD45RA', 'B2-Microglobulin', 'TIM3',
    'SLP-76', 'Tox', 'Histone H3', 'CD30', 'CXCR5', 'Ki-67', 'Granzyme B',
    'CD8', 'PD-L1', 'CD45', 'HLA1', 'CD153 (CD30L)', 'pSLP-76', 'HLA-DR',
    'CD16', 'dsDNA', 'CD86', 'CD11c', 'CD15', 'CD28', 'CD123',
    'anti-H2AX (pS139)', 'CD163', 'IL-10',
]

# cHL — markers to exclude from Eva input (structural / non-protein)
CHL_DEFAULT_EXCLUDE = ['Histone H3', 'dsDNA', 'anti-H2AX (pS139)', 'pSLP-76', 'SLP-76']

# cHL — map cHL marker names → Eva/GenePT lookup names where they differ
CHL_TO_EVA_NAME = {
    'FoxP3':            'FOXP3',
    'Pax-5':            'PAX5',
    'TCRgd':            'TRGV',
    'RORgT':            'RORC',
    'Na-K ATPase':      'ATP1A1',
    'Lag3':             'LAG3',
    'B2-Microglobulin': 'B2M',
    'TIM3':             'HAVCR2',
    'Tox':              'TOX',
    'Granzyme B':       'GZMB',
    'PD-L1':            'CD274',
    'CD153 (CD30L)':    'TNFSF8',
    'HLA-DR':           'HLA-DRA',
    'CD86':             'CD86',
    'Ki-67':            'MKI67',
    'CXCR5':            'CXCR5',
    'GATA3':            'GATA3',
    'Tbet':             'TBX21',
    'HLA1':             'HLA-A',
    'IL-10':            'IL10',
}

# cHL — cell-type labels to exclude from training/evaluation
CHL_EXCLUDE_LABELS = {'undefined', 'unedfined'}   # note dataset typo


# Eva Model Loading

def load_eva_model(eva_dir, device):
    """
    Load Eva foundation model.
    Mirrors the authors' approach: omegaconf config + load_from_hf.
    eva_dir should point to the project root that contains config.yaml and the Eva package.
    """
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


# Marker Helpers (per dataset   )

def get_immucan_markers(data_dir, exclude_markers):
    """
    Read IMMUcan marker list from CellTypes/channels.txt and return
    clean channel indices + Eva/GenePT-compatible biomarker names.

    Returns:
        clean_indices : list[int]   channel positions to keep
        biomarkers    : list[str]   marker names for Eva forward_encoder
    """
    channels_path = Path(data_dir) / 'CellTypes' / 'channels.txt'
    with open(channels_path) as f:
        all_markers = [line.strip() for line in f if line.strip()]

    exclude_set   = set(exclude_markers)
    clean_indices = [i for i, m in enumerate(all_markers) if m not in exclude_set]
    biomarkers    = [all_markers[i] for i in clean_indices]

    print(f"IMMUcan markers: {len(all_markers)} total, "
          f"{len(clean_indices)} kept (excluded: {sorted(exclude_set)})")
    return clean_indices, biomarkers


def get_chl_markers(exclude_markers):
    """
    Build channel indices and Eva-compatible biomarker names for cHL_2_MIBI.

    Returns:
        clean_indices : list[int]   channel positions to keep from (46, H, W)
        biomarkers    : list[str]   marker names for Eva forward_encoder
    """
    exclude_set   = set(exclude_markers)
    clean_indices = [i for i, c in enumerate(CHL_CHANNELS) if c not in exclude_set]
    clean_names   = [c for c in CHL_CHANNELS if c not in exclude_set]
    biomarkers    = [CHL_TO_EVA_NAME.get(m, m) for m in clean_names]

    print(f"cHL markers: {len(CHL_CHANNELS)} total, "
          f"{len(clean_indices)} kept (excluded: {sorted(exclude_set)})")
    for orig, eva in zip(clean_names, biomarkers):
        if orig != eva:
            print(f"  Mapped: {orig!r} → {eva!r}")
    return clean_indices, biomarkers


# Data Loading <— per-dataset functions to read metadata, folds, image paths, segmentation paths

def load_immucan_data(data_dir):
    """
    Load IMMUcan metadata, folds, image paths, and segmentation paths.

    Actual directory layout:
        <data_dir>/
            CellTypes/
                data/images/        ← {image_id}.npz  key='data'  (40,600,600) uint16
                cells2labels/       ← {image_id}.txt  one integer label per line
                                       line N = cell N (1-indexed), -1 = unlabelled
                folds.json          ← keys fold_0_train_set, fold_0_test_set, …
                labels_cell_type.csv← phenotype,label  (integer → string)
                channels.txt        ← marker names in channel order
            segmentation/           ← {image_id}.tiff  (segmentation mask, flat)

    Returns:
        meta_df   : DataFrame  columns [cell_id, sample_id, cell_type, label_int]
                    cell_id is 1-indexed (matches segmentation mask values)
        folds     : list of 5 dicts with keys 'train', 'test'
                    each value is a list of image_ids (image-level splits)
        img_paths : dict {image_id: Path}   ← .npz paths
        seg_paths : dict {image_id: Path}   ← .tiff segmentation paths
        label_map : dict {str → int}        ← phenotype string → integer
    """
    data_dir = Path(data_dir)
    ct_dir   = data_dir / 'CellTypes'
    img_dir  = ct_dir / 'data' / 'images'
    c2l_dir  = ct_dir / 'cells2labels'
    seg_dir  = data_dir / 'segmentation'

    # Integer → phenotype string mapping from labels_cell_type.csv
    lbl_csv  = pd.read_csv(ct_dir / 'labels_cell_type.csv')
    # columns: phenotype, label  (label is the integer)
    int2type = dict(zip(lbl_csv['label'], lbl_csv['phenotype']))
    # string → int label_map for downstream use
    label_map = dict(zip(lbl_csv['phenotype'], lbl_csv['label']))

    # Build image paths and per-image metadata from cells2labels .txt files
    img_paths = {}
    seg_paths = {}
    records   = []

    for npz_path in sorted(img_dir.glob('*.npz')):
        if npz_path.name.startswith('.'):
            continue
        img_id = npz_path.stem
        img_paths[img_id] = npz_path

        # Segmentation mask: segmentation/{image_id}.tiff
        seg_path = seg_dir / f'{img_id}.tiff'
        if seg_path.exists():
            seg_paths[img_id] = seg_path

        # Cell labels: cells2labels/{image_id}.txt
        # Each line = label integer for cell (line_number) starting at cell 1
        c2l_path = c2l_dir / f'{img_id}.txt'
        if not c2l_path.exists():
            continue
        with open(c2l_path) as f:
            label_ints = [int(line.strip()) for line in f if line.strip()]
        for cell_id, lbl_int in enumerate(label_ints, start=1):
            if lbl_int == -1:
                continue   # unlabelled cell
            cell_type = int2type.get(lbl_int, f'unknown_{lbl_int}')
            records.append({
                'cell_id':   cell_id,
                'sample_id': img_id,
                'cell_type': cell_type,
                'label_int': lbl_int,
            })

    meta_df = pd.DataFrame(records)
    print(f"Loaded IMMUcan metadata: {len(meta_df):,} cells")
    print(f"Cell types: {meta_df['cell_type'].value_counts().to_dict()}")
    print(f"Found {len(img_paths)} images, {len(seg_paths)} segmentation masks")

    # Folds: image-level splits
    # folds.json keys: fold_0_train_set, fold_0_test_set, fold_1_train_set, …
    with open(ct_dir / 'folds.json') as f:
        raw_folds = json.load(f)

    # Determine number of folds from keys
    fold_indices = sorted({int(k.split('_')[1])
                            for k in raw_folds if k.startswith('fold_')})
    folds = []
    for i in fold_indices:
        folds.append({
            'train': raw_folds.get(f'fold_{i}_train_set', []),
            'test':  raw_folds.get(f'fold_{i}_test_set',  []),
        })
    print(f"Loaded {len(folds)} folds (image-level splits)")

    return meta_df, folds, img_paths, seg_paths, label_map


def load_chl_data(data_dir):
    """
    Load cHL_2_MIBI metadata, folds, image paths, and segmentation paths.

    Directory layout expected:
        <data_dir>/
            markers.txt
            raw_images/
                multistack_tiffs/       ← {id}_stacked.ome.tif
                single_channel_tiffs/   (not used here)
            segmentation/
                <image_id>/
                    …/segmentationMap.tif
            quantification/
                processed/
                    cHL_2_MIBI_quantification.csv
                    kfolds_StratifiedGroupKFold_level3/
                        fold_indices.json

    Returns:
        meta_df   : DataFrame  columns [cell_id, sample_id, cell_type, x, y]
        folds     : list[dict] keys 'train', 'test', (optionally 'validation')
        img_paths : dict {sample_id: Path}   ← multistack tiff paths
        seg_paths : dict {sample_id: Path}   ← segmentationMap.tif paths
        label_map : dict {str → int}
    """
    data_dir = Path(data_dir)
    proc_dir = data_dir / 'quantification' / 'processed'

    # Metadata
    meta_df = pd.read_csv(proc_dir / 'cHL_2_MIBI_quantification.csv')
    meta_df['sample_id'] = (meta_df['sample_id'].astype(str)
                                                 .str.replace('.csv', '', regex=False))
    # Drop excluded cell types
    n_before = len(meta_df)
    meta_df  = meta_df[~meta_df['cell_type'].isin(CHL_EXCLUDE_LABELS)].copy()
    print(f"Loaded cHL metadata: {len(meta_df):,} cells "
          f"(dropped {n_before - len(meta_df):,} undefined)")
    print(f"Cell types: {meta_df['cell_type'].value_counts().to_dict()}")

    # Folds
    fold_path = (proc_dir / 'kfolds_StratifiedGroupKFold_level3'
                           / 'fold_indices.json')
    with open(fold_path) as f:
        fold_data = json.load(f)
    folds = fold_data['folds']
    print(f"Loaded {len(folds)} folds (cell-level splits)")

    # Image paths
    img_dir   = data_dir / 'raw_images' / 'multistack_tiffs'
    img_paths = {}
    for tif in sorted(img_dir.glob('*_stacked.ome.tif')):
        if tif.name.startswith('.'):
            continue
        img_id = tif.name.replace('_stacked.ome.tif', '')
        img_paths[img_id] = tif
    print(f"Found {len(img_paths)} images: {sorted(img_paths.keys())}")

    # Segmentation paths
    seg_dir   = data_dir / 'segmentation'
    seg_paths = {}
    for img_id in img_paths:
        seg_files = [
            p for p in (seg_dir / img_id).rglob('segmentationMap.tif')
            if not p.name.startswith('.')
        ]
        if seg_files:
            seg_paths[img_id] = seg_files[0]
    print(f"Found {len(seg_paths)} segmentation maps")

    # Label map
    all_labels = sorted(meta_df['cell_type'].dropna().unique())
    label_map  = {lbl: i for i, lbl in enumerate(all_labels)}

    return meta_df, folds, img_paths, seg_paths, label_map


# Image/Segmentation Reading

def read_immucan_image(img_path, clean_indices):
    """
    Load IMMUcan .npz → (C_clean, H, W) float32.
    """
    arr = np.load(img_path, allow_pickle=True)
    img = arr['data'].astype(np.float32)  # (C, H, W)
    return img[clean_indices]


def read_chl_image(img_path, clean_indices):
    """
    Load cHL_2_MIBI multistack OME-TIFF → (C_clean, H, W) float32.
    """
    img = tifffile.imread(str(img_path)).astype(np.float32)  # (46, H, W)
    return img[clean_indices]


def read_segmentation(seg_path):
    """Load segmentation map and return as int32."""
    return tifffile.imread(str(seg_path)).astype(np.int32)


# Feature Extraction

def get_bbox_patch(img, ys, xs, patch_size=224):
    """
    Extract adaptive bounding-box crop of cell pixels, zero-padded to
    (C, patch_size, patch_size).

    img : (C, H, W) float32
    ys, xs : 1-D arrays of pixel row / col indices belonging to this cell
    """
    C, H, W = img.shape
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    crop = img[:, y_min:y_max + 1, x_min:x_max + 1]  # (C, h, w)
    pad  = np.zeros((C, patch_size, patch_size), dtype=np.float32)
    ch   = min(crop.shape[1], patch_size)
    cw   = min(crop.shape[2], patch_size)
    pad[:, :ch, :cw] = crop[:, :ch, :cw]
    return pad


def extract_features_batch(patches, model, biomarkers, device):
    """
    Run Eva forward_encoder on a batch of patches.

    patches    : list of (C, 224, 224) float32 arrays
    biomarkers : list of marker-name strings (same length as C)

    Returns: (N, 768) float32 numpy array — CLS token embeddings
    """
    batch_input = torch.tensor(
        np.stack(patches), dtype=torch.float32
    ).to(device)                              # (N, C, 224, 224)

    batch_bms = [biomarkers] * len(patches)

    with torch.no_grad():
        token_out, _ = model.model.forward_encoder(batch_input, batch_bms)
        # token_out: (N, 1+num_patches, 768) — index 0 is CLS
        cls_tokens = token_out[:, 0, :].cpu().numpy()  # (N, 768)

    return cls_tokens


def extract_features(args, model, all_images, clean_indices, biomarkers,
                     label_map, meta_df, img_paths, seg_paths, read_img_fn):
    """
    Loop over all images, extract Eva CLS-token embeddings for every cell.

    Returns:
        img_feature_store : dict {img_id: {'feats': (N_img, 768),
                                           'labels': (N_img,),
                                           'cell_ids': (N_img,)}}
        all_feats_arr     : (N_total, 768) float32
        metadata_all      : DataFrame [cell_id, sample_id, cell_type, label_int]
    """
    img_feature_store = {}
    all_feats_list    = []
    meta_records      = []

    t0_total = time.time()

    for img_id in sorted(all_images):
        if img_id not in img_paths or img_id not in seg_paths:
            print(f"  [skip] {img_id}: missing image or segmentation")
            continue

        t0 = time.time()
        print(f"\n── {img_id} ─────────────────────────────────────────")

        img = read_img_fn(img_paths[img_id], clean_indices)    # (C, H, W)
        seg = read_segmentation(seg_paths[img_id])              # (H, W) int32

        meta_img = meta_df[meta_df['sample_id'] == img_id].copy()
        if meta_img.empty:
            print(f"  [skip] {img_id}: no metadata rows")
            continue

        feats_list    = []
        labels_list   = []
        cell_ids_list = []
        patches_buf   = []
        cids_buf      = []
        lbls_buf      = []

        for _, row in meta_img.iterrows():
            cid   = row['cell_id']
            ltype = row['cell_type']
            label_int = label_map.get(ltype, -1)

            ys, xs = np.where(seg == cid)
            if len(ys) == 0:
                continue

            patch = get_bbox_patch(img, ys, xs, patch_size=224)
            patches_buf.append(patch)
            cids_buf.append(cid)
            lbls_buf.append(label_int)

            if len(patches_buf) >= args.batch_size:
                feats = extract_features_batch(
                    patches_buf, model, biomarkers, args.device)
                feats_list.extend(feats)
                labels_list.extend(lbls_buf)
                cell_ids_list.extend(cids_buf)
                patches_buf, cids_buf, lbls_buf = [], [], []

        # flush remainder
        if patches_buf:
            feats = extract_features_batch(
                patches_buf, model, biomarkers, args.device)
            feats_list.extend(feats)
            labels_list.extend(lbls_buf)
            cell_ids_list.extend(cids_buf)

        if not feats_list:
            print(f"  [skip] {img_id}: no cells extracted")
            continue

        feats_arr  = np.array(feats_list,    dtype=np.float32)
        labels_arr = np.array(labels_list,   dtype=np.int32)
        cids_arr   = np.array(cell_ids_list)

        img_feature_store[img_id] = {
            'feats':    feats_arr,
            'labels':   labels_arr,
            'cell_ids': cids_arr,
        }
        all_feats_list.append(feats_arr)

        for cid, ltype_int, ltype_str in zip(
                cids_arr, labels_arr,
                [meta_img.loc[meta_img['cell_id'] == c, 'cell_type'].values[0]
                 for c in cids_arr]):
            meta_records.append({
                'cell_id':   cid,
                'sample_id': img_id,
                'cell_type': ltype_str,
                'label_int': int(ltype_int),
            })

        elapsed = time.time() - t0
        print(f"  {len(feats_arr)} cells | {elapsed:.1f}s")

    all_feats_arr = np.concatenate(all_feats_list, axis=0)
    metadata_all  = pd.DataFrame(meta_records)
    print(f"\nExtracted {len(all_feats_arr):,} cells total "
          f"in {time.time() - t0_total:.1f}s")

    # Save embeddings
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'embeddings.npy',    all_feats_arr)
    metadata_all.to_csv(out_dir / 'metadata.csv', index=False)
    import pickle
    with open(out_dir / 'img_feature_store.pkl', 'wb') as f:
        pickle.dump(img_feature_store, f)
    print(f"Saved embeddings → {out_dir}")

    return img_feature_store, all_feats_arr, metadata_all


# Load precomputed embeddings 

def load_embeddings(output_dir):
    """Try to load previously saved embeddings. Returns (None, None) on miss."""
    out_dir = Path(output_dir)
    emb_path  = out_dir / 'embeddings.npy'
    meta_path = out_dir / 'metadata.csv'
    if emb_path.exists() and meta_path.exists():
        print(f"Loading precomputed embeddings from {out_dir}")
        return np.load(emb_path), pd.read_csv(meta_path)
    return None, None


def rebuild_img_feature_store(output_dir, all_images):
    """Reload per-image feature store from disk (saved during extract)."""
    import pickle
    pkl_path = Path(output_dir) / 'img_feature_store.pkl'
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    raise FileNotFoundError(
        f"img_feature_store.pkl not found in {output_dir}. "
        "Run 'extract' mode first.")


# Argument Parser
def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_eva.py',
        description='Eva unified pipeline — IMMUcan + cHL_2_MIBI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)

    for mode in ['extract', 'supervised', 'leiden', 'all']:
        p = sub.add_parser(mode,
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # dataset fork 
        p.add_argument('--dataset', choices=['immucan', 'chl'], required=True,
                       help='Which dataset to use')
        p.add_argument('--data-dir', required=True,
                       help='Root of dataset')
        p.add_argument('--eva-dir', required=True,
                       help='Eva project root (contains config.yaml + Eva package)')
        p.add_argument('--output-dir', required=True,
                       help='Output directory for embeddings and predictions')

        # extraction 
        p.add_argument('--device', default='cuda:1',
                       help='Torch device')
        p.add_argument('--batch-size', type=int, default=16,
                       help='Cells per Eva forward pass (reduce if OOM)')

        # marker exclusion 
        p.add_argument('--exclude-markers', nargs='+', default=None,
                       help='Markers to exclude (default: per-dataset defaults)')

        #  cross-validation 
        p.add_argument('--n-folds', type=int, default=5)
        p.add_argument('--fold',    type=int, default=None,
                       help='Single fold only (default: all folds)')

        #  supervised RF 
        p.add_argument('--n-estimators', type=int, default=200)
        p.add_argument('--n-jobs',       type=int, default=-1)

        #  Leiden 
        p.add_argument('--leiden-resolution',  type=float, default=2.0)
        p.add_argument('--leiden-n-neighbors', type=int,   default=15)
        p.add_argument('--leiden-subsample',   type=int,   default=50000)

        #  spCellEval integration 
        p.add_argument('--spceleval-dir', default=None,
                       help='spCellEval root (for greedy_f1_utils import)')

    return parser


# Main
def main():
    parser = build_parser()
    args   = parser.parse_args()
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"run_eva.py | mode={args.mode} | dataset={args.dataset}")
    print(f"{'='*60}")

    #  Add spCellEval to path (for greedy_f1_utils) 
    if args.spceleval_dir:
        sys.path.insert(0, str(Path(args.spceleval_dir) / 'src' / 'methods' / 'utils'))

    # Add utils_benchmark to path 
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here))
    from utils_benchmark import (
        add_shared_args,   # noqa: F401 — already called via build_parser above
        run_supervised,
        run_leiden,
    )

    # Resolve marker exclusion defaults 
    if args.exclude_markers is None:
        args.exclude_markers = (IMMUCAN_DEFAULT_EXCLUDE
                                if args.dataset == 'immucan'
                                else CHL_DEFAULT_EXCLUDE)

    # Load dataset 
    if args.dataset == 'immucan':
        meta_df, folds, img_paths, seg_paths, label_map = \
            load_immucan_data(args.data_dir)
        clean_indices, biomarkers = \
            get_immucan_markers(args.data_dir, args.exclude_markers)
        read_img_fn = read_immucan_image

    else:  # chl
        meta_df, folds, img_paths, seg_paths, label_map = \
            load_chl_data(args.data_dir)
        clean_indices, biomarkers = \
            get_chl_markers(args.exclude_markers)
        read_img_fn = read_chl_image

    all_images = sorted(img_paths.keys())

    # Subset folds if requested 
    if args.fold is not None:
        folds = [folds[args.fold]]
        print(f"Running single fold: {args.fold}")

    # Method name strings (for output file naming) 
    supervised_name = f"EVA_supervised_{args.dataset}"
    leiden_name     = f"EVA_leiden_{args.dataset}"

    # Try loading precomputed embeddings 
    all_feats_arr, metadata_all = load_embeddings(args.output_dir)

    if all_feats_arr is not None and args.mode in ('supervised', 'leiden'):
        img_feature_store = rebuild_img_feature_store(args.output_dir, all_images)
    else:
        # Load Eva model and extract features
        model = load_eva_model(args.eva_dir, args.device)
        img_feature_store, all_feats_arr, metadata_all = extract_features(
            args, model, all_images, clean_indices, biomarkers,
            label_map, meta_df, img_paths, seg_paths, read_img_fn,
        )

    # Downstream evaluation 
    if args.mode in ('supervised', 'all'):
        run_supervised(
            args, folds, img_feature_store,
            all_feats_arr, metadata_all,
            method_name=supervised_name,
        )

    if args.mode in ('leiden', 'all'):
        run_leiden(
            args, folds, img_feature_store,
            all_feats_arr, metadata_all,
            method_name=leiden_name,
        )

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"✓ run_eva.py complete | dataset={args.dataset} | "
          f"total: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()