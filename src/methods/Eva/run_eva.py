#!/usr/bin/env python3
"""
run_eva.py — Eva Foundation Model: config-driven unified benchmark CLI
======================================================================

Supports the IMMUcan and cHL_2_MIBI datasets via --dataset + --config.
All dataset-specific constants (markers, exclusions, name maps, paths) live in
src/methods/configs/eva.json. The script stays dataset-agnostic.
Eva-specific: model loading + feature extraction only. Supervised RF, Leiden
clustering, greedy F1, output saving are handled by utils_benchmark.py (shared).

Usage
-----
# IMMUcan — full pipeline
python run_eva.py all \
    --dataset    immucan \
    --config     src/methods/configs/eva.json \
    --eva-dir    /home/juliaoesterle/eva/project \
    --output-dir /home/juliaoesterle/results/eva_immucan/bbox \
    --device     cuda:1

# cHL — extract embeddings only
python run_eva.py extract \
    --dataset    chl \
    --config     src/methods/configs/eva.json \
    --eva-dir    /home/juliaoesterle/eva/project \
    --output-dir /home/juliaoesterle/results/eva_chl/bbox \
    --device     cuda:1 --batch-size 4
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

# CONFIG LOADING

def load_config(config_path: str, dataset: str) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    if dataset not in cfg["datasets"]:
        raise ValueError(f"Dataset '{dataset}' not in {config_path}. "
                         f"Available: {list(cfg['datasets'])}")
    merged = dict(cfg.get("defaults", {}))
    merged.update(cfg["datasets"][dataset])
    merged["dataset"] = dataset
    return merged

# EVA MODEL LOADING

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

# MARKER HELPERS  (config-driven)

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
    clean_names   = [c for c in channels if c not in exclude_set]
    biomarkers    = [name_map.get(m, m) for m in clean_names]

    print(f"cHL markers: {len(channels)} total, "
          f"{len(clean_indices)} kept (excluded: {sorted(exclude_set)})")
    for orig, eva in zip(clean_names, biomarkers):
        if orig != eva:
            print(f"  Mapped: {orig!r} -> {eva!r}")
    return clean_indices, biomarkers

# DATA LOADING  (config-driven paths)

def load_immucan_data(cfg, data_root):
    ct_dir   = data_root / "CellTypes"
    img_dir  = data_root / cfg["image_dir"]
    c2l_dir  = data_root / cfg["label_dir"]
    seg_dir  = data_root / cfg["segmentation_dir"]

    lbl_csv   = pd.read_csv(data_root / cfg["labels_csv"])
    int2type  = dict(zip(lbl_csv['label'], lbl_csv['phenotype']))
    label_map = dict(zip(lbl_csv['phenotype'], lbl_csv['label']))

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
                continue
            records.append({
                'cell_id':   cell_id,
                'sample_id': img_id,
                'cell_type': int2type.get(lbl_int, f'unknown_{lbl_int}'),
                'label_int': lbl_int,
            })

    meta_df = pd.DataFrame(records)
    print(f"Loaded IMMUcan metadata: {len(meta_df):,} cells")
    print(f"Found {len(img_paths)} images, {len(seg_paths)} segmentation masks")

    with open(data_root / cfg["folds_json"]) as f:
        raw_folds = json.load(f)
    fold_indices = sorted({int(k.split('_')[1]) for k in raw_folds if k.startswith('fold_')})
    folds = [{'train': raw_folds.get(f'fold_{i}_train_set', []),
              'test':  raw_folds.get(f'fold_{i}_test_set',  [])} for i in fold_indices]
    print(f"Loaded {len(folds)} folds (image-level splits)")

    return meta_df, folds, img_paths, seg_paths, label_map


def load_chl_data(cfg, data_root):
    proc_dir = data_root / "quantification" / "processed"

    meta_df = pd.read_csv(data_root / cfg["quant_csv"])
    meta_df['sample_id'] = (meta_df['sample_id'].astype(str)
                                                 .str.replace('.csv', '', regex=False))
    exclude_labels = set(cfg.get("exclude_labels", []))
    n_before = len(meta_df)
    meta_df  = meta_df[~meta_df['cell_type'].isin(exclude_labels)].copy()
    print(f"Loaded cHL metadata: {len(meta_df):,} cells "
          f"(dropped {n_before - len(meta_df):,} undefined)")

    with open(data_root / cfg["folds_json"]) as f:
        fold_data = json.load(f)
    folds = fold_data['folds']
    print(f"Loaded {len(folds)} folds (cell-level splits)")

    img_dir   = data_root / cfg["image_dir"]
    img_paths = {}
    for tif in sorted(img_dir.glob('*_stacked.ome.tif')):
        if tif.name.startswith('.'):
            continue
        img_paths[tif.name.replace('_stacked.ome.tif', '')] = tif
    print(f"Found {len(img_paths)} images")

    seg_dir   = data_root / cfg["segmentation_dir"]
    seg_paths = {}
    for img_id in img_paths:
        seg_files = [p for p in (seg_dir / img_id).rglob('segmentationMap.tif')
                     if not p.name.startswith('.')]
        if seg_files:
            seg_paths[img_id] = seg_files[0]
    print(f"Found {len(seg_paths)} segmentation maps")

    all_labels = sorted(meta_df['cell_type'].dropna().unique())
    label_map  = {lbl: i for i, lbl in enumerate(all_labels)}
    return meta_df, folds, img_paths, seg_paths, label_map


# IMAGE / SEGMENTATION READING

def read_immucan_image(img_path, clean_indices):
    arr = np.load(img_path, allow_pickle=True)
    img = arr['data'].astype(np.float32)
    return img[clean_indices]


def read_chl_image(img_path, clean_indices):
    img = tifffile.imread(str(img_path)).astype(np.float32)
    return img[clean_indices]


def read_segmentation(seg_path):
    return tifffile.imread(str(seg_path)).astype(np.int32)

# FEATURE EXTRACTION

def get_bbox_patch(img, ys, xs, patch_size=224):
    C, H, W = img.shape
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    crop = img[:, y_min:y_max + 1, x_min:x_max + 1]
    pad  = np.zeros((C, patch_size, patch_size), dtype=np.float32)
    ch   = min(crop.shape[1], patch_size)
    cw   = min(crop.shape[2], patch_size)
    pad[:, :ch, :cw] = crop[:, :ch, :cw]
    return pad


def extract_features_batch(patches, model, biomarkers, device):
    batch_input = torch.tensor(np.stack(patches), dtype=torch.float32).to(device)
    batch_bms   = [biomarkers] * len(patches)
    with torch.no_grad():
        token_out, _ = model.model.forward_encoder(batch_input, batch_bms)
        cls_tokens = token_out[:, 0, :].cpu().numpy()
    return cls_tokens


def extract_token_map(img, model, biomarkers, device, patch_size, stride=None):
    """
    Tile mode (v3) — the authors' full-tile / spatial-token-map approach, as
    opposed to v7's per-cell adaptive-bbox crop. Overlapping patch_size x
    patch_size tiles cover the whole image -> one H x W x 768 spatial token
    map. Tokens are averaged where tiles overlap. Per-cell feature = mean of
    token vectors over that cell's mask pixels (done by the caller).
    img: (H, W, C) float32.
    """
    H, W, C  = img.shape
    feat_dim = 768
    stride   = stride if stride else patch_size // 2

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

            # Drop CLS, mean over marker channels -> (n_patches, 768) -> (grid, grid, 768)
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


def extract_features(args, model, all_images, clean_indices, biomarkers,
                     label_map, meta_df, img_paths, seg_paths, read_img_fn):
    img_feature_store = {}
    all_feats_list    = []
    meta_records      = []
    t0_total = time.time()

    for img_id in sorted(all_images):
        if img_id not in img_paths or img_id not in seg_paths:
            print(f"  [skip] {img_id}: missing image or segmentation")
            continue

        t0 = time.time()
        print(f"\n-- {img_id} --")
        img = read_img_fn(img_paths[img_id], clean_indices)
        seg = read_segmentation(seg_paths[img_id])

        meta_img = meta_df[meta_df['sample_id'] == img_id].copy()
        if meta_img.empty:
            print(f"  [skip] {img_id}: no metadata rows")
            continue

        feats_list, labels_list, cell_ids_list = [], [], []
        patches_buf, cids_buf, lbls_buf = [], [], []

        for _, row in meta_img.iterrows():
            cid   = row['cell_id']
            ltype = row['cell_type']
            label_int = label_map.get(ltype, -1)
            ys, xs = np.where(seg == cid)
            if len(ys) == 0:
                continue
            patches_buf.append(get_bbox_patch(img, ys, xs, patch_size=args.patch_size))
            cids_buf.append(cid)
            lbls_buf.append(label_int)
            if len(patches_buf) >= args.batch_size:
                feats = extract_features_batch(patches_buf, model, biomarkers, args.device)
                feats_list.extend(feats); labels_list.extend(lbls_buf); cell_ids_list.extend(cids_buf)
                patches_buf, cids_buf, lbls_buf = [], [], []

        if patches_buf:
            feats = extract_features_batch(patches_buf, model, biomarkers, args.device)
            feats_list.extend(feats); labels_list.extend(lbls_buf); cell_ids_list.extend(cids_buf)

        if not feats_list:
            print(f"  [skip] {img_id}: no cells extracted")
            continue

        feats_arr  = np.array(feats_list,  dtype=np.float32)
        labels_arr = np.array(labels_list, dtype=np.int32)
        cids_arr   = np.array(cell_ids_list)

        img_feature_store[img_id] = {'feats': feats_arr, 'labels': labels_arr, 'cell_ids': cids_arr}
        all_feats_list.append(feats_arr)

        for cid, ltype_int, ltype_str in zip(
                cids_arr, labels_arr,
                [meta_img.loc[meta_img['cell_id'] == c, 'cell_type'].values[0] for c in cids_arr]):
            meta_records.append({'cell_id': cid, 'sample_id': img_id,
                                 'cell_type': ltype_str, 'label_int': int(ltype_int)})

        print(f"  {len(feats_arr)} cells | {time.time() - t0:.1f}s")

    all_feats_arr = np.concatenate(all_feats_list, axis=0)
    metadata_all  = pd.DataFrame(meta_records)
    print(f"\nExtracted {len(all_feats_arr):,} cells total in {time.time() - t0_total:.1f}s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'embeddings.npy', all_feats_arr)
    metadata_all.to_csv(out_dir / 'metadata.csv', index=False)
    import pickle
    with open(out_dir / 'img_feature_store.pkl', 'wb') as f:
        pickle.dump(img_feature_store, f)
    print(f"Saved embeddings -> {out_dir}")

    return img_feature_store, all_feats_arr, metadata_all


def extract_features_tile(args, model, all_images, clean_indices, biomarkers,
                          label_map, meta_df, img_paths, seg_paths, read_img_fn):
    """
    v3 full-tile pipeline: one token map per image (extract_token_map), then
    per-cell feature = mean token vector over the cell's segmentation-mask
    pixels. No per-cell forward pass, unlike the bbox path.
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
        print(f"\n-- {img_id} --")
        img_chw = read_img_fn(img_paths[img_id], clean_indices)    # (C, H, W)
        img_hwc = np.ascontiguousarray(img_chw.transpose(1, 2, 0)) # (H, W, C)
        seg     = read_segmentation(seg_paths[img_id])

        meta_img = meta_df[meta_df['sample_id'] == img_id].copy()
        if meta_img.empty:
            print(f"  [skip] {img_id}: no metadata rows")
            continue

        token_map = extract_token_map(img_hwc, model, biomarkers, args.device,
                                      patch_size=args.tile_size, stride=args.tile_stride)

        feats_list, labels_list, cell_ids_list = [], [], []
        for _, row in meta_img.iterrows():
            cid   = row['cell_id']
            ltype = row['cell_type']
            label_int  = label_map.get(ltype, -1)
            cell_pixels = seg == cid
            if not cell_pixels.any():
                continue
            feats_list.append(token_map[cell_pixels].mean(axis=0))
            labels_list.append(label_int)
            cell_ids_list.append(cid)

        if not feats_list:
            print(f"  [skip] {img_id}: no cells extracted")
            continue

        feats_arr  = np.array(feats_list,  dtype=np.float32)
        labels_arr = np.array(labels_list, dtype=np.int32)
        cids_arr   = np.array(cell_ids_list)

        img_feature_store[img_id] = {'feats': feats_arr, 'labels': labels_arr, 'cell_ids': cids_arr}
        all_feats_list.append(feats_arr)

        for cid, ltype_int, ltype_str in zip(
                cids_arr, labels_arr,
                [meta_img.loc[meta_img['cell_id'] == c, 'cell_type'].values[0] for c in cids_arr]):
            meta_records.append({'cell_id': cid, 'sample_id': img_id,
                                 'cell_type': ltype_str, 'label_int': int(ltype_int)})

        print(f"  {len(feats_arr)} cells | {time.time() - t0:.1f}s")

    all_feats_arr = np.concatenate(all_feats_list, axis=0)
    metadata_all  = pd.DataFrame(meta_records)
    print(f"\nExtracted {len(all_feats_arr):,} cells total in {time.time() - t0_total:.1f}s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'embeddings.npy', all_feats_arr)
    metadata_all.to_csv(out_dir / 'metadata.csv', index=False)
    import pickle
    with open(out_dir / 'img_feature_store.pkl', 'wb') as f:
        pickle.dump(img_feature_store, f)
    print(f"Saved embeddings -> {out_dir}")

    return img_feature_store, all_feats_arr, metadata_all


def load_embeddings(output_dir):
    out_dir = Path(output_dir)
    emb_path  = out_dir / 'embeddings.npy'
    meta_path = out_dir / 'metadata.csv'
    if emb_path.exists() and meta_path.exists():
        print(f"Loading precomputed embeddings from {out_dir}")
        return np.load(emb_path), pd.read_csv(meta_path)
    return None, None


def rebuild_img_feature_store(output_dir, all_images):
    import pickle
    pkl_path = Path(output_dir) / 'img_feature_store.pkl'
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    raise FileNotFoundError(f"img_feature_store.pkl not found in {output_dir}. Run 'extract' first.")

# ARGUMENT PARSER

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_eva.py',
        description='Eva config-driven pipeline — IMMUcan + cHL_2_MIBI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ['extract', 'supervised', 'leiden', 'all']:
        p = sub.add_parser(mode, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('--dataset', choices=['immucan', 'chl'], required=True)
        p.add_argument('--config',  required=True, help='Path to eva.json')
        p.add_argument('--eva-dir', required=True, help='Eva project root')
        p.add_argument('--output-dir', required=True)
        # optional overrides (default None -> config)
        p.add_argument('--device',     default=None)
        p.add_argument('--batch-size', type=int, default=None)
        p.add_argument('--embedding-mode', choices=['bbox', 'tile'], default=None,
                       help='tile (v3, authors full-tile, default) or bbox (v7, adaptive bbox)')
        p.add_argument('--patch-size', type=int, default=None,
                       help='[bbox mode] per-cell crop size')
        p.add_argument('--tile-size',   type=int, default=None,
                       help='[tile mode] sliding-tile size over the whole image')
        p.add_argument('--tile-stride', type=int, default=None,
                       help='[tile mode] sliding-tile stride (default: tile-size // 2)')
        p.add_argument('--exclude-markers', nargs='+', default=None)
        p.add_argument('--n-folds',    type=int, default=None)
        p.add_argument('--fold',       type=int, default=None)
        p.add_argument('--n-estimators', type=int, default=None)
        p.add_argument('--n-jobs',     type=int, default=None)
        p.add_argument('--leiden-resolution',  type=float, default=None)
        p.add_argument('--leiden-n-neighbors', type=int,   default=None)
        p.add_argument('--leiden-subsample',   type=int,   default=None)
        p.add_argument('--spceleval-dir', default=None)
    return parser


# MAIN

def main():
    parser = build_parser()
    args   = parser.parse_args()
    total_start = time.time()

    cfg = load_config(args.config, args.dataset)

    # Fill any unset CLI args from config
    def fill(attr, key, default=None):
        if getattr(args, attr) is None:
            setattr(args, attr, cfg.get(key, default))
    fill('device', 'device', 'cuda:1')
    fill('batch_size', 'batch_size', 16)
    fill('embedding_mode', 'embedding_mode', 'tile')
    fill('patch_size', 'patch_size', 224)
    fill('tile_size', 'tile_size', 224)
    fill('tile_stride', 'tile_stride', 112)
    fill('n_folds', 'n_folds', 5)
    fill('n_estimators', 'n_estimators', 200)
    fill('n_jobs', 'n_jobs', -1)
    fill('leiden_resolution', 'leiden_resolution', 2.0)
    fill('leiden_n_neighbors', 'leiden_n_neighbors', 15)
    fill('leiden_subsample', 'leiden_subsample', 50000)
    if args.exclude_markers is None:
        args.exclude_markers = cfg["exclude_markers"]

    print(f"\n{'='*60}")
    print(f"run_eva.py | mode={args.mode} | dataset={args.dataset} | embedding_mode={args.embedding_mode}")
    print(f"{'='*60}")

    if args.spceleval_dir:
        sys.path.insert(0, str(Path(args.spceleval_dir) / 'src' / 'methods' / 'utils'))
    else:
        # fall back to the utils/ dir two levels up (src/methods/utils) when
        # this script lives at src/methods/Eva/run_eva.py
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
    from utils_foundational_models import run_supervised, run_leiden

    data_root = Path(cfg["data_root"])

    if args.dataset == 'immucan':
        # allow CLI override of exclude markers to flow into marker resolution
        cfg_local = dict(cfg); cfg_local["exclude_markers"] = args.exclude_markers
        meta_df, folds, img_paths, seg_paths, label_map = load_immucan_data(cfg, data_root)
        clean_indices, biomarkers = get_immucan_markers(cfg_local, data_root)
        read_img_fn = read_immucan_image
    else:
        cfg_local = dict(cfg); cfg_local["exclude_markers"] = args.exclude_markers
        meta_df, folds, img_paths, seg_paths, label_map = load_chl_data(cfg, data_root)
        clean_indices, biomarkers = get_chl_markers(cfg_local)
        read_img_fn = read_chl_image

    all_images = sorted(img_paths.keys())
    if args.fold is not None:
        folds = [folds[args.fold]]
        print(f"Running single fold: {args.fold}")

    supervised_name = f"EVA_supervised_{args.dataset}_{args.embedding_mode}"
    leiden_name     = f"EVA_leiden_{args.dataset}_{args.embedding_mode}"

    all_feats_arr, metadata_all = load_embeddings(args.output_dir)
    if all_feats_arr is not None and args.mode in ('supervised', 'leiden'):
        img_feature_store = rebuild_img_feature_store(args.output_dir, all_images)
    else:
        model = load_eva_model(args.eva_dir, args.device)
        extract_fn = extract_features_tile if args.embedding_mode == 'tile' else extract_features
        img_feature_store, all_feats_arr, metadata_all = extract_fn(
            args, model, all_images, clean_indices, biomarkers,
            label_map, meta_df, img_paths, seg_paths, read_img_fn,
        )

    if args.mode in ('supervised', 'all'):
        run_supervised(args, folds, img_feature_store, all_feats_arr, metadata_all,
                       method_name=supervised_name)
    if args.mode in ('leiden', 'all'):
        run_leiden(args, folds, img_feature_store, all_feats_arr, metadata_all,
                   method_name=leiden_name)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"run_eva.py complete | dataset={args.dataset} | total: {total_time:.1f}s")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()