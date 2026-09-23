#!/usr/bin/env python3
"""
run_virtues.py — VirTues config-driven unified pipeline: IMMUcan + cHL_2_MIBI
=============================================================================

All dataset-specific constants (channel lists, UniProt maps, exclusions, paths)
live in src/methods/configs/virtues.json and are selected with --dataset.

(Note to Julia) IMPORTANT: run inside nix develop + stellar micromamba env on beauty:
    cd /home/juliaoesterle/VirTues
    nix develop
    micromamba activate stellar

Usage
-----
# IMMUcan — all 5 folds, crop64 (best on IMMUcan):
python run_virtues.py all \
    --dataset     immucan \
    --config      src/methods/configs/virtues.json \
    --output-dir  /home/juliaoesterle/results/virtues/immucan/crop64 \
    --crop-size   64 --stride 21 --device cuda:0

# cHL_2_MIBI — all 5 folds:
python run_virtues.py all \
    --dataset     chl \
    --config      src/methods/configs/virtues.json \
    --output-dir  /home/juliaoesterle/results/virtues/chl/crop64 \
    --crop-size   64 --stride 21 --device cuda:0
"""

import os
import sys
import time
import json
import argparse
import warnings
from pathlib import Path

os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.3')
os.environ.setdefault('LD_LIBRARY_PATH',
                      '/usr/local/cuda-12.3/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from utils_foundational_models import (
    load_label_map, load_folds, get_label,
    save_embeddings, load_embeddings, rebuild_img_feature_store,
    run_supervised, run_leiden, add_shared_args,
)

warnings.filterwarnings('ignore')


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


# MARKER INDEX BUILDER 

def get_clean_markers(all_channels, exclude, uniprot_map, marker_embedding_dir):
    exclude_set   = set(exclude)
    available_ids = {f.removesuffix('.pt')
                     for f in os.listdir(marker_embedding_dir) if f.endswith('.pt')}
    sorted_ids = sorted(available_ids)
    embed_idx  = {pid: i for i, pid in enumerate(sorted_ids)}

    clean_indices, clean_names, channel_mask, marker_indices, no_embedding = [], [], [], [], []
    for i, m in enumerate(all_channels):
        if m in exclude_set:
            continue
        clean_indices.append(i)
        clean_names.append(m)
        uniprot = uniprot_map.get(m)
        if uniprot and uniprot in available_ids:
            channel_mask.append(True)
            marker_indices.append(embed_idx[uniprot])
        else:
            channel_mask.append(False)
            no_embedding.append(m)
            marker_indices.append(0)

    print(f"  Markers: {len(clean_names)} used | {sum(channel_mask)} with embeddings | "
          f"{len(no_embedding)} skipped")
    if no_embedding:
        print(f"  No embedding: {no_embedding}")

    channel_mask_t   = torch.tensor(channel_mask, dtype=torch.bool)
    marker_indices_t = torch.tensor(
        [marker_indices[i] for i, has in enumerate(channel_mask) if has], dtype=torch.long)
    return clean_indices, clean_names, channel_mask_t, marker_indices_t, no_embedding


# MODEL LOADING 

def load_virtues_model(virtues_dir, device):
    sys.path.insert(0, str(virtues_dir))
    os.chdir(str(virtues_dir))

    from omegaconf import OmegaConf
    from safetensors import safe_open
    from virtues.modules.multiplex_virtues import MultiplexVirtues
    from virtues.utils.utils import load_marker_embeddings

    conf = OmegaConf.load(Path(virtues_dir) / 'configs' / 'base_config.yaml')

    weights_path = None
    for candidate in [
        Path(virtues_dir) / 'assets' / 'checkpoints' / 'model.safetensors',
        Path(virtues_dir) / 'weights' / 'model.safetensors',
    ]:
        if candidate.exists():
            weights_path = candidate
            break
    if weights_path is None:
        raise FileNotFoundError("model.safetensors not found.")

    marker_embedding_dir = str(
        Path(virtues_dir) / 'assets' / 'example_dataset' / 'marker_embeddings')
    marker_embeddings = load_marker_embeddings(marker_embedding_dir)
    print(f"Loaded {len(marker_embeddings)} marker embeddings")

    model = MultiplexVirtues(
        use_default_config=False, custom_config=None,
        prior_bias_embeddings=marker_embeddings,
        prior_bias_embedding_type='esm',
        prior_bias_embedding_fusion_type='add',
        patch_size=conf.model.patch_size,
        model_dim=conf.model.model_dim,
        feedforward_dim=conf.model.feedforward_dim,
        encoder_pattern=conf.model.encoder_pattern,
        num_encoder_heads=conf.model.num_encoder_heads,
        decoder_pattern=conf.model.decoder_pattern,
        num_decoder_heads=conf.model.num_decoder_heads,
        num_hidden_layers=conf.model.num_decoder_hidden_layers,
        positional_embedding_type=conf.model.positional_embedding_type,
        dropout=conf.model.dropout,
        group_layers=conf.model.group_layers,
        norm_after_encoder_decoder=conf.model.norm_after_encoder_decoder,
        verbose=False,
    )

    weights = {}
    with safe_open(str(weights_path), framework='pt', device='cpu') as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()

    embedding_dim = conf.model.model_dim
    print(f"VirTues loaded on: {device} | embedding_dim={embedding_dim}")
    return model, embedding_dim, conf, marker_embedding_dir

# PREPROCESSING  

def preprocess_image(img_raw, channel_mask):
    from torchvision.transforms import GaussianBlur
    img_t = torch.from_numpy(img_raw).float()
    img_t = img_t[channel_mask]
    C, H, W = img_t.shape
    quantiles = torch.quantile(img_t.reshape(C, -1), 0.99, dim=1)
    img_t = torch.clamp(img_t, min=torch.zeros_like(quantiles[:, None, None]),
                        max=quantiles[:, None, None])
    img_t = torch.log1p(img_t)
    img_t = GaussianBlur(kernel_size=3, sigma=1.0)(img_t)
    means = img_t.reshape(C, -1).mean(dim=1)
    stds  = img_t.reshape(C, -1).std(dim=1)
    img_t = (img_t - means[:, None, None]) / (stds[:, None, None] + 1e-9)
    return img_t, means, stds


def run_compute_cell_tokens(img_processed, seg_np, marker_indices, model, conf, args, pad_size):
    from virtues.utils.cell_tokens import compute_cell_tokens
    mask_t      = torch.from_numpy(seg_np.astype(np.int32))
    img_padded  = F.pad(img_processed, (pad_size, pad_size, pad_size, pad_size),
                        mode='constant', value=0)
    mask_padded = F.pad(mask_t, (pad_size, pad_size, pad_size, pad_size),
                        mode='constant', value=0)
    crop_size = args.crop_size if args.crop_size else conf.data.crop_size
    cell_ids_out, cell_tokens, _, _ = compute_cell_tokens(
        model=model, img=img_padded, channel=marker_indices,
        segmentation_mask=mask_padded.numpy(), device=args.device,
        crop_size=crop_size, patch_size=conf.model.patch_size,
        stride=args.stride, chunk_size=args.chunk_size,
    )
    return cell_ids_out, cell_tokens.numpy()

# IMMUcan PIPELINE  (config-driven paths)

def extract_features_immucan(args, cfg, model, embedding_dim, conf,
                              all_images, clean_indices, clean_names,
                              channel_mask, marker_indices, label_map):
    data_root = Path(cfg["data_root"])
    cache_dir = Path(args.output_dir) / 'embeddings' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    NPZ_DIR    = data_root / cfg["image_dir"]
    MASK_DIR   = data_root / cfg["segmentation_dir"]
    LABELS_DIR = data_root / cfg["label_dir"]

    img_feature_store = {}
    crop_size = args.crop_size if args.crop_size else conf.data.crop_size
    print(f"\n[IMMUcan] crop_size={crop_size} | stride={args.stride} | "
          f"pad={args.pad_size} | emb_dim={embedding_dim}")

    for img_name in tqdm(all_images, desc="VirTues IMMUcan"):
        cache_feat = cache_dir / f"{img_name}_feat.npy"
        cache_meta = cache_dir / f"{img_name}_meta.csv"
        if cache_feat.exists() and cache_meta.exists():
            feats = np.load(cache_feat)
            meta  = pd.read_csv(cache_meta)
            img_feature_store[img_name] = (feats, meta['label'].tolist(), meta['cell_id'].tolist())
            continue

        npz_file   = NPZ_DIR    / f"{img_name}.npz"
        mask_file  = MASK_DIR   / f"{img_name}.tiff"
        label_file = LABELS_DIR / f"{img_name}.txt"
        if not all(f.exists() for f in [npz_file, mask_file, label_file]):
            print(f"  Skipping {img_name} — missing files")
            continue

        img_raw = np.load(npz_file)['data'].astype(np.float32)[clean_indices]
        img_processed, _, _ = preprocess_image(img_raw, channel_mask)

        seg_np = tifffile.imread(mask_file)
        with open(label_file) as f:
            cell_labels = [int(l.strip()) for l in f.readlines()]

        cell_ids_out, feats_list = run_compute_cell_tokens(
            img_processed, seg_np, marker_indices, model, conf, args, args.pad_size)

        labels_list   = [get_label(int(cid), cell_labels, label_map) for cid in cell_ids_out]
        cell_ids_list = [int(cid) for cid in cell_ids_out]
        feats_arr     = np.array(feats_list)

        np.save(cache_feat, feats_arr)
        pd.DataFrame({'image_id': [img_name]*len(cell_ids_list),
                      'cell_id': cell_ids_list, 'label': labels_list}
                     ).to_csv(cache_meta, index=False)
        img_feature_store[img_name] = (feats_arr, labels_list, cell_ids_list)
        torch.cuda.empty_cache()

    all_f, all_l, all_i, all_n = [], [], [], []
    for img_name, (feats, labels, cell_ids) in img_feature_store.items():
        all_f.extend(feats); all_l.extend(labels)
        all_i.extend(cell_ids); all_n.extend([img_name]*len(cell_ids))

    all_feats_arr = np.array(all_f)
    metadata_all  = pd.DataFrame({'image_id': all_n, 'cell_id': all_i, 'label': all_l})
    save_embeddings(args.output_dir, all_feats_arr, metadata_all)
    print(f"\nExtraction complete: {len(all_feats_arr):,} cells")
    return img_feature_store, all_feats_arr, metadata_all


# cHL PIPELINE  (config-driven paths)

def load_chl_metadata(cfg):
    data_root = Path(cfg["data_root"])
    meta_df = pd.read_csv(data_root / cfg["quant_csv"])
    meta_df["sample_id"] = meta_df["sample_id"].astype(str).str.replace(".csv", "", regex=False)
    print(f"cHL: {len(meta_df):,} cells, {meta_df['cell_type'].nunique()} types")

    with open(data_root / cfg["folds_json"]) as f:
        folds = json.load(f)["folds"]
    print(f"cHL: {len(folds)} cell-level folds")

    img_dir   = data_root / cfg["image_dir"]
    img_paths = {p.name.replace("_stacked.ome.tif", ""): p
                 for p in sorted(img_dir.glob("*_stacked.ome.tif"))}
    print(f"cHL: {len(img_paths)} images")

    seg_paths = {}
    for img_id in img_paths:
        d = data_root / cfg["segmentation_dir"] / img_id
        if not d.exists():
            continue
        hits = [f for f in d.rglob("segmentationMap.tif") if not f.name.startswith(".")]
        if hits:
            seg_paths[img_id] = hits[0]
    print(f"cHL: {len(seg_paths)} segmentations found")
    return meta_df, folds, img_paths, seg_paths


def extract_features_chl(args, cfg, model, embedding_dim, conf,
                          clean_indices, channel_mask, marker_indices):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    meta_df, folds, img_paths, seg_paths = load_chl_metadata(cfg)
    exclude_labels = set(cfg.get("exclude_labels", []))

    cache_dir = Path(args.output_dir) / 'embeddings' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_df["_key"] = meta_df["sample_id"] + "_" + meta_df["cell_id"].astype(str)
    key_to_row = dict(zip(meta_df["_key"], meta_df.index))

    crop_size = args.crop_size if args.crop_size else conf.data.crop_size
    print(f"\n[cHL] crop_size={crop_size} | stride={args.stride} | "
          f"pad={args.pad_size} | emb_dim={embedding_dim}")

    feat_store = {}
    for img_id, img_path in tqdm(sorted(img_paths.items()), desc="VirTues cHL", total=len(img_paths)):
        if img_id not in seg_paths:
            print(f"  [skip] {img_id} — no segmentation")
            continue

        cache_feat = cache_dir / f"{img_id}_feat.npy"
        cache_keys = cache_dir / f"{img_id}_keys.npy"
        if cache_feat.exists() and cache_keys.exists():
            feats    = np.load(cache_feat)
            row_keys = np.load(cache_keys)
            for i, rk in enumerate(row_keys):
                feat_store[int(rk)] = feats[i]
            continue

        img_raw = tifffile.imread(img_path).astype(np.float32)
        seg_np  = tifffile.imread(seg_paths[img_id]).astype(np.int32)
        img_sel = img_raw[clean_indices]
        img_processed, _, _ = preprocess_image(img_sel, channel_mask)

        cell_ids_out, feats_list = run_compute_cell_tokens(
            img_processed, seg_np, marker_indices, model, conf, args, args.pad_size)

        row_keys_buf, feats_buf = [], []
        for i, cid in enumerate(cell_ids_out):
            key = f"{img_id}_{int(cid)}"
            if key in key_to_row:
                row_idx = key_to_row[key]
                feat_store[row_idx] = feats_list[i]
                row_keys_buf.append(row_idx)
                feats_buf.append(feats_list[i])

        if feats_buf:
            np.save(cache_feat, np.array(feats_buf))
            np.save(cache_keys, np.array(row_keys_buf))
        torch.cuda.empty_cache()

    print(f"Features extracted: {len(feat_store):,} cells")

    pred_dir = Path(args.output_dir) / "VIRTUES_supervised" / "level3"
    pred_dir.mkdir(parents=True, exist_ok=True)

    fold_range = [args.fold] if args.fold is not None else range(len(folds))
    fold_times, train_times, predict_times = [], [], []
    for fold_idx in fold_range:
        fold_start = time.time()
        fold      = folds[fold_idx]
        train_idx = [int(i) for i in fold["train"]]
        test_idx  = [int(i) for i in fold["test"]]

        X_tr, y_tr = [], []
        for idx in train_idx:
            if idx not in feat_store:
                continue
            lbl = meta_df.loc[idx, "cell_type"]
            if lbl in exclude_labels:
                continue
            X_tr.append(feat_store[idx]); y_tr.append(lbl)

        X_te, y_te, keys, sids = [], [], [], []
        for idx in test_idx:
            if idx not in feat_store:
                continue
            lbl = meta_df.loc[idx, "cell_type"]
            if lbl in exclude_labels:
                continue
            X_te.append(feat_store[idx]); y_te.append(lbl); keys.append(idx)
            sids.append(meta_df.loc[idx, "sample_id"])

        print(f"\n[Fold {fold_idx}] Train={len(X_tr):,} Test={len(X_te):,}")

        train_start = time.time()
        clf = RandomForestClassifier(n_estimators=500, n_jobs=args.n_jobs,
                                     random_state=42, class_weight="balanced")
        clf.fit(np.array(X_tr), y_tr)
        train_time = time.time() - train_start

        predict_start = time.time()
        y_pred  = clf.predict(np.array(X_te))
        y_proba = clf.predict_proba(np.array(X_te))
        predict_time = time.time() - predict_start

        acc = accuracy_score(y_te, y_pred)
        mf1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        wf1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
        print(f"  Acc={acc:.4f}  MacroF1={mf1:.4f}  WF1={wf1:.4f}")
        print(classification_report(y_te, y_pred, zero_division=0))

        # Same predictions_{fold}.csv schema as the IMMUcan path (image_id,
        # cell_id, fold, true_phenotype, predicted_phenotype, confidence) --
        # was cell_id/true_label/pred_label only before, inconsistent with
        # every other method's output here.
        pd.DataFrame({
            "image_id":            sids,
            "cell_id":             keys,
            "fold":                fold_idx,
            "true_phenotype":      y_te,
            "predicted_phenotype": y_pred,
            "confidence":          y_proba.max(axis=1),
        }).to_csv(pred_dir / f"predictions_{fold_idx}.csv", index=False)
        print(f"  -> {pred_dir}/predictions_{fold_idx}.csv")

        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        train_times.append(train_time)
        predict_times.append(predict_time)
        print(f"  Time: {fold_time:.1f}s (train={train_time:.1f}s, predict={predict_time:.1f}s)")

    with open(pred_dir / "fold_times.txt", "w") as f:
        for i, t, tr, pr in zip(fold_range, fold_times, train_times, predict_times):
            f.write(f"fold_{i}: {t:.2f}s (train={tr:.2f}s, predict={pr:.2f}s)\n")
        f.write(f"total: {sum(fold_times):.2f}s\n")
        f.write(f"total_train: {sum(train_times):.2f}s\n")
        f.write(f"total_predict: {sum(predict_times):.2f}s\n")
        f.write(f"mean_fold: {np.mean(fold_times):.2f}s\n")
    print(f"  -> {pred_dir}/fold_times.txt")


# ARGUMENT PARSER

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_virtues.py',
        description='VirTues config-driven pipeline — IMMUcan + cHL_2_MIBI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)
    for m in ['extract', 'supervised', 'leiden', 'all']:
        p = sub.add_parser(m, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('--dataset',     choices=['immucan', 'chl'], required=True)
        p.add_argument('--config',      required=True, help='Path to virtues.json')
        p.add_argument('--virtues-dir', default=None)
        p.add_argument('--device',      default=None)
        p.add_argument('--crop-size',   type=int, default=None)
        p.add_argument('--stride',      type=int, default=None)
        p.add_argument('--chunk-size',  type=int, default=None)
        p.add_argument('--pad-size',    type=int, default=None)
        p.add_argument('--fold',        type=int, default=None)
        p.add_argument('--exclude-markers', nargs='+', default=None)
        add_shared_args(p)   # --data-dir, --output-dir, --n-folds, --n-estimators etc.
    return parser


# MAIN

def main():
    parser = build_parser()
    args   = parser.parse_args()
    total_start = time.time()

    cfg = load_config(args.config, args.dataset)

    # Fill unset CLI args from config defaults
    def fill(attr, key, default=None):
        if getattr(args, attr, None) is None:
            setattr(args, attr, cfg.get(key, default))
    fill('virtues_dir', 'virtues_dir', '/home/juliaoesterle/VirTues')
    fill('device', 'device', 'cuda:1')
    fill('crop_size', 'crop_size', None)
    fill('stride', 'stride', 42)
    fill('chunk_size', 'chunk_size', 32)
    fill('pad_size', 'pad_size', 120)
    if getattr(args, 'n_folds', None) is None:
        args.n_folds = cfg.get('n_folds', 5)
    if args.exclude_markers is None:
        args.exclude_markers = cfg["exclude_markers"]

    print(f"\n{'='*60}")
    print(f"run_virtues.py | dataset={args.dataset} | mode={args.mode}")
    print(f"{'='*60}")

    virtues_dir = Path(args.virtues_dir)
    emb_dir     = str(virtues_dir / 'assets' / 'example_dataset' / 'marker_embeddings')

    model, embedding_dim, conf, _ = load_virtues_model(virtues_dir, args.device)

    if args.dataset == 'immucan':
        clean_indices, clean_names, channel_mask, marker_indices, _ = get_clean_markers(
            cfg["channels"], args.exclude_markers, cfg["uniprot_map"], emb_dir)

        data_root         = cfg["data_root"]
        label_map         = load_label_map(data_root)
        folds, all_images = load_folds(data_root, args.n_folds)

        all_feats_arr, metadata_all = load_embeddings(args.output_dir)
        if all_feats_arr is not None and args.mode in ('supervised', 'leiden'):
            img_feature_store = rebuild_img_feature_store(args.output_dir, all_images)
        else:
            img_feature_store, all_feats_arr, metadata_all = extract_features_immucan(
                args, cfg, model, embedding_dim, conf, all_images,
                clean_indices, clean_names, channel_mask, marker_indices, label_map)

        if args.mode in ('supervised', 'all'):
            run_supervised(args, folds, img_feature_store, all_feats_arr, metadata_all,
                           method_name='VIRTUES_supervised')
        if args.mode in ('leiden', 'all'):
            run_leiden(args, folds, img_feature_store, all_feats_arr, metadata_all,
                       method_name='VIRTUES_leiden')

    else:  # chl
        clean_indices, clean_names, channel_mask, marker_indices, _ = get_clean_markers(
            cfg["channels"], args.exclude_markers, cfg["uniprot_map"], emb_dir)
        extract_features_chl(args, cfg, model, embedding_dim, conf,
                             clean_indices, channel_mask, marker_indices)

    print(f"\nTotal: {(time.time()-total_start)/60:.1f} min")


if __name__ == '__main__':
    main()