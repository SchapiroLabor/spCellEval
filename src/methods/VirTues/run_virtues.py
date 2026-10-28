#!/usr/bin/env python3
"""
run_virtues.py — VirTues config-driven pipeline (dataset-agnostic)
==================================================================

Per-dataset settings (marker exclusions, UniProt map) live in
src/methods/configs/virtues.json, keyed by the dataset folder name. Data
loading is shared via benchmark_io.py; the supervised LogReg head, Leiden
clustering, greedy F1 and output saving via utils_foundational_models.py.

--checkpoint selects the pretrained VirTues weights from the bunnelab/virtues
HF repo: sp32 (default), sp31, imc14, or legacy (local model.safetensors).

Marker (ESM) embeddings are read from marker_embeddings/<dataset>/ next to this
script unless --marker-embedding-dir is given.

Usage
-----
python src/methods/VirTues/run_virtues.py all \\
    --dataset-path <data>/IMMUcan \\
    --config       src/methods/configs/virtues.json \\
    --virtues-dir  <path to VirTues checkout> \\
    --output-dir   <results>/IMMUcan/virtues_sp32
"""

import os
import sys
import time
import json
import argparse
import warnings
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / 'utils'))
from benchmark_io import load_config, read_markers, load_benchmark_data, read_image, read_segmentation
from utils_foundational_models import (
    save_embeddings, load_embeddings, rebuild_img_feature_store,
    check_fold_coverage, run_supervised, run_leiden,
)

warnings.filterwarnings('ignore', category=FutureWarning)


#  MARKERS

def get_clean_markers(all_channels, exclude, uniprot_map, marker_embedding_dir):
    exclude_set = set(exclude)
    # Sort by filename, exactly like VirTues' load_marker_embeddings: sorting the stripped
    # ids instead reorders e.g. P08575 vs P08575-5 and silently swaps marker embeddings.
    pt_files      = sorted(f for f in os.listdir(marker_embedding_dir) if f.endswith('.pt'))
    available_ids = {f.removesuffix('.pt') for f in pt_files}
    embed_idx     = {f.removesuffix('.pt'): i for i, f in enumerate(pt_files)}

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
    return clean_indices, clean_names, channel_mask_t, marker_indices_t


#  MODEL LOADING

CHECKPOINT_MAP = {
    'sp32':  'virtues-sp32/model.safetensors',   # 32 spatial-proteomics datasets, CC BY-NC 4.0
    'sp31':  'virtues-sp31/model.safetensors',   # 31 spatial-proteomics datasets, MIT
    'imc14': 'virtues-imc14/model.safetensors',  # 14 IMC datasets, CC BY-NC 4.0
}


def load_virtues_model(virtues_dir, device, checkpoint, marker_embedding_dir):
    sys.path.insert(0, str(virtues_dir))
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from virtues.modules.multiplex_virtues import MultiplexVirtues
    from virtues.utils.utils import load_marker_embeddings

    conf = OmegaConf.load(virtues_dir / 'configs' / 'base_config.yaml')

    if checkpoint == 'legacy':
        candidates = [virtues_dir / 'assets' / 'checkpoints' / 'model.safetensors',
                      virtues_dir / 'weights' / 'model.safetensors']
        weights_path = next((p for p in candidates if p.exists()), None)
        if weights_path is None:
            raise FileNotFoundError(f"legacy model.safetensors not found in {candidates}")
    else:
        from huggingface_hub import hf_hub_download
        weights_path = Path(hf_hub_download(
            repo_id='bunnelab/virtues', filename=CHECKPOINT_MAP[checkpoint],
            local_dir=str(virtues_dir / 'assets' / 'checkpoints')))

    marker_embeddings = load_marker_embeddings(str(marker_embedding_dir))
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
    print(f"VirTues loaded on: {device} | checkpoint={checkpoint}")
    return model, conf


#  PREPROCESSING

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
    return img_t


def run_compute_cell_tokens(img_processed, seg_np, marker_indices, model, conf, args):
    from virtues.utils.cell_tokens import compute_cell_tokens
    mask_t      = torch.from_numpy(seg_np.astype(np.int32))
    img_padded  = F.pad(img_processed, (args.pad_size,) * 4, mode='constant', value=0)
    mask_padded = F.pad(mask_t, (args.pad_size,) * 4, mode='constant', value=0)
    cell_ids_out, cell_tokens, _, _ = compute_cell_tokens(
        model=model, img=img_padded, channel=marker_indices,
        segmentation_mask=mask_padded.numpy(), device=args.device,
        crop_size=args.crop_size, patch_size=conf.model.patch_size,
        stride=args.stride, chunk_size=args.chunk_size,
    )
    return cell_ids_out, cell_tokens.numpy()


#  EMBEDDING CACHE

def check_cache_params(output_dir, params):
    """
    Bind the embedding cache in <output_dir>/embeddings to the extraction settings:
    write params.json for a new cache, refuse to reuse one made with other settings.
    """
    emb_dir = Path(output_dir) / 'embeddings'
    params_file = emb_dir / 'params.json'
    if params_file.exists():
        with open(params_file) as f:
            cached = json.load(f)
        diff = {k: (cached.get(k), v) for k, v in params.items() if cached.get(k) != v}
        if diff:
            raise ValueError(f"{emb_dir} was extracted with different settings "
                             f"(cached, requested): {diff}. Use another --output-dir.")
        return
    has_cache = (emb_dir / 'all_cell_features.npy').exists() or any((emb_dir / 'cache').glob('*_feat.npy'))
    if has_cache:
        raise ValueError(f"{emb_dir} contains embeddings without params.json; "
                         f"their settings are unknown. Use another --output-dir.")
    emb_dir.mkdir(parents=True, exist_ok=True)
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)


#  FEATURE EXTRACTION

def extract_features(args, model, conf, all_images, clean_indices, channel_mask,
                     marker_indices, meta_df, img_paths, seg_paths):
    """
    Runs compute_cell_tokens per image (one token per segmented cell) and keeps
    the cells present in meta_df. Per-image results are cached in the format
    rebuild_img_feature_store()/load_embeddings() expect.
    """
    cache_dir = Path(args.output_dir) / 'embeddings' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    img_feature_store = {}
    all_feats_list, meta_records = [], []
    print(f"\ncrop_size={args.crop_size} | stride={args.stride} | pad={args.pad_size}")

    for img_id in tqdm(sorted(all_images), desc="VirTues"):
        cache_feat = cache_dir / f"{img_id}_feat.npy"
        cache_meta = cache_dir / f"{img_id}_meta.csv"
        if cache_feat.exists() and cache_meta.exists():
            feats = np.load(cache_feat)
            meta  = pd.read_csv(cache_meta, dtype={'image_id': str})
            img_feature_store[img_id] = (feats, meta['label'].tolist(), meta['cell_id'].tolist())
            all_feats_list.append(feats)
            meta_records.append(meta)
            continue

        if img_id not in img_paths or img_id not in seg_paths:
            print(f"  [skip] {img_id}: missing image or segmentation")
            continue

        meta_img = meta_df[meta_df['image_id'] == img_id]
        label_lookup = dict(zip(meta_img['cell_id'], meta_img['label']))

        img_raw = read_image(img_paths[img_id], clean_indices)
        img_processed = preprocess_image(img_raw, channel_mask)
        seg_np = read_segmentation(seg_paths[img_id])

        cell_ids_out, feats_list = run_compute_cell_tokens(
            img_processed, seg_np, marker_indices, model, conf, args)

        feats, labels, cell_ids = [], [], []
        for cid, feat in zip(cell_ids_out, feats_list):
            cid = int(cid)
            if cid not in label_lookup:
                continue
            feats.append(feat)
            labels.append(label_lookup[cid])
            cell_ids.append(cid)
        torch.cuda.empty_cache()

        if not feats:
            print(f"  [skip] {img_id}: no ground-truth cells extracted")
            continue

        feats_arr = np.array(feats)
        np.save(cache_feat, feats_arr)
        meta_out = pd.DataFrame({'image_id': img_id, 'cell_id': cell_ids, 'label': labels})
        meta_out.to_csv(cache_meta, index=False)

        img_feature_store[img_id] = (feats_arr, labels, cell_ids)
        all_feats_list.append(feats_arr)
        meta_records.append(meta_out)

    all_feats_arr = np.concatenate(all_feats_list, axis=0)
    metadata_all  = pd.concat(meta_records, ignore_index=True)
    print(f"\nExtraction complete: {len(all_feats_arr):,} cells")
    print(metadata_all['label'].value_counts().to_string())

    save_embeddings(args.output_dir, all_feats_arr, metadata_all)
    return img_feature_store, all_feats_arr, metadata_all


#  ARGUMENT PARSER

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_virtues.py',
        description='VirTues Foundation Model — config-driven pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ['extract', 'supervised', 'leiden', 'all']:
        p = sub.add_parser(
            mode,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            help={
                'extract':    'Extract VirTues embeddings only (reusable downstream)',
                'supervised': 'Logistic-regression classifier head, k-fold CV',
                'leiden':     'Leiden clustering + greedy F1',
                'all':        'extract -> supervised -> leiden in sequence',
            }[mode],
        )
        p.add_argument('--dataset-path', required=True,
                       help='Dataset directory (markers.txt, quantification/, raw_images/, segmentation/)')
        p.add_argument('--config', required=True, help='Path to virtues.json')
        p.add_argument('--virtues-dir', required=True, help='VirTues repository checkout')
        p.add_argument('--output-dir', required=True)
        p.add_argument('--folds-dir', default=None,
                       help='k-fold directory, absolute or relative to --dataset-path')
        p.add_argument('--include-validation', action=argparse.BooleanOptionalAction, default=None,
                       help='Add fold_<i>_validation.csv cells to the training set')
        p.add_argument('--marker-embedding-dir', default=None,
                       help='Directory of <UniProt id>.pt ESM embeddings '
                            '(default: marker_embeddings/<dataset> next to this script)')
        p.add_argument('--spceleval-dir', default=None)
        p.add_argument('--checkpoint', choices=['sp32', 'sp31', 'imc14', 'legacy'], default=None)
        p.add_argument('--device',      default=None)
        p.add_argument('--crop-size',   type=int, default=None)
        p.add_argument('--stride',      type=int, default=None)
        p.add_argument('--chunk-size',  type=int, default=None)
        p.add_argument('--pad-size',    type=int, default=None)
        p.add_argument('--logreg-c',    type=float, default=None)
        p.add_argument('--leiden-resolution',  type=float, default=None)
        p.add_argument('--leiden-n-neighbors', type=int,   default=None)
        p.add_argument('--leiden-subsample',   type=int,   default=None)
    return parser


#  MAIN

def main():
    args = build_parser().parse_args()
    total_start = time.time()

    data_root       = Path(args.dataset_path).resolve()
    virtues_dir     = Path(args.virtues_dir).resolve()
    args.output_dir = str(Path(args.output_dir).resolve())
    if args.spceleval_dir:
        args.spceleval_dir = str(Path(args.spceleval_dir).resolve())
    args.dataset = data_root.name
    cfg = load_config(args.config, args.dataset)

    def fill(attr, key, default=None):
        if getattr(args, attr) is None:
            setattr(args, attr, cfg.get(key, default))
    fill('checkpoint', 'checkpoint', 'sp32')
    fill('device', 'device', 'cuda:0')
    fill('crop_size', 'crop_size', 64)
    fill('stride', 'stride', 21)
    fill('chunk_size', 'chunk_size', 32)
    fill('pad_size', 'pad_size', 120)
    fill('logreg_c', 'logreg_c', 1.0)
    fill('leiden_resolution', 'leiden_resolution', 2.0)
    fill('leiden_n_neighbors', 'leiden_n_neighbors', 15)
    fill('leiden_subsample', 'leiden_subsample', 50000)
    fill('folds_dir', 'folds_dir', 'quantification/processed/kfolds_StratifiedGroupKFold_level3')
    fill('include_validation', 'include_validation', False)

    emb_dir = Path(args.marker_embedding_dir or SCRIPT_DIR / 'marker_embeddings' / args.dataset).resolve()
    if not emb_dir.is_dir():
        raise FileNotFoundError(f"Marker embedding directory not found: {emb_dir}")

    print(f"\n{'='*60}")
    print(f"run_virtues.py | mode={args.mode} | dataset={args.dataset}")
    print(f"{'='*60}")

    channels = read_markers(data_root)
    clean_indices, clean_names, channel_mask, marker_indices = get_clean_markers(
        channels, cfg.get('exclude_markers', []), cfg.get('uniprot_map', {}), emb_dir)

    meta_df, folds, img_paths, seg_paths = load_benchmark_data(
        data_root, data_root / args.folds_dir, cfg.get('exclude_labels', []), args.include_validation)
    all_images = sorted(img_paths.keys())

    check_cache_params(args.output_dir, {
        'checkpoint': args.checkpoint, 'crop_size': args.crop_size, 'stride': args.stride,
        'pad_size': args.pad_size, 'markers': clean_names,
        'markers_with_embedding': [m for m, has in zip(clean_names, channel_mask.tolist()) if has],
    })

    all_feats_arr, metadata_all = load_embeddings(args.output_dir)
    if all_feats_arr is not None and args.mode in ('supervised', 'leiden'):
        img_feature_store = rebuild_img_feature_store(args.output_dir, all_images)
    else:
        model, conf = load_virtues_model(virtues_dir, args.device, args.checkpoint, emb_dir)
        if args.crop_size % conf.model.patch_size:
            raise ValueError(f"crop_size {args.crop_size} must be a multiple of the "
                             f"model patch_size {conf.model.patch_size}")
        img_feature_store, all_feats_arr, metadata_all = extract_features(
            args, model, conf, all_images, clean_indices, channel_mask,
            marker_indices, meta_df, img_paths, seg_paths,
        )

    check_fold_coverage(folds, zip(metadata_all['image_id'], metadata_all['cell_id']))

    if args.mode in ('supervised', 'all'):
        run_supervised(args, folds, img_feature_store, all_feats_arr, metadata_all,
                       method_name='VIRTUES_supervised', fold_type='cell')
    if args.mode in ('leiden', 'all'):
        run_leiden(args, folds, img_feature_store, all_feats_arr, metadata_all,
                   method_name='VIRTUES_leiden', fold_type='cell')

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"run_virtues.py complete | total: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
