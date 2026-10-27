#!/usr/bin/env python3
"""
run_virtues.py — VirTues config-driven unified pipeline (dataset-agnostic)
==========================================================================

Dataset-agnostic CLI for VirTues. All dataset-specific constants (paths,
channels, UniProt maps, exclusions, fold layout) live in
src/methods/configs/virtues.json; this script only implements model loading
and feature extraction. The supervised classifier, Leiden clustering,
greedy F1 and output saving are shared with the other foundation-model
methods via utils_foundational_models.py.

--checkpoint selects which pretrained VirTues weights to load: the bunnelab/
virtues HF repo's sp32 (32 spatial-proteomics datasets, default), sp31 (31
datasets, MIT), imc14 (14 IMC datasets), or legacy (the original unversioned
checkpoint used for the first benchmark run, for comparison). Everything
else about the pipeline is identical across checkpoints.

(Note to Julia) run inside nix develop + stellar micromamba env on beauty:
    cd /home/juliaoesterle/VirTues
    nix develop
    micromamba activate stellar

Usage
-----
python run_virtues.py all \\
    --dataset-path /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan \\
    --config       src/methods/configs/virtues.json \\
    --output-dir   /home/juliaoesterle/results/virtues/IMMUcan \\
    --device       cuda:1

python run_virtues.py all \\
    --dataset-path /home/juliaoesterle/data/phenotyping_benchmark/cHL_2_MIBI \\
    --config       src/methods/configs/virtues.json \\
    --output-dir   /home/juliaoesterle/results/virtues/cHL_2_MIBI \\
    --checkpoint   legacy \\
    --device       cuda:1
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
    save_embeddings, load_embeddings, rebuild_img_feature_store,
    run_supervised, run_leiden,
)

warnings.filterwarnings('ignore')


#  CONFIG LOADING

def _rebase_local_paths(obj, prefix="/home/juliaoesterle", env_var="SPCELLEVAL_ROOT"):
    """virtues.json hardcodes absolute paths rooted at beauty (/home/juliaoesterle).
    Set SPCELLEVAL_ROOT to re-root every such path onto another machine's
    data root (e.g. Helix's /mnt/sds-hd/sd22c003) without editing the JSON."""
    root = os.environ.get(env_var)
    if not root:
        return obj
    if isinstance(obj, str):
        return root.rstrip('/') + obj[len(prefix):] if obj.startswith(prefix) else obj
    if isinstance(obj, dict):
        return {k: _rebase_local_paths(v, prefix, env_var) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rebase_local_paths(v, prefix, env_var) for v in obj]
    return obj


def load_config(config_path: str, dataset: str) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    merged = dict(cfg.get("defaults", {}))
    merged.update(cfg["datasets"].get(dataset, {}))
    merged["dataset"] = dataset
    return _rebase_local_paths(merged)


#  MARKER HELPERS

def get_clean_markers(all_channels, exclude, uniprot_map, marker_embedding_dir):
    exclude_set = set(exclude)
    # MUST sort by filename (not the stripped id) to match load_marker_embeddings'
    # own sort key exactly (virtues/utils/utils.py: `sorted(os.listdir(embedding_dir))`,
    # filtered to .pt afterwards) -- sorting stripped ids instead silently reorders
    # rows whenever one id is a hyphenated prefix of another (e.g. P08575 vs
    # P08575-5/P08575-8: '-' < '.' in ASCII, so filename order and stripped-id
    # order disagree), which swaps marker identity embeddings with no error.
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

# Pretrained checkpoints published under the bunnelab/virtues HF repo (2026-09),
# replacing the single unversioned checkpoint used for the first benchmark run.
# Same architecture (base_config.yaml unchanged) -- only the trained weights differ.
CHECKPOINT_MAP = {
    'sp32':  'virtues-sp32/model.safetensors',   # 32 spatial-proteomics datasets, CC BY-NC 4.0
    'sp31':  'virtues-sp31/model.safetensors',   # 31 spatial-proteomics datasets, MIT
    'imc14': 'virtues-imc14/model.safetensors',  # 14 IMC datasets, CC BY-NC 4.0
}


def load_virtues_model(virtues_dir, device, checkpoint, marker_embedding_dir):
    sys.path.insert(0, str(virtues_dir))
    os.chdir(str(virtues_dir))

    from omegaconf import OmegaConf
    from safetensors import safe_open
    from virtues.modules.multiplex_virtues import MultiplexVirtues
    from virtues.utils.utils import load_marker_embeddings

    conf = OmegaConf.load(Path(virtues_dir) / 'configs' / 'base_config.yaml')

    if checkpoint == 'legacy':
        weights_path = None
        for candidate in [Path(virtues_dir) / 'assets' / 'checkpoints' / 'model.safetensors',
                          Path(virtues_dir) / 'weights' / 'model.safetensors']:
            if candidate.exists():
                weights_path = candidate
                break
        if weights_path is None:
            raise FileNotFoundError("legacy model.safetensors not found.")
    else:
        if checkpoint not in CHECKPOINT_MAP:
            raise ValueError(f"Unknown checkpoint '{checkpoint}'. "
                             f"Choices: legacy, {', '.join(CHECKPOINT_MAP)}")
        from huggingface_hub import hf_hub_download
        weights_path = Path(hf_hub_download(
            repo_id='bunnelab/virtues', filename=CHECKPOINT_MAP[checkpoint],
            local_dir=str(Path(virtues_dir) / 'assets' / 'checkpoints')))

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
    print(f"VirTues loaded on: {device} | checkpoint={checkpoint} | embedding_dim={embedding_dim}")
    return model, embedding_dim, conf


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


#  DATA LOADING

def _file_stem(name):
    for ext in ('.ome.tiff', '.ome.tif', '.tiff', '.tif'):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def find_image(img_dir, img_id):
    files = [p for p in img_dir.iterdir() if not p.name.startswith('.')]
    exact = [p for p in files if _file_stem(p.name) == img_id]
    if exact:
        return exact[0]
    prefixed = [p for p in files if _file_stem(p.name).startswith(f'{img_id}_')]
    return prefixed[0] if len(prefixed) == 1 else None


def find_segmentation(seg_dir, img_id):
    for ext in ('.tiff', '.tif'):
        if (seg_dir / f'{img_id}{ext}').exists():
            return seg_dir / f'{img_id}{ext}'
    if (seg_dir / img_id).is_dir():
        seg_files = sorted(p for p in (seg_dir / img_id).rglob('segmentationMap.tif')
                           if not p.name.startswith('.'))
        if seg_files:
            return seg_files[0]
    return None


def load_data(cfg, data_root, folds_dir, include_validation=False):
    """
    Returns meta_df[image_id, cell_id, label], folds (cell-level: sets of
    (image_id, cell_id) tuples), img_paths, seg_paths.

    Ground truth comes directly from the quantification table's own
    'cell_type' column. Folds are read from the per-fold fold_{i}_train.csv/
    fold_{i}_test.csv files.
    """
    quant_csv = data_root / 'quantification' / 'processed' / f'{data_root.name}_quantification.csv'
    raw_df = pd.read_csv(quant_csv, usecols=['sample_id', 'cell_id', 'cell_type'])
    raw_df['sample_id'] = raw_df['sample_id'].astype(str).str.replace('.csv', '', regex=False)

    def load_keys(path):
        df = pd.read_csv(path, usecols=['sample_id', 'cell_id'])
        df['sample_id'] = df['sample_id'].astype(str).str.replace('.csv', '', regex=False)
        return set(zip(df['sample_id'], df['cell_id']))

    folds   = []
    fold_i  = 1
    while (folds_dir / f'fold_{fold_i}_train.csv').exists():
        train_keys = load_keys(folds_dir / f'fold_{fold_i}_train.csv')
        if include_validation and (folds_dir / f'fold_{fold_i}_validation.csv').exists():
            train_keys |= load_keys(folds_dir / f'fold_{fold_i}_validation.csv')
        folds.append({
            'train': train_keys,
            'test':  load_keys(folds_dir / f'fold_{fold_i}_test.csv'),
        })
        fold_i += 1
    print(f"Loaded {len(folds)} folds (cell-level) from {folds_dir}")

    exclude_labels = set(cfg.get("exclude_labels", []))
    meta_df = raw_df[~raw_df['cell_type'].isin(exclude_labels)].copy()
    meta_df = meta_df.rename(columns={'sample_id': 'image_id', 'cell_type': 'label'})
    meta_df = meta_df[['image_id', 'cell_id', 'label']]
    print(f"Loaded metadata: {len(meta_df):,} cells "
          f"(dropped {len(raw_df) - len(meta_df):,} excluded-label cells)")

    img_dir   = data_root / 'raw_images' / 'multistack_tiffs'
    seg_dir   = data_root / 'segmentation'
    img_paths, seg_paths = {}, {}
    for img_id in sorted(meta_df['image_id'].unique()):
        img_path = find_image(img_dir, img_id)
        if img_path is not None:
            img_paths[img_id] = img_path
        seg_path = find_segmentation(seg_dir, img_id)
        if seg_path is not None:
            seg_paths[img_id] = seg_path
    print(f"Found {len(img_paths)} images, {len(seg_paths)} segmentation maps")

    return meta_df, folds, img_paths, seg_paths


#  IMAGE / SEGMENTATION READING

def read_image(img_path, clean_indices):
    return tifffile.imread(str(img_path)).astype(np.float32)[clean_indices]


def read_segmentation(seg_path):
    return tifffile.imread(str(seg_path)).astype(np.int32)


#  FEATURE EXTRACTION

def extract_features(args, model, conf, all_images, clean_indices, channel_mask,
                     marker_indices, meta_df, img_paths, seg_paths):
    """
    Runs compute_cell_tokens per image (one token per segmented cell), then
    keeps only the cells present in meta_df (ground truth). Per-image results
    are cached in the format utils_foundational_models.rebuild_img_feature_store()/
    load_embeddings() already expect.
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
        if meta_img.empty:
            print(f"  [skip] {img_id}: no metadata rows")
            continue
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

        if not feats:
            print(f"  [skip] {img_id}: no ground-truth cells extracted")
            torch.cuda.empty_cache()
            continue

        feats_arr = np.array(feats)
        np.save(cache_feat, feats_arr)
        meta_out = pd.DataFrame({'image_id': img_id, 'cell_id': cell_ids, 'label': labels})
        meta_out.to_csv(cache_meta, index=False)

        img_feature_store[img_id] = (feats_arr, labels, cell_ids)
        all_feats_list.append(feats_arr)
        meta_records.append(meta_out)
        torch.cuda.empty_cache()

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
                'supervised': 'Logistic-regression classifier head + 5-fold CV',
                'leiden':     'Leiden clustering + greedy F1',
                'all':        'extract -> supervised -> leiden in sequence',
            }[mode],
        )
        p.add_argument('--dataset-path', required=True,
                       help='Dataset directory (markers.txt, quantification/, raw_images/, segmentation/)')
        p.add_argument('--folds-dir', default=None,
                       help='k-fold directory, absolute or relative to --dataset-path')
        p.add_argument('--include-validation', action=argparse.BooleanOptionalAction, default=None,
                       help='Add fold_{i}_validation.csv cells to the training set')
        p.add_argument('--config',  required=True, help='Path to virtues.json')
        p.add_argument('--output-dir', required=True)
        p.add_argument('--spceleval-dir', default=None)
        p.add_argument('--checkpoint', choices=['sp32', 'sp31', 'imc14', 'legacy'], default=None,
                       help='Which pretrained VirTues weights to load')
        # optional overrides (default None -> config)
        p.add_argument('--virtues-dir', default=None)
        p.add_argument('--device',      default=None)
        p.add_argument('--crop-size',   type=int, default=None)
        p.add_argument('--stride',      type=int, default=None)
        p.add_argument('--chunk-size',  type=int, default=None)
        p.add_argument('--pad-size',    type=int, default=None)
        p.add_argument('--n-folds',     type=int, default=None)
        p.add_argument('--logreg-c',    type=float, default=None)
        p.add_argument('--n-jobs',      type=int, default=None)
        p.add_argument('--leiden-resolution',  type=float, default=None)
        p.add_argument('--leiden-n-neighbors', type=int,   default=None)
        p.add_argument('--leiden-subsample',   type=int,   default=None)
    return parser


#  MAIN

def main():
    parser = build_parser()
    args   = parser.parse_args()
    total_start = time.time()

    data_root    = Path(args.dataset_path)
    args.dataset = data_root.name
    cfg = load_config(args.config, args.dataset)

    def fill(attr, key, default=None):
        if getattr(args, attr) is None:
            setattr(args, attr, cfg.get(key, default))
    fill('checkpoint', 'checkpoint', 'sp32')
    fill('virtues_dir', 'virtues_dir', '/home/juliaoesterle/VirTues')
    fill('device', 'device', 'cuda:1')
    fill('crop_size', 'crop_size', 64)
    fill('stride', 'stride', 42)
    fill('chunk_size', 'chunk_size', 32)
    fill('pad_size', 'pad_size', 120)
    fill('n_folds', 'n_folds', 5)
    fill('logreg_c', 'logreg_c', 1.0)
    fill('n_jobs', 'n_jobs', -1)
    fill('leiden_resolution', 'leiden_resolution', 2.0)
    fill('leiden_n_neighbors', 'leiden_n_neighbors', 15)
    fill('leiden_subsample', 'leiden_subsample', 50000)
    fill('folds_dir', 'folds_dir', 'quantification/processed/kfolds_StratifiedGroupKFold_level3')
    fill('include_validation', 'include_validation', False)

    print(f"\n{'='*60}")
    print(f"run_virtues.py | mode={args.mode} | dataset={args.dataset}")
    print(f"{'='*60}")

    virtues_dir = Path(args.virtues_dir)
    # The shipped assets/example_dataset/marker_embeddings is NOT a general
    # marker library -- it's an exact match to IMMUcan's own clean markers.
    # Other datasets need their own embedding dir (see virtues.json).
    emb_dir = str(virtues_dir / cfg.get('marker_embedding_dir', 'assets/example_dataset/marker_embeddings'))

    with open(data_root / 'markers.txt') as f:
        channels = [line.strip() for line in f if line.strip()]
    clean_indices, _, channel_mask, marker_indices = get_clean_markers(
        channels, cfg.get("exclude_markers", []), cfg.get("uniprot_map", {}), emb_dir)

    meta_df, folds, img_paths, seg_paths = load_data(
        cfg, data_root, data_root / args.folds_dir, args.include_validation)

    all_images = sorted(img_paths.keys())

    all_feats_arr, metadata_all = load_embeddings(args.output_dir)
    if all_feats_arr is not None and args.mode in ('supervised', 'leiden'):
        img_feature_store = rebuild_img_feature_store(args.output_dir, all_images)
    else:
        model, embedding_dim, conf = load_virtues_model(
            virtues_dir, args.device, args.checkpoint, emb_dir)
        img_feature_store, all_feats_arr, metadata_all = extract_features(
            args, model, conf, all_images, clean_indices, channel_mask,
            marker_indices, meta_df, img_paths, seg_paths,
        )

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
