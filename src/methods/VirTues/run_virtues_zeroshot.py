#!/usr/bin/env python3
"""
run_virtues_zeroshot.py — VirTues zero-shot cell phenotyping (no training)
=============================================================================

Per Aroj's review on PR #27: VirTues ships a pretrained U-Net-style
segmentation head (notebooks/3_demo_segmentation.ipynb, VirtuesSegmentationHead)
that predicts BOTH cell instances and a per-pixel 9-class cell-type semantic
map, with no training needed. This script uses only the semantic branch
("the zero-shot prediction") and discards the instance branch entirely --
we already have real segmentation masks for every dataset in this benchmark,
so VirTues's own combined instance segmentation is neither needed nor used.

We do NOT call VirtuesSegmentationHead.segment_tile()/segment_tissue()
directly: those always run the instance-segmentation decoder AND its
InstanceProcessor postprocessing, which (a) we would discard anyway and
(b) has a hardcoded device='cuda' in VirtuesSegmentationHead.__init__,
which crashes on any non-default CUDA device (confirmed directly). Instead,
semantic_tiled_inference() below reuses only the tiling/stitching utilities
from instanseg (the same ones segment_tissue() itself uses) and calls
VirtuesSegmentationHead._encode() + .decoder_phenotypes() directly, skipping
the instance decoder and InstanceProcessor altogether.

Requires VirtuesSegmentationHead, which does not exist in the VirTues commit
pinned for run_virtues.py/run_virtues_sp32.py. See virtues_zeroshot.json's
top-level comment for why this points at a separate VirTues checkout.

Usage
-----
python run_virtues_zeroshot.py \
    --dataset     cHL_2_MIBI \
    --config      src/methods/configs/virtues_zeroshot.json \
    --output-dir  /home/juliaoesterle/results/virtues_zeroshot/chl \
    --device      cuda:1
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from collections import Counter

os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.3')
os.environ.setdefault('LD_LIBRARY_PATH',
                      '/usr/local/cuda-12.3/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import tifffile
import torch
from torchvision.transforms import GaussianBlur
from tqdm import tqdm

warnings.filterwarnings('ignore')


# CONFIG LOADING

def _rebase_local_paths(obj, prefix="/home/juliaoesterle", env_var="SPCELLEVAL_ROOT"):
    """Configs hardcode absolute paths rooted at beauty (/home/juliaoesterle).
    Set SPCELLEVAL_ROOT to re-root every such path onto another machine
    (e.g. Helix's /mnt/sds-hd/sd22c003) without editing the JSON files."""
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
    if dataset not in cfg["datasets"]:
        raise ValueError(f"Dataset '{dataset}' not in {config_path}. "
                         f"Available: {list(cfg['datasets'])}")
    merged = dict(cfg.get("defaults", {}))
    merged.update(cfg["datasets"][dataset])
    merged["dataset"] = dataset
    return _rebase_local_paths(merged)


# CHECKPOINT REGISTRY
# Only 'sp32' has a published segmentation head as of 2026-10. dim_out is the
# number of instance-segmentation channels (computed internally, discarded
# here); num_celltypes is the number of zero-shot semantic classes.
CHECKPOINT_HEADS = {
    'sp32': {
        'model': 'virtues-sp32/model.safetensors',
        'head':  'virtues-sp32/segmentation-head.safetensors',
        'dim_out': 5,
        'num_celltypes': 9,
    },
}


# MODEL LOADING

def load_virtues_zeroshot_model(virtues_dir, device, checkpoint, marker_embedding_dir):
    sys.path.insert(0, str(virtues_dir))
    os.chdir(str(virtues_dir))

    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from virtues.modules.multiplex_virtues import MultiplexVirtues
    from virtues.modules.segmentation.unet import VirtuesSegmentationHead
    from virtues.utils.utils import load_marker_embeddings, load_marker_embedding_dict

    if checkpoint not in CHECKPOINT_HEADS:
        raise ValueError(f"No published segmentation head for checkpoint '{checkpoint}'. "
                         f"Choices: {list(CHECKPOINT_HEADS)}")
    spec = CHECKPOINT_HEADS[checkpoint]

    marker_embeddings = load_marker_embeddings(marker_embedding_dir)
    marker_embedding_dict = load_marker_embedding_dict(marker_embedding_dir)
    print(f"Loaded {len(marker_embeddings)} marker embeddings from {marker_embedding_dir}")

    model = MultiplexVirtues(prior_bias_embeddings=marker_embeddings)
    model_path = hf_hub_download(repo_id='bunnelab/virtues', filename=spec['model'],
                                 local_dir=str(Path(virtues_dir) / 'assets' / 'checkpoints'))
    model.load_state_dict(load_file(model_path, device='cpu'))
    model = model.to(device).eval()

    seg_model = VirtuesSegmentationHead(
        virtues_model=model, dim_out=spec['dim_out'], num_celltypes=spec['num_celltypes'],
    ).to(device).eval()
    head_path = hf_hub_download(repo_id='bunnelab/virtues', filename=spec['head'],
                                local_dir=str(Path(virtues_dir) / 'assets' / 'checkpoints'))
    # strict=False: this checkpoint only contains the new decoder heads, not
    # the backbone (already loaded above into the same virtues_model object).
    seg_model.load_state_dict(load_file(head_path, device=device), strict=False)
    print(f"VirTues zero-shot segmentation head loaded on {device} | checkpoint={checkpoint}")
    return seg_model, marker_embedding_dict


# MARKER INDEX BUILDER

def get_clean_markers(all_channels, exclude, uniprot_map, marker_embedding_dict):
    exclude_set = set(exclude)
    clean_indices, clean_names, midxs, no_embedding = [], [], [], []
    for i, m in enumerate(all_channels):
        if m in exclude_set:
            continue
        uid = uniprot_map.get(m)
        if uid and uid in marker_embedding_dict:
            clean_indices.append(i)
            clean_names.append(m)
            midxs.append(marker_embedding_dict[uid])
        else:
            no_embedding.append(m)
    print(f"  Markers: {len(clean_names)} usable for zero-shot | {len(no_embedding)} skipped")
    if no_embedding:
        print(f"  No embedding: {no_embedding}")
    return clean_indices, clean_names, torch.tensor(midxs, dtype=torch.long)


# PREPROCESSING -- identical convention to run_virtues.py/run_virtues_sp32.py

def preprocess_image(img_sel):
    img_t = torch.from_numpy(img_sel).float()
    C = img_t.shape[0]
    quantiles = torch.quantile(img_t.reshape(C, -1), 0.99, dim=1)
    img_t = torch.clamp(img_t, min=torch.zeros_like(quantiles[:, None, None]),
                        max=quantiles[:, None, None])
    img_t = torch.log1p(img_t)
    img_t = GaussianBlur(kernel_size=3, sigma=1.0)(img_t)
    means = img_t.reshape(C, -1).mean(dim=1)
    stds = img_t.reshape(C, -1).std(dim=1)
    img_t = (img_t - means[:, None, None]) / (stds[:, None, None] + 1e-9)
    return img_t


# ZERO-SHOT SEMANTIC-ONLY TILED INFERENCE

@torch.no_grad()
def semantic_tiled_inference(seg_model, multiplex_tissue, channel_ids, tile_size, overlap, batch_size):
    """Per-pixel zero-shot cell-type logits only. See module docstring for why
    this does not call VirtuesSegmentationHead.segment_tile()/segment_tissue()."""
    from instanseg.utils.tiling import _chops, _tiles_from_chops, _stitch_mean

    h, w = int(multiplex_tissue.shape[-2]), int(multiplex_tissue.shape[-1])
    tile_hw = (min(tile_size, h), min(tile_size, w))
    chop_idx = _chops(multiplex_tissue.shape, shape=tile_hw, overlap=2 * overlap)
    tiles = _tiles_from_chops(multiplex_tissue, shape=tile_hw, tuple_index=chop_idx)

    logits_tiles = []
    for i in range(0, len(tiles), batch_size):
        batch = torch.stack(tiles[i:i + batch_size])
        z0, intermediate = seg_model._encode([t for t in batch], [channel_ids] * len(batch))
        pred = seg_model.decoder_phenotypes(z0, intermediate).detach()
        if pred.shape[-2:] != tile_hw:
            pred = torch.nn.functional.interpolate(pred, size=tile_hw, mode='bilinear', align_corners=False)
        logits_tiles.extend([p for p in pred])

    stitched = _stitch_mean(
        logits_tiles, shape=tile_hw, chop_list=chop_idx,
        final_shape=(logits_tiles[0].shape[0], h, w),
    )
    return stitched.cpu()


def predict_cell_types(seg_model, img_t, midxs, cfg, args):
    semantic_logits = semantic_tiled_inference(
        seg_model, img_t.to(args.device), midxs.to(args.device),
        tile_size=args.tile_size, overlap=args.overlap, batch_size=args.batch_size,
    )
    cell_type_mapping = {int(k): v for k, v in cfg['cell_type_mapping'].items()}
    reverse = {v: k for k, v in cell_type_mapping.items()}
    suppressed_idx = [reverse[c] for c in cfg.get('suppressed_classes', [])]

    probs = torch.softmax(semantic_logits, dim=0)
    logits_s = semantic_logits.clone()
    if suppressed_idx:
        logits_s[suppressed_idx, :, :] = float('-inf')
    semantic_pred = torch.argmax(logits_s, dim=0).numpy()
    confidence = probs.max(dim=0).values.numpy()
    return semantic_pred, confidence, cell_type_mapping


def aggregate_per_cell(seg_mask, semantic_pred, confidence):
    """Majority-vote cell-type assignment onto OUR real segmentation mask --
    never VirTues's own instance mask, which this script never computes."""
    cell_ids_present = np.unique(seg_mask)
    cell_ids_present = cell_ids_present[cell_ids_present != 0]
    if len(cell_ids_present) == 0:
        return {}

    flat_mask = seg_mask.ravel()
    flat_pred = semantic_pred.ravel()
    flat_conf = confidence.ravel()
    order = np.argsort(flat_mask, kind='stable')
    sorted_mask = flat_mask[order]
    sorted_pred = flat_pred[order]
    sorted_conf = flat_conf[order]
    boundaries = np.searchsorted(sorted_mask, np.concatenate([cell_ids_present, [cell_ids_present.max() + 1]]))

    result = {}
    for i, cid in enumerate(cell_ids_present):
        start, end = boundaries[i], boundaries[i + 1]
        preds_here = sorted_pred[start:end]
        vals, counts = np.unique(preds_here, return_counts=True)
        winner = int(vals[np.argmax(counts)])
        mean_conf = float(sorted_conf[start:end][preds_here == winner].mean())
        result[int(cid)] = (winner, mean_conf)
    return result


# IMMUCAN PIPELINE

def run_immucan(seg_model, marker_embedding_dict, cfg, args):
    data_root = Path(cfg['data_root'])
    clean_indices, clean_names, midxs = get_clean_markers(
        cfg['channels'], cfg['exclude_markers'], cfg['uniprot_map'], marker_embedding_dict)

    label_csv = pd.read_csv(data_root / cfg['labels_csv'])
    int2name = dict(zip(label_csv['label'], label_csv['phenotype']))
    gt_map = cfg['ground_truth_to_virtues9class']
    cell_type_mapping = {int(k): v for k, v in cfg['cell_type_mapping'].items()}

    with open(data_root / cfg['folds_json']) as f:
        folds_raw = json.load(f)
    fold_of_image = {}
    fold_idx = sorted({int(k.split('_')[1]) for k in folds_raw if k.startswith('fold_')})
    for i in fold_idx:
        for img in folds_raw.get(f'fold_{i}_test_set', []):
            fold_of_image[img] = i

    img_dir = data_root / cfg['image_dir']
    seg_dir = data_root / cfg['segmentation_dir']
    label_dir = data_root / cfg['label_dir']

    rows = []
    img_names = sorted(p.stem for p in img_dir.glob('*.npz'))
    if args.max_images:
        img_names = img_names[:args.max_images]
    for img_name in tqdm(img_names, desc='VirTues zero-shot IMMUcan'):
        npz_file = img_dir / f'{img_name}.npz'
        seg_file = seg_dir / f'{img_name}.tiff'
        label_file = label_dir / f'{img_name}.txt'
        if not all(f.exists() for f in [npz_file, seg_file, label_file]):
            continue

        img_raw = np.load(npz_file)['data'].astype(np.float32)[clean_indices]
        seg_np = tifffile.imread(seg_file).astype(np.int32)
        img_t = preprocess_image(img_raw)

        with open(label_file) as f:
            raw_labels = [int(l.strip()) for l in f]

        semantic_pred, confidence, _ = predict_cell_types(seg_model, img_t, midxs, cfg, args)
        per_cell = aggregate_per_cell(seg_np, semantic_pred, confidence)

        for cid, (pred_idx, conf) in per_cell.items():
            raw_lbl = raw_labels[cid - 1] if (cid - 1) < len(raw_labels) else -1
            if raw_lbl == -1:
                continue  # no annotation at all for this cell -- can't score it
            true_name = int2name.get(raw_lbl, 'Unknown_gt')
            rows.append({
                'image_id': img_name,
                'cell_id': cid,
                'fold': fold_of_image.get(img_name, -1),
                'true_phenotype_original': true_name,
                'true_phenotype': gt_map.get(true_name, 'Unknown'),
                'predicted_phenotype': cell_type_mapping[pred_idx],
                'confidence': conf,
            })
    return pd.DataFrame(rows)


# cHL PIPELINE

def run_chl(seg_model, marker_embedding_dict, cfg, args):
    data_root = Path(cfg['data_root'])
    clean_indices, clean_names, midxs = get_clean_markers(
        cfg['channels'], cfg['exclude_markers'], cfg['uniprot_map'], marker_embedding_dict)

    meta_df = pd.read_csv(data_root / cfg['quant_csv'])
    meta_df['sample_id'] = meta_df['sample_id'].astype(str).str.replace('.csv', '', regex=False)
    gt_map = cfg['ground_truth_to_virtues9class']
    cell_type_mapping = {int(k): v for k, v in cfg['cell_type_mapping'].items()}

    with open(data_root / cfg['folds_json']) as f:
        folds = json.load(f)['folds']
    fold_of_key = {}
    for i, fold in enumerate(folds):
        for idx in fold['test']:
            fold_of_key[int(idx)] = i

    meta_df['_key'] = meta_df['sample_id'] + '_' + meta_df['cell_id'].astype(str)
    key_to_row = dict(zip(meta_df['_key'], meta_df.index))

    img_dir = data_root / cfg['image_dir']
    img_paths = {p.name.replace('_stacked.ome.tif', ''): p
                for p in sorted(img_dir.glob('*_stacked.ome.tif'))}
    items = sorted(img_paths.items())
    if args.max_images:
        items = items[:args.max_images]

    rows = []
    for img_id, img_path in tqdm(items, desc='VirTues zero-shot cHL', total=len(items)):
        seg_dir = data_root / cfg['segmentation_dir'] / img_id
        seg_hits = [f for f in seg_dir.rglob('segmentationMap.tif') if not f.name.startswith('.')]
        if not seg_hits:
            continue

        img_raw = tifffile.imread(img_path).astype(np.float32)[clean_indices]
        seg_np = tifffile.imread(seg_hits[0]).astype(np.int32)
        img_t = preprocess_image(img_raw)

        semantic_pred, confidence, _ = predict_cell_types(seg_model, img_t, midxs, cfg, args)
        per_cell = aggregate_per_cell(seg_np, semantic_pred, confidence)

        for cid, (pred_idx, conf) in per_cell.items():
            key = f'{img_id}_{cid}'
            row_idx = key_to_row.get(key)
            if row_idx is None:
                continue
            true_name = meta_df.loc[row_idx, 'cell_type']
            rows.append({
                'image_id': img_id,
                'cell_id': cid,
                'fold': fold_of_key.get(row_idx, -1),
                'true_phenotype_original': true_name,
                'true_phenotype': gt_map.get(true_name, 'Unknown'),
                'predicted_phenotype': cell_type_mapping[pred_idx],
                'confidence': conf,
            })
    return pd.DataFrame(rows)


# MAIN

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_virtues_zeroshot.py',
        description='VirTues zero-shot cell phenotyping (no training) -- IMMUcan + cHL_2_MIBI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', choices=['immucan', 'cHL_2_MIBI'], required=True)
    parser.add_argument('--config', required=True, help='Path to virtues_zeroshot.json')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--virtues-dir', default=None)
    parser.add_argument('--checkpoint', choices=list(CHECKPOINT_HEADS), default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--tile-size', type=int, default=None)
    parser.add_argument('--overlap', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--max-images', type=int, default=None,
                        help='Smoke-test cap on the number of images processed.')
    return parser


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config, args.dataset)

    def fill(attr, key, default=None):
        if getattr(args, attr, None) is None:
            setattr(args, attr, cfg.get(key, default))
    fill('virtues_dir', 'virtues_dir', '/home/juliaoesterle/VirTues_zeroshot')
    fill('checkpoint', 'checkpoint', 'sp32')
    fill('device', 'device', 'cuda:1')
    fill('tile_size', 'tile_size', 128)
    fill('overlap', 'overlap', 32)
    fill('batch_size', 'batch_size', 16)

    print(f"\n{'='*60}")
    print(f"run_virtues_zeroshot.py | dataset={args.dataset} | checkpoint={args.checkpoint}")
    print(f"NOTE: zero-shot, no training -- 'fold' here only records which "
          f"fold's test split a cell belongs to, for comparison with other methods.")
    print(f"{'='*60}")

    total_start = time.time()
    virtues_dir = Path(args.virtues_dir)
    emb_dir = str(virtues_dir / cfg['marker_embedding_dir'])
    seg_model, marker_embedding_dict = load_virtues_zeroshot_model(
        virtues_dir, args.device, args.checkpoint, emb_dir)

    if args.dataset == 'immucan':
        df = run_immucan(seg_model, marker_embedding_dict, cfg, args)
    else:
        df = run_chl(seg_model, marker_embedding_dict, cfg, args)

    out_dir = Path(args.output_dir) / 'VIRTUES_zeroshot' / 'level3'
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'predictions_zeroshot.csv', index=False)
    print(f"\n{len(df):,} cells -> {out_dir}/predictions_zeroshot.csv")

    known = df[df['true_phenotype'] != 'Unknown']
    if len(known):
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(known['true_phenotype'], known['predicted_phenotype'])
        mf1 = f1_score(known['true_phenotype'], known['predicted_phenotype'], average='macro', zero_division=0)
        print(f"Harmonized 9-class zero-shot (excluding 'Unknown' ground truth): "
              f"Acc={acc:.4f}  MacroF1={mf1:.4f}")
        print("Reminder: the ground_truth_to_virtues9class mapping in the config "
              "is a PROPOSED DEFAULT, not yet confirmed by Lukas/Aroj.")
    print(f"Total time: {time.time() - total_start:.1f}s")


if __name__ == '__main__':
    main()
