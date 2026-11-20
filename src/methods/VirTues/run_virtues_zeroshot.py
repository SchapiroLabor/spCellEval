#!/usr/bin/env python3
"""
run_virtues_zeroshot.py — VirTues zero-shot cell phenotyping (no training)
==========================================================================

Uses VirTues' pretrained segmentation head as in the authors'
notebooks/3_demo_segmentation.ipynb. Images are standardised as by the authors'
loader (spora-io, quantile_clipping_log1p with use_mean_std and gaussian_blur):
each channel clipped at its 99th percentile, log1p, z-scored with the dataset's
mean and std, then blurred. The per-pixel cell-type logits are
computed tile by tile and stitched as in VirtuesSegmentationHead.segment_tissue,
'None' and 'Unknown' are suppressed, and each cell gets the majority label of
its pixels (virtues.utils.segmentation.assign_cell_types). VirTues' own instance
segmentation is not used: cells come from our segmentation masks.

Markers (exclusions, UniProt map) and marker embeddings are shared with
run_virtues.py via configs/virtues.json.
The segmentation head needs a newer VirTues checkout than run_virtues.py
(bunnelab/virtues at 2c9c6cb or later), passed as --virtues-dir.

Predictions keep VirTues' own labels (B cell, CD4 T cell, CD8 T cell,
Fibroblasts /Stroma, Myeloid, Tumor, Vessel / Endo.); mapping them onto a
dataset's cell types is left to the evaluation. Nothing is trained, so the
folds only select the test cells written to each predictions_<fold>.csv.

Usage
-----
python src/methods/VirTues/run_virtues_zeroshot.py \\
    --dataset-path <data>/IMMUcan \\
    --config       src/methods/configs/virtues.json \\
    --virtues-dir  <VirTues checkout with the segmentation head> \\
    --output-dir   <results>/IMMUcan/virtues_zeroshot
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torchvision.transforms import GaussianBlur
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_virtues import get_clean_markers  # noqa: E402
from benchmark_io import (  # noqa: E402  (path set by run_virtues)
    fill_from_config, load_benchmark_data, load_config, read_image, read_markers, read_segmentation,
)

# Weights published with a segmentation head, and the head's classes (notebooks/3_demo_segmentation.ipynb)
CHECKPOINTS = {'sp32': ('virtues-sp32/model.safetensors', 'virtues-sp32/segmentation-head.safetensors')}
CELL_TYPES = ['None', 'B cell', 'CD4 T cell', 'CD8 T cell', 'Fibroblasts /Stroma', 'Myeloid', 'Tumor',
              'Unknown', 'Vessel / Endo.']
SUPPRESSED = ['None', 'Unknown']


def log_clipped(img, channel_mask):
    """(C, H, W) channels with an embedding, clipped at their 99th percentile, log1p."""
    x = torch.from_numpy(img).float()[channel_mask]
    high = torch.quantile(x.reshape(len(x), -1), 0.99, dim=1)[:, None, None]
    return torch.log1p(torch.clamp(x, min=torch.zeros_like(high), max=high))


def dataset_mean_std(images, channel_mask):
    """Per-channel mean and std of the log_clipped values over all pixels of all images."""
    total = total_sq = n = 0
    for img in images:
        x = log_clipped(img, channel_mask).double().reshape(int(channel_mask.sum()), -1)
        total, total_sq, n = total + x.sum(1), total_sq + (x ** 2).sum(1), n + x.shape[1]
    mean = total / n
    return mean.float()[:, None, None], (total_sq / n - mean ** 2).sqrt().float()[:, None, None]


def standardise(img, channel_mask, mean, std):
    return GaussianBlur(kernel_size=3, sigma=1.0)((log_clipped(img, channel_mask) - mean) / (std + 1e-8))


def load_segmentation_head(virtues_dir, device, checkpoint, marker_embedding_dir):
    """VirTues and its segmentation head, loaded as in notebooks/3_demo_segmentation.ipynb."""
    sys.path.insert(0, str(virtues_dir))
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from virtues.modules.multiplex_virtues import MultiplexVirtues
    from virtues.modules.segmentation.unet import VirtuesSegmentationHead
    from virtues.utils.utils import load_marker_embeddings

    cache = str(virtues_dir / 'assets' / 'checkpoints')
    model_file, head_file = CHECKPOINTS[checkpoint]
    model = MultiplexVirtues(prior_bias_embeddings=load_marker_embeddings(str(marker_embedding_dir)))
    model.load_state_dict(load_file(hf_hub_download('bunnelab/virtues', model_file, local_dir=cache)))
    head = VirtuesSegmentationHead(virtues_model=model.to(device).eval(), dim_out=5,
                                   num_celltypes=len(CELL_TYPES)).to(device).eval()
    head.load_state_dict(load_file(hf_hub_download('bunnelab/virtues', head_file, local_dir=cache)),
                         strict=False)  # the head file holds the decoders only
    print(f"VirTues segmentation head loaded on {device} | checkpoint={checkpoint}")
    return head


@torch.no_grad()
def cell_type_logits(head, img, channel_ids, tile_size, overlap, batch_size):
    """(n_cell_types, H, W) logits, tiled and stitched as in VirtuesSegmentationHead.segment_tissue."""
    from instanseg.utils.tiling import _chops, _stitch_mean, _tiles_from_chops
    h, w = img.shape[-2:]
    tile_hw = (min(tile_size, h), min(tile_size, w))
    chops = _chops(img.shape, shape=tile_hw, overlap=2 * overlap)
    tiles = _tiles_from_chops(img, shape=tile_hw, tuple_index=chops)
    logits = []
    for i in range(0, len(tiles), batch_size):
        batch = torch.stack(tiles[i:i + batch_size])
        logits.extend(head(batch, [channel_ids] * len(batch))[:, head.dim_out:])
    return _stitch_mean(logits, shape=tile_hw, chop_list=chops, final_shape=(len(CELL_TYPES), h, w))


def predict_cells(logits, seg, cell_ids):
    """Majority cell type of each cell's pixels after suppressing SUPPRESSED (in place in logits),
    and the cell's mean probability for it."""
    from virtues.utils.segmentation import assign_cell_types
    logits[[CELL_TYPES.index(c) for c in SUPPRESSED]] = float('-inf')
    prob = torch.softmax(logits, dim=0)
    cell_map = assign_cell_types(seg, prob.argmax(0).cpu().numpy())
    winner_prob = prob.gather(0, torch.from_numpy(cell_map).long().to(prob.device)[None])[0].cpu().numpy()
    ids, first = np.unique(seg, return_index=True)
    types = dict(zip(ids, cell_map.ravel()[first]))
    return ([CELL_TYPES[types[c]] for c in cell_ids],
            ndimage.mean(winner_prob, labels=seg, index=cell_ids))


def build_parser():
    p = argparse.ArgumentParser(prog='run_virtues_zeroshot.py', description=__doc__.split('\n')[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset-path', required=True,
                   help='Dataset directory (markers.txt, quantification/, raw_images/, segmentation/)')
    p.add_argument('--config', required=True, help='Path to virtues.json')
    p.add_argument('--virtues-dir', required=True, help='VirTues checkout with the segmentation head')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--folds-dir', default=None, help='k-fold directory, absolute or relative to --dataset-path')
    p.add_argument('--marker-embedding-dir', default=None,
                   help='Directory of <UniProt id>.pt ESM embeddings '
                        '(default: marker_embeddings/ next to this script)')
    p.add_argument('--checkpoint', choices=list(CHECKPOINTS), default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--tile-size', type=int, default=128, help='VirTues training crop size')
    p.add_argument('--overlap', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=16)
    return p


def main():
    args = build_parser().parse_args()
    start = time.time()
    data_root = Path(args.dataset_path).resolve()
    cfg = load_config(args.config, data_root.name)
    fill_from_config(args, cfg)
    emb_dir = Path(args.marker_embedding_dir or SCRIPT_DIR / 'marker_embeddings').resolve()

    clean_indices, _, channel_mask, marker_indices = get_clean_markers(
        read_markers(data_root), cfg.get('exclude_markers', []), cfg.get('uniprot_map', {}), emb_dir)
    meta_df, folds, img_paths, seg_paths = load_benchmark_data(
        data_root, data_root / args.folds_dir, cfg.get('exclude_labels', []))
    head = load_segmentation_head(Path(args.virtues_dir).resolve(), args.device, args.checkpoint, emb_dir)
    channel_ids = marker_indices.to(args.device)
    images = meta_df.groupby('image_id')
    mean, std = dataset_mean_std((read_image(img_paths[i], clean_indices) for i, _ in images), channel_mask)

    preds = []
    for img_id, cells in tqdm(images, desc='VirTues zero-shot'):
        img = standardise(read_image(img_paths[img_id], clean_indices), channel_mask, mean, std).to(args.device)
        seg = read_segmentation(seg_paths[img_id])
        logits = cell_type_logits(head, img, channel_ids, args.tile_size, args.overlap, args.batch_size)
        cells = cells[cells['cell_id'].isin(np.unique(seg))]
        labels, confidence = predict_cells(logits, seg, cells['cell_id'].to_numpy())
        preds.append(cells.assign(predicted_phenotype=labels, confidence=confidence))
    preds = pd.concat(preds, ignore_index=True)

    out_dir = Path(args.output_dir) / 'VIRTUES_zeroshot' / 'level3'
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = pd.Series(list(zip(preds['image_id'], preds['cell_id'])))
    for i, fold in enumerate(folds):
        test = preds[keys.isin(fold['test']).values].rename(columns={'label': 'true_phenotype'})
        test.assign(fold=i)[['image_id', 'cell_id', 'true_phenotype', 'fold', 'predicted_phenotype',
                             'confidence']].to_csv(out_dir / f'predictions_{i}.csv', index=False)
    print(f"{len(preds):,} cells -> {out_dir} | {time.time() - start:.0f}s")


if __name__ == '__main__':
    main()
