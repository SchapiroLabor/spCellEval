#!/usr/bin/env python3
"""
run_virtues.py — VirTues unified pipeline: IMMUcan + cHL_2_MIBI
================================================================
Author: Julia Oesterle
Date:   June 2026

Forked from run_virtues1.py (the working full-tissue compute_cell_tokens
approach, best result crop64 F1=0.579 on IMMUcan).

The IMMUcan path is identical to run_virtues1.py — same utils_foundational_models,
same preprocessing, same compute_cell_tokens call.
The cHL path forks only in data loading and fold handling.

IMPORTANT: Run inside nix develop + stellar micromamba env on beauty:
    cd /home/juliaoesterle/VirTues
    nix develop
    micromamba activate stellar

Usage
-----
# IMMUcan — all 5 folds, crop64 (best on IMMUcan):
python run_virtues.py all \
    --dataset     immucan \
    --virtues-dir /home/juliaoesterle/VirTues \
    --data-dir    /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan/ \
    --output-dir  /home/juliaoesterle/results/virtues/immucan/crop64 \
    --crop-size   64 --stride 21 --device cuda:0

# cHL_2_MIBI — all 5 folds:
python run_virtues.py all \
    --dataset     chl \
    --virtues-dir /home/juliaoesterle/VirTues \
    --output-dir  /home/juliaoesterle/results/virtues/chl/crop64 \
    --crop-size   64 --stride 21 --device cuda:0

# Single fold smoke test:
python run_virtues.py all \
    --dataset immucan --fold 0 --crop-size 64 \
    --virtues-dir /home/juliaoesterle/VirTues \
    --data-dir /home/juliaoesterle/data/phenotyping_benchmark/IMMUcan/ \
    --output-dir /home/juliaoesterle/results/virtues/immucan/crop64 \
    --device cuda:0
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
                      '/usr/local/cuda-12.3/lib64:' +
                      os.environ.get('LD_LIBRARY_PATH', ''))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils_foundational_models import (
    load_label_map, load_folds, get_label,
    save_embeddings, load_embeddings, rebuild_img_feature_store,
    run_supervised, run_leiden, add_shared_args,
)

warnings.filterwarnings('ignore')

# Marker Definitions

# IMMUcan 
ALL_BIOMARKERS = [
    "MPO","HistoneH3","SMA","CD16","CD38","HLADR","CD27","CD15",
    "CD45RA","CD163","B2M","CD20","CD68","Ido1","CD3","LAG3",
    "CD11c","PD1","PDGFRb","CD7","GrzB","PDL1","TCF7","CD45RO",
    "FOXP3","ICOS","CD8a","CarbonicAnhydrase","CD33","Ki67",
    "VISTA","CD40","CD4","CD14","Ecad","CD303","CD206",
    "cleavedPARP","DNA1","DNA2"
]
DEFAULT_EXCLUDE = ['DNA1','DNA2','HistoneH3']

IMMUCAN_UNIPROT = {
    'MPO':'P05164','SMA':'P62736','CD16':'P08637','CD38':'P28907',
    'HLADR':'P01903','CD27':'P26842','CD15':'P22083','CD45RA':'P08575-8',
    'CD163':'Q86VB7','B2M':'P61769','CD20':'P11836','CD68':'P34810',
    'Ido1':'P14902','CD3':'P07766','LAG3':'P18627','CD11c':'P20701',
    'PD1':'Q15116','PDGFRb':'P09619','CD7':'P09564','GrzB':'P10144',
    'PDL1':'Q9NZQ7','TCF7':'P36402','CD45RO':'P08575-5','FOXP3':'Q9BZS1',
    'ICOS':'Q9Y288','CD8a':'P01732','CarbonicAnhydrase':'Q16790',
    'CD33':'P20138','Ki67':'P46013','VISTA':'Q9H7M9','CD40':'P25942',
    'CD4':'P01730','CD14':'P08235','Ecad':'P12830','CD303':'Q8WTT0',
    'CD206':'P22897','cleavedPARP':'P09874',
}

# cHL_2_MIBI
CHL_ALL_CHANNELS = [
    "B2-Microglobulin","CD103","CD11b","CD11c","CD138",
    "CD14","CD15","CD161","CD163","CD20","CD21","CD28",
    "CD3","CD33","CD38","CD4","CD44","CD45RO","CD56",
    "CD68","CD7","CD8a","Collagen1","Granzyme B","HLA1",
    "HLADR","IL-10","Ki-67","Lag3","MPO","Na-K ATPase",
    "PD-1","PD-L1","Pax-5","RORgT","TCRgd","Tox",
    "anti-H2AX","dsDNA","pSLP-76","pSTAT3","pSTAT5",
    "pan-Cytokeratin","pimidazole","HistoneH3","SLP-76",
]
CHL_DEFAULT_EXCLUDE = ["dsDNA","HistoneH3","anti-H2AX","pSLP-76","SLP-76"]
CHL_EXCLUDE_LABELS  = {"undefined"}

CHL_UNIPROT = {
    "B2-Microglobulin":"P61769","CD103":"P38570","CD11b":"P11215",
    "CD11c":"P20702","CD138":"P18827","CD14":"P08235","CD15":"P15291",
    "CD161":"P26d85","CD163":"Q86VB7","CD20":"P11836","CD21":"P20023",
    "CD28":"P10747","CD3":"P09693","CD33":"P20138","CD38":"P28907",
    "CD4":"P01730","CD44":"P16070","CD45RO":"P08575","CD56":"P13591",
    "CD68":"P34810","CD7":"P09564","CD8a":"P01732","Collagen1":"P02452",
    "Granzyme B":"P10144","HLA1":"P04439","HLADR":"P01903","IL-10":"P22301",
    "Ki-67":"P46013","Lag3":"P18627","MPO":"P05164","Na-K ATPase":"P05023",
    "PD-1":"Q15116","PD-L1":"Q9NZQ7","Pax-5":"Q02548","RORgT":"P51449",
    "TCRgd":"P04054","Tox":"O94900","pSTAT3":"P40763","pSTAT5":"P42229",
    "pan-Cytokeratin":"P04264",
    # pimidazole: no UniProt (hypoxia marker) — will be skipped silently
}

# Marker Index Builder

def get_clean_markers(all_channels, exclude, uniprot_map, marker_embedding_dir):
    exclude_set   = set(exclude)
    available_ids = {f.removesuffix('.pt')
                     for f in os.listdir(marker_embedding_dir) if f.endswith('.pt')}
    sorted_ids = sorted(available_ids)
    embed_idx  = {pid: i for i, pid in enumerate(sorted_ids)}

    clean_indices  = []
    clean_names    = []
    channel_mask   = []
    marker_indices = []
    no_embedding   = []

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

    print(f"  Markers: {len(clean_names)} used | "
          f"{sum(channel_mask)} with embeddings | "
          f"{len(no_embedding)} skipped")
    if no_embedding:
        print(f"  No embedding: {no_embedding}")

    channel_mask_t   = torch.tensor(channel_mask, dtype=torch.bool)
    marker_indices_t = torch.tensor(
        [marker_indices[i] for i, has in enumerate(channel_mask) if has],
        dtype=torch.long)
    return clean_indices, clean_names, channel_mask_t, marker_indices_t, no_embedding

#  Model loading

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

# Preprocessing

def preprocess_image(img_raw, channel_mask):
    from torchvision.transforms import GaussianBlur
    img_t = torch.from_numpy(img_raw).float()
    img_t = img_t[channel_mask]
    C, H, W = img_t.shape
    quantiles = torch.quantile(img_t.reshape(C,-1), 0.99, dim=1)
    img_t = torch.clamp(img_t, min=torch.zeros_like(quantiles[:,None,None]),
                        max=quantiles[:,None,None])
    img_t = torch.log1p(img_t)
    img_t = GaussianBlur(kernel_size=3, sigma=1.0)(img_t)
    means = img_t.reshape(C,-1).mean(dim=1)
    stds  = img_t.reshape(C,-1).std(dim=1)
    img_t = (img_t - means[:,None,None]) / (stds[:,None,None] + 1e-9)
    return img_t, means, stds

# Wrapper for compute_cell_tokens with padding 

PAD_SIZE = 120

def run_compute_cell_tokens(img_processed, seg_np, marker_indices,
                             model, conf, args):
    """
    Pad + call compute_cell_tokens exactly as in run_virtues1.py.
    seg_np passed as numpy (not tensor) — that is what the function expects.
    """
    from virtues.utils.cell_tokens import compute_cell_tokens

    mask_t = torch.from_numpy(seg_np.astype(np.int32))
    img_padded  = F.pad(img_processed,
                        (PAD_SIZE, PAD_SIZE, PAD_SIZE, PAD_SIZE),
                        mode='constant', value=0)
    mask_padded = F.pad(mask_t,
                        (PAD_SIZE, PAD_SIZE, PAD_SIZE, PAD_SIZE),
                        mode='constant', value=0)

    crop_size = args.crop_size if args.crop_size else conf.data.crop_size

    cell_ids_out, cell_tokens, _, _ = compute_cell_tokens(
        model=model,
        img=img_padded,
        channel=marker_indices,
        segmentation_mask=mask_padded.numpy(),
        device=args.device,
        crop_size=crop_size,
        patch_size=conf.model.patch_size,
        stride=args.stride,
        chunk_size=args.chunk_size,
    )
    return cell_ids_out, cell_tokens.numpy()

# IMMUcan Pipeline

def extract_features_immucan(args, model, embedding_dim, conf,
                              all_images, clean_indices, clean_names,
                              channel_mask, marker_indices, label_map):
    data_dir  = Path(args.data_dir)
    cache_dir = Path(args.output_dir) / 'embeddings' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    NPZ_DIR    = data_dir / 'CellTypes' / 'data' / 'images'
    MASK_DIR   = data_dir / 'segmentation'
    LABELS_DIR = data_dir / 'CellTypes' / 'cells2labels'

    img_feature_store = {}
    crop_size = args.crop_size if args.crop_size else conf.data.crop_size
    print(f"\n[IMMUcan] crop_size={crop_size} | stride={args.stride} | "
          f"pad={PAD_SIZE} | emb_dim={embedding_dim}")

    for img_name in tqdm(all_images, desc="VirTues IMMUcan"):
        cache_feat = cache_dir / f"{img_name}_feat.npy"
        cache_meta = cache_dir / f"{img_name}_meta.csv"

        if cache_feat.exists() and cache_meta.exists():
            feats = np.load(cache_feat)
            meta  = pd.read_csv(cache_meta)
            img_feature_store[img_name] = (
                feats, meta['label'].tolist(), meta['cell_id'].tolist())
            continue

        npz_file   = NPZ_DIR    / f"{img_name}.npz"
        mask_file  = MASK_DIR   / f"{img_name}.tiff"
        label_file = LABELS_DIR / f"{img_name}.txt"
        if not all(f.exists() for f in [npz_file, mask_file, label_file]):
            print(f"  Skipping {img_name} — missing files")
            continue

        img_raw = np.load(npz_file)['data'].astype(np.float32)
        img_raw = img_raw[clean_indices]
        img_processed, _, _ = preprocess_image(img_raw, channel_mask)

        seg_np = tifffile.imread(mask_file)
        with open(label_file) as f:
            cell_labels = [int(l.strip()) for l in f.readlines()]

        cell_ids_out, feats_list = run_compute_cell_tokens(
            img_processed, seg_np, marker_indices, model, conf, args)

        labels_list   = [get_label(int(cid), cell_labels, label_map)
                         for cid in cell_ids_out]
        cell_ids_list = [int(cid) for cid in cell_ids_out]
        feats_arr     = np.array(feats_list)

        np.save(cache_feat, feats_arr)
        pd.DataFrame({'image_id':[img_name]*len(cell_ids_list),
                      'cell_id':cell_ids_list,'label':labels_list,
                      }).to_csv(cache_meta, index=False)
        img_feature_store[img_name] = (feats_arr, labels_list, cell_ids_list)
        torch.cuda.empty_cache()

    all_f, all_l, all_i, all_n = [], [], [], []
    for img_name, (feats, labels, cell_ids) in img_feature_store.items():
        all_f.extend(feats); all_l.extend(labels)
        all_i.extend(cell_ids); all_n.extend([img_name]*len(cell_ids))

    all_feats_arr = np.array(all_f)
    metadata_all  = pd.DataFrame({'image_id':all_n,'cell_id':all_i,'label':all_l})
    save_embeddings(args.output_dir, all_feats_arr, metadata_all)
    print(f"\nExtraction complete: {len(all_feats_arr):,} cells")
    print(pd.Series(all_l).value_counts().to_string())
    return img_feature_store, all_feats_arr, metadata_all

# cHL Data Loading

BASE_CHL = Path("/home/juliaoesterle/data/phenotyping_benchmark/cHL_2_MIBI")

def load_chl_metadata():
    meta_df = pd.read_csv(
        BASE_CHL / "quantification/processed/cHL_2_MIBI_quantification.csv")
    meta_df["sample_id"] = (meta_df["sample_id"].astype(str)
                            .str.replace(".csv","",regex=False))
    print(f"cHL: {len(meta_df):,} cells, {meta_df['cell_type'].nunique()} types")

    with open(BASE_CHL / "quantification/processed/"
              "kfolds_StratifiedGroupKFold_level3/fold_indices.json") as f:
        folds = json.load(f)["folds"]
    print(f"cHL: {len(folds)} cell-level folds")

    img_dir  = BASE_CHL / "raw_images/multistack_tiffs"
    img_paths = {p.name.replace("_stacked.ome.tif",""):p
                 for p in sorted(img_dir.glob("*_stacked.ome.tif"))}
    print(f"cHL: {len(img_paths)} images: {list(img_paths.keys())}")

    seg_paths = {}
    for img_id in img_paths:
        d = BASE_CHL / "segmentation" / img_id
        if not d.exists(): continue
        hits = [f for f in d.rglob("segmentationMap.tif")
                if not f.name.startswith(".")]
        if hits: seg_paths[img_id] = hits[0]
    print(f"cHL: {len(seg_paths)} segmentations found")

    return meta_df, folds, img_paths, seg_paths

# cHL Pipeline (extract + RF in one go, no image-level store needed)

def extract_features_chl(args, model, embedding_dim, conf,
                          clean_indices, channel_mask, marker_indices):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    meta_df, folds, img_paths, seg_paths = load_chl_metadata()

    cache_dir = Path(args.output_dir) / 'embeddings' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_df["_key"] = meta_df["sample_id"] + "_" + meta_df["cell_id"].astype(str)
    key_to_row = dict(zip(meta_df["_key"], meta_df.index))

    crop_size = args.crop_size if args.crop_size else conf.data.crop_size
    print(f"\n[cHL] crop_size={crop_size} | stride={args.stride} | "
          f"pad={PAD_SIZE} | emb_dim={embedding_dim}")

    feat_store = {}  # row_idx → feature vector

    for img_id, img_path in tqdm(sorted(img_paths.items()),
                                  desc="VirTues cHL", total=len(img_paths)):
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

        img_raw = tifffile.imread(img_path).astype(np.float32)   # (46, H, W)
        seg_np  = tifffile.imread(seg_paths[img_id]).astype(np.int32)

        img_sel = img_raw[clean_indices]                          # (C_clean, H, W)
        img_processed, _, _ = preprocess_image(img_sel, channel_mask)

        cell_ids_out, feats_list = run_compute_cell_tokens(
            img_processed, seg_np, marker_indices, model, conf, args)

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

    # 5-fold CV (cell-level)
    fold_range = [args.fold] if args.fold is not None else range(len(folds))

    for fold_idx in fold_range:
        fold      = folds[fold_idx]
        train_idx = [int(i) for i in fold["train"]]
        test_idx  = [int(i) for i in fold["test"]]

        X_tr, y_tr = [], []
        for idx in train_idx:
            if idx not in feat_store: continue
            lbl = meta_df.loc[idx,"cell_type"]
            if lbl in CHL_EXCLUDE_LABELS: continue
            X_tr.append(feat_store[idx]); y_tr.append(lbl)

        X_te, y_te, keys = [], [], []
        for idx in test_idx:
            if idx not in feat_store: continue
            lbl = meta_df.loc[idx,"cell_type"]
            if lbl in CHL_EXCLUDE_LABELS: continue
            X_te.append(feat_store[idx]); y_te.append(lbl)
            keys.append(idx)

        print(f"\n[Fold {fold_idx}] Train={len(X_tr):,} Test={len(X_te):,}")
        clf = RandomForestClassifier(n_estimators=500, n_jobs=args.n_jobs,
                                     random_state=42, class_weight="balanced")
        clf.fit(np.array(X_tr), y_tr)
        y_pred = clf.predict(np.array(X_te))

        acc = accuracy_score(y_te, y_pred)
        mf1 = f1_score(y_te, y_pred, average="macro",    zero_division=0)
        wf1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
        print(f"  Acc={acc:.4f}  MacroF1={mf1:.4f}  WF1={wf1:.4f}")
        print(classification_report(y_te, y_pred, zero_division=0))

        pred_dir = Path(args.output_dir) / "VIRTUES_supervised" / "level3"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"cell_id":keys,"true_label":y_te,"pred_label":y_pred}
                     ).to_csv(pred_dir / f"predictions_{fold_idx}.csv", index=False)
        print(f"  -> {pred_dir}/predictions_{fold_idx}.csv")

# Argument Parser

def build_parser():
    parser = argparse.ArgumentParser(
        prog='run_virtues.py',
        description='VirTues unified pipeline — IMMUcan + cHL_2_MIBI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode', required=True)

    for m in ['extract','supervised','leiden','all']:
        p = sub.add_parser(m, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('--dataset',     choices=['immucan','chl'], required=True)
        p.add_argument('--virtues-dir', default='/home/juliaoesterle/VirTues')
        p.add_argument('--device',      default='cuda:1')
        p.add_argument('--crop-size',   type=int, default=None,
                       help='Override crop size (None=128 from config; 64=IMMUcan best)')
        p.add_argument('--stride',      type=int, default=42,
                       help='Stride for compute_cell_tokens (42=config default; 21 for crop64)')
        p.add_argument('--chunk-size',  type=int, default=32)
        p.add_argument('--fold',        type=int, default=None,
                       help='Single fold only (default: all 5)')
        p.add_argument('--exclude-markers', nargs='+', default=DEFAULT_EXCLUDE)
        add_shared_args(p)   # --data-dir, --output-dir, --n-folds, --n-estimators etc.

    return parser

# Main

def main():
    parser = build_parser()
    args   = parser.parse_args()

    total_start = time.time()
    print(f"\n{'='*60}")
    print(f"run_virtues.py | dataset={args.dataset} | mode={args.mode}")
    print(f"{'='*60}")

    virtues_dir = Path(args.virtues_dir)
    emb_dir     = str(virtues_dir / 'assets' / 'example_dataset' / 'marker_embeddings')

    model, embedding_dim, conf, marker_embedding_dir = \
        load_virtues_model(virtues_dir, args.device)

    if args.dataset == 'immucan':
        clean_indices, clean_names, channel_mask, marker_indices, _ = \
            get_clean_markers(ALL_BIOMARKERS, args.exclude_markers,
                              IMMUCAN_UNIPROT, emb_dir)

        label_map         = load_label_map(args.data_dir)
        folds, all_images = load_folds(args.data_dir, args.n_folds)

        all_feats_arr, metadata_all = load_embeddings(args.output_dir)
        if all_feats_arr is not None and args.mode in ('supervised','leiden'):
            img_feature_store = rebuild_img_feature_store(args.output_dir, all_images)
        else:
            img_feature_store, all_feats_arr, metadata_all = \
                extract_features_immucan(
                    args, model, embedding_dim, conf,
                    all_images, clean_indices, clean_names,
                    channel_mask, marker_indices, label_map)

        if args.mode in ('supervised','all'):
            run_supervised(args, folds, img_feature_store,
                           all_feats_arr, metadata_all,
                           method_name='VIRTUES_supervised')
        if args.mode in ('leiden','all'):
            run_leiden(args, folds, img_feature_store,
                       all_feats_arr, metadata_all,
                       method_name='VIRTUES_leiden')

    else:  # chl
        clean_indices, clean_names, channel_mask, marker_indices, _ = \
            get_clean_markers(CHL_ALL_CHANNELS, CHL_DEFAULT_EXCLUDE,
                              CHL_UNIPROT, emb_dir)
        # cHL runs extract+RF in one go (cell-level folds, no image-level store needed)
        extract_features_chl(args, model, embedding_dim, conf,
                              clean_indices, channel_mask, marker_indices)

    print(f"\nTotal: {(time.time()-total_start)/60:.1f} min")


if __name__ == '__main__':
    main()