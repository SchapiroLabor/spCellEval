#!/usr/bin/env python3
"""
prepare_dataset.py  —  dataset-agnostic CellSighter preprocessing
==================================================================
One prepare script for every dataset. All dataset specifics
come from src/methods/configs/cellsighter.json, selected with --dataset:

    python prepare_dataset.py --dataset immucan --config <...>/cellsighter.json
    python prepare_dataset.py --dataset chl     --config <...>/cellsighter.json

Produces CellSighter's expected layout under <output_root>:
    CellTypes/data/images/<sid>.npz     channel-filtered image (H,W,C), key 'data'
    CellTypes/cells/<sid>.npz           integer segmentation mask, key 'data'
    CellTypes/cells2labels/<sid>.txt    ONE label per line, length seg.max()+1
    CellTypes/channels.txt              kept channel names, one per line
    CellTypes/folds.json                IMAGE-LEVEL folds
    CellTypes/label_map.json            {class_name: int}

THE label-file convention 
----------------------------------------------------------------
CellSighter does `cl2lbl[cell_id]` with cell_id running 1..seg.max()
(see data/utils.py). The canonical file therefore has length seg.max()+1:
index 0 is a background placeholder, indices 1..N hold cell ids 1..N.

The existing IMMUcan cells2labels/<sid>.txt files are in the FOUNDATION-MODEL
convention (length == seg.max(); line k 0-indexed = cell k+1). Reused verbatim
they shift every label by one and drop the highest id. We fix this by prepending
a single -1 placeholder line (verified to round-trip through CellSighter's loader).

Two label sources, chosen per dataset via "label_source":
  * "cells2labels": reuse the existing per-cell label files + prepend -1.
                    label map comes from labels_cell_type.csv (phenotype,label).
  * "quant_csv":    build labels from a quantification CSV keyed by
                    (sample_id, cell_id) with a string cell_type column.

"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io


# CONFIG 
def load_config(config_path: str, dataset: str) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    if dataset not in cfg["datasets"]:
        raise ValueError(f"Dataset '{dataset}' not in {config_path}. "
                         f"Available: {list(cfg['datasets'])}")
    merged = dict(cfg.get("defaults", {}))
    merged.update(cfg["datasets"][dataset])
    merged["method"] = cfg.get("method", "cellsighter")
    merged["dataset"] = dataset
    return merged


# CHANNELS 
def get_channel_info(channels_source: str, excluded: list):
    with open(channels_source) as f:
        all_names = [ln.strip() for ln in f if ln.strip()]
    excluded = list(excluded or [])
    unknown = [c for c in excluded if c not in all_names]
    if unknown:
        print(f"  [!] excluded_channels not found in channels.txt (ignored): {unknown}")
    excl = set(excluded)
    kept = [(i, n) for i, n in enumerate(all_names) if n not in excl]
    print(f"  channel names ({len(all_names)} total): {all_names}")
    return [n for _, n in kept], [i for i, _ in kept]


# IMAGE (npz C,H,W / npz H,W,C / tiff) 
def load_raw_image(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=True)
        img = d["data"] if "data" in d else d[list(d.keys())[0]]
    else:
        img = io.imread(str(path))
    if img.ndim == 3 and int(np.argmin(img.shape)) == 0:   # C,H,W -> H,W,C
        img = np.moveaxis(img, 0, -1)
    return img


def process_image(raw_path: Path, keep_idx: list, out_path: Path):
    img = load_raw_image(raw_path)[..., keep_idx]
    np.savez_compressed(out_path, data=img.astype(np.float32))


def process_mask(seg_path: Path, out_path: Path) -> int:
    seg = io.imread(str(seg_path)).astype(np.int32)
    np.savez_compressed(out_path, data=seg)
    return int(seg.max())


# LABELS - source A: reuse existing files, prepend -1 
def cells2labels_from_existing(existing_txt: Path, n_cells: int, out_path: Path,
                               remap_to_neg=None) -> None:
    remap = set(remap_to_neg or [])
    vals = [int(float(x)) for x in existing_txt.read_text().strip().split("\n") if x != ""]
    if remap:
        vals = [-1 if v in remap else v for v in vals]
    if len(vals) == n_cells + 1:
        labels = vals
    elif len(vals) == n_cells:
        labels = [-1] + vals
    else:
        per_cell = (vals + [-1] * n_cells)[:n_cells]
        labels = [-1] + per_cell
        print(f"  [!] {existing_txt.name}: len {len(vals)} != seg.max() {n_cells}; padded.")
    assert len(labels) == n_cells + 1
    out_path.write_text("\n".join(str(v) for v in labels))


def label_map_from_csv(labels_csv: Path, excluded=None) -> dict:
    excluded = set(excluded or [])
    df = pd.read_csv(labels_csv)
    return {str(p): int(l) for p, l in zip(df["phenotype"], df["label"])
            if int(l) >= 0 and str(p) not in excluded}


# LABELS - source B: build from quantification CSV 
def build_label_map_from_quant(quant_df, cell_type_col, excluded):
    excluded = set(excluded or [])
    classes = sorted(c for c in quant_df[cell_type_col].astype(str).unique() if c not in excluded)
    str2int = {c: i for i, c in enumerate(classes)}
    for c in excluded:
        str2int[c] = -1
    return str2int


def cells2labels_from_quant(sample_df, n_cells, str2int, cell_id_col, cell_type_col, out_path):
    df = sample_df.copy()
    df["label_id"] = df[cell_type_col].astype(str).map(str2int).fillna(-1).astype(int)
    per_cell = (df.set_index(cell_id_col)["label_id"]
                  .reindex(range(1, n_cells + 1), fill_value=-1).astype(int).tolist())
    labels = [-1] + per_cell
    assert len(labels) == n_cells + 1
    out_path.write_text("\n".join(str(v) for v in labels))


# FOLDS (always image-level) 
def load_image_level_folds(cfg, quant_df=None) -> dict:
    with open(cfg["folds_path"]) as f:
        raw = json.load(f)
    if cfg.get("folds_level", "image") == "image":
        test_keys = sorted(k for k in raw if k.endswith("_test_set"))
        if test_keys:
            return {str(i): list(raw[k]) for i, k in enumerate(test_keys)}
        return {str(k): list(v) for k, v in raw.items()}
    print("  [!] folds_level='cell': collapsing per-cell folds to image level.")
    sid_col = cfg["sample_id_col"]
    cell_fold = {int(i): str(fid) for fid, idxs in raw.items() for i in idxs}
    q = quant_df.copy()
    q["_fold"] = q.index.map(lambda i: cell_fold.get(int(i)))
    img_fold = (q.dropna(subset=["_fold"]).groupby(sid_col)["_fold"]
                 .agg(lambda s: s.value_counts().idxmax()))
    folds = {}
    for sid, fid in img_fold.items():
        folds.setdefault(str(fid), []).append(str(sid))
    return folds


# MAIN 
def find_file(directory: Path, sid: str, exts: list):
    for ext in exts:
        c = directory / f"{sid}.{ext}"
        if c.exists():
            return c
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Dataset-agnostic CellSighter preprocessing")
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.dataset)
    out_root = Path(cfg["output_root"])
    out_ct = out_root / "CellTypes"
    out_img, out_seg, out_c2l = out_ct / "data" / "images", out_ct / "cells", out_ct / "cells2labels"
    for d in (out_img, out_seg, out_c2l):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Preparing '{args.dataset}' -> {out_root}  (label_source={cfg['label_source']})")

    keep_names, keep_idx = get_channel_info(cfg["channels_source"], cfg["excluded_channels"])
    (out_ct / "channels.txt").write_text("\n".join(keep_names))
    print(f"  channels: {len(keep_names)} kept")

    quant_df = None
    excluded_ids = set()
    if cfg["label_source"] == "quant_csv":
        quant_df = pd.read_csv(cfg["quant_csv"])
        quant_df[cfg["cell_type_col"]] = quant_df[cfg["cell_type_col"]].astype(str)
        str2int = build_label_map_from_quant(quant_df, cfg["cell_type_col"], cfg["excluded_cell_types"])
        label_map = {c: i for c, i in str2int.items() if i >= 0}
        by_sample = {sid: df for sid, df in quant_df.groupby(cfg["sample_id_col"])}
    else:
        excluded_names = cfg.get("excluded_cell_types", [])
        label_map = label_map_from_csv(Path(cfg["labels_csv"]), excluded_names)
        full = pd.read_csv(cfg["labels_csv"])
        excluded_ids = {int(l) for p, l in zip(full["phenotype"], full["label"])
                        if str(p) in set(excluded_names) and int(l) >= 0}
        if excluded_ids:
            print(f"  excluding cell types {excluded_names} -> ids {sorted(excluded_ids)} remapped to -1")
    json.dump(label_map, open(out_ct / "label_map.json", "w"), indent=2)
    print(f"  {len(label_map)} classes: {sorted(label_map, key=label_map.get)}")

    folds = load_image_level_folds(cfg, quant_df)
    json.dump(folds, open(out_ct / "folds.json", "w"), indent=2)
    sample_ids = sorted({s for v in folds.values() for s in v})
    print(f"  {len(folds)} folds, {len(sample_ids)} images")

    n_done = 0
    for sid in sample_ids:
        raw_img = find_file(Path(cfg["raw_image_dir"]), sid, cfg["raw_image_exts"])
        raw_seg = find_file(Path(cfg["segmentation_dir"]), sid, cfg["segmentation_exts"])
        if raw_img is None or raw_seg is None:
            print(f"  [!] missing image/seg for {sid}, skipping")
            continue
        process_image(raw_img, keep_idx, out_img / f"{sid}.npz")
        n_cells = process_mask(raw_seg, out_seg / f"{sid}.npz")

        if cfg["label_source"] == "quant_csv":
            sdf = by_sample.get(sid, pd.DataFrame(columns=quant_df.columns))
            cells2labels_from_quant(sdf, n_cells, str2int, cfg["cell_id_col"],
                                    cfg["cell_type_col"], out_c2l / f"{sid}.txt")
        else:
            existing = find_file(Path(cfg["existing_c2l_dir"]), sid, ["txt"])
            if existing is None:
                print(f"  [!] no existing cells2labels for {sid}, skipping")
                continue
            cells2labels_from_existing(existing, n_cells, out_c2l / f"{sid}.txt",
                                       remap_to_neg=excluded_ids)
        n_done += 1

    print(f"\nDone. {n_done}/{len(sample_ids)} samples written to {out_root}")
    print(f"  num_classes = {len(label_map)}")


if __name__ == "__main__":
    main()