"""
benchmark_io.py — Shared data access for the config-driven benchmark scripts
(Eva, KRONOS, VirTues, KRONOS2, Stellar).

Expected dataset layout (<data_root> = .../<dataset>):
    <data_root>/markers.txt
    <data_root>/raw_images/multistack_tiffs/<image>[...].tif(f)
    <data_root>/segmentation/<image>.tif(f)  or  <data_root>/segmentation/<image>/**/segmentationMap.tif
    <data_root>/quantification/processed/<dataset>_quantification.csv   (sample_id, cell_id, cell_type)
    <folds_dir>/fold_<i>_{train,validation,test}.csv                    (sample_id, cell_id)

Functions
---------
load_config(config_path, dataset)
read_markers(data_root)
file_stem(name)
find_image(img_dir, img_id)
find_segmentation(seg_dir, img_id)
read_image(img_path, channel_indices=None)
read_segmentation(seg_path)
load_cell_folds(folds_dir, include_validation=False)
load_benchmark_data(data_root, folds_dir, exclude_labels=(), include_validation=False)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


def load_config(config_path, dataset):
    """Merge a config's 'defaults' with its block for `dataset`; unknown datasets raise."""
    with open(config_path) as f:
        cfg = json.load(f)
    datasets = cfg.get("datasets", {})
    if dataset not in datasets:
        raise ValueError(f"Dataset '{dataset}' not found in {config_path}. "
                         f"Available: {sorted(datasets)}")
    merged = dict(cfg.get("defaults", {}))
    merged.update(datasets[dataset])
    merged["dataset"] = dataset
    return merged


def read_markers(data_root):
    with open(Path(data_root) / "markers.txt") as f:
        return [line.strip() for line in f if line.strip()]


def file_stem(name):
    for ext in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def find_image(img_dir, img_id):
    files = [p for p in Path(img_dir).iterdir() if not p.name.startswith(".")]
    exact = [p for p in files if file_stem(p.name) == img_id]
    if exact:
        return exact[0]
    prefixed = [p for p in files if file_stem(p.name).startswith(f"{img_id}_")]
    return prefixed[0] if len(prefixed) == 1 else None


def find_segmentation(seg_dir, img_id):
    seg_dir = Path(seg_dir)
    for ext in (".tiff", ".tif"):
        if (seg_dir / f"{img_id}{ext}").exists():
            return seg_dir / f"{img_id}{ext}"
    if (seg_dir / img_id).is_dir():
        seg_files = sorted(p for p in (seg_dir / img_id).rglob("segmentationMap.tif")
                           if not p.name.startswith("."))
        if seg_files:
            return seg_files[0]
    return None


def read_image(img_path, channel_indices=None):
    img = tifffile.imread(str(img_path)).astype(np.float32)
    return img if channel_indices is None else img[channel_indices]


def read_segmentation(seg_path):
    return tifffile.imread(str(seg_path)).astype(np.int32)


def _normalize_ids(series):
    return series.astype(str).str.replace(".csv", "", regex=False)


def _read_fold_keys(path):
    df = pd.read_csv(path, usecols=["sample_id", "cell_id"])
    return set(zip(_normalize_ids(df["sample_id"]), df["cell_id"]))


def _check_split(fold_i, train, validation, test, folds_dir):
    shared_cells = (train | validation) & test
    if shared_cells:
        raise ValueError(f"{folds_dir}: fold {fold_i} has {len(shared_cells):,} cells "
                         f"in both training and test")
    train_imgs = {img for img, _ in train}
    val_imgs = {img for img, _ in validation}
    test_imgs = {img for img, _ in test}
    # Image-disjoint train/test means a group split; validation must then not reach into test images.
    leaked = val_imgs & test_imgs
    if leaked and not (train_imgs & test_imgs):
        n_cells = sum(1 for img, _ in validation if img in leaked)
        raise ValueError(f"{folds_dir}: fold {fold_i} validation contains {n_cells:,} cells "
                         f"from {len(leaked)} test images; this is not the canonical split")


def load_cell_folds(folds_dir, include_validation=False):
    """
    Read fold_<i>_{train,validation,test}.csv into [{'train': set, 'test': set}, ...]
    of (image_id, cell_id) keys. With include_validation the validation cells are
    added to training. Raises if training (incl. validation) overlaps the test set.
    """
    folds_dir = Path(folds_dir)
    n_folds = 0
    while (folds_dir / f"fold_{n_folds + 1}_train.csv").exists():
        n_folds += 1
    if n_folds == 0:
        raise FileNotFoundError(f"No fold_<i>_train.csv files in {folds_dir}")

    folds = []
    for i in range(1, n_folds + 1):
        train = _read_fold_keys(folds_dir / f"fold_{i}_train.csv")
        test = _read_fold_keys(folds_dir / f"fold_{i}_test.csv")
        val_path = folds_dir / f"fold_{i}_validation.csv"
        validation = _read_fold_keys(val_path) if include_validation and val_path.exists() else set()
        _check_split(i, train, validation, test, folds_dir)
        folds.append({"train": train | validation, "test": test})
    print(f"Loaded {n_folds} folds from {folds_dir} "
          f"({'train+validation' if include_validation else 'train only'})")
    return folds


def load_benchmark_data(data_root, folds_dir, exclude_labels=(), include_validation=False):
    """
    Returns meta_df[image_id, cell_id, label], folds (see load_cell_folds; restricted
    to cells whose label is not excluded), img_paths, seg_paths.
    Ground truth is the quantification table's 'cell_type' column.
    """
    data_root = Path(data_root)
    quant_csv = data_root / "quantification" / "processed" / f"{data_root.name}_quantification.csv"
    meta_df = pd.read_csv(quant_csv, usecols=["sample_id", "cell_id", "cell_type"])
    meta_df = meta_df.rename(columns={"sample_id": "image_id", "cell_type": "label"})
    meta_df["image_id"] = _normalize_ids(meta_df["image_id"])

    folds = load_cell_folds(folds_dir, include_validation)
    all_keys = set(zip(meta_df["image_id"], meta_df["cell_id"]))
    for i, fold in enumerate(folds):
        unknown = (fold["train"] | fold["test"]) - all_keys
        if unknown:
            raise ValueError(f"Fold {i}: {len(unknown):,} fold cells are not in {quant_csv}")

    excluded = meta_df["label"].isin(set(exclude_labels))
    if excluded.any():
        dropped = set(zip(meta_df.loc[excluded, "image_id"], meta_df.loc[excluded, "cell_id"]))
        folds = [{split: keys - dropped for split, keys in fold.items()} for fold in folds]
        meta_df = meta_df[~excluded]
    meta_df = meta_df[["image_id", "cell_id", "label"]].reset_index(drop=True)
    print(f"Loaded metadata: {len(meta_df):,} cells "
          f"({int(excluded.sum()):,} excluded by label: {sorted(set(exclude_labels))})")

    img_dir = data_root / "raw_images" / "multistack_tiffs"
    seg_dir = data_root / "segmentation"
    img_paths, seg_paths = {}, {}
    for img_id in sorted(meta_df["image_id"].unique()):
        img_path = find_image(img_dir, img_id)
        if img_path is not None:
            img_paths[img_id] = img_path
        seg_path = find_segmentation(seg_dir, img_id)
        if seg_path is not None:
            seg_paths[img_id] = seg_path
    print(f"Found {len(img_paths)} images, {len(seg_paths)} segmentation maps")
    return meta_df, folds, img_paths, seg_paths
