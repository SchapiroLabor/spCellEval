#!/usr/bin/env python3
"""
prepare_immucan.py  —  IMMUcan preprocessing for CellSighter

Prepares the IMMUcan dataset in CellSighter-compatible format under
IMMUcan_CellSighter/. Must be run once before run_cellsighter.py.

Key design decisions:
  - Raw tiff images (CxHxW) are transposed to (HxWxC) and saved as npz
  - Non-protein channels (DNA1, DNA2, HistoneH3) are excluded by name
  - Cell labels come from IMMUcan_quantification.csv 
  - Missing cell IDs (segmentation gaps) are filled with -1 via reindexing
    to range(1, seg.max()+1) — this fixes CellSighter's ndimage.find_objects()
  - 'unlabelled' cells map to -1 and are excluded from training/eval
  - Original sample IDs and folds.json are preserved unchanged

"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from skimage import io

# CONFIGURATION

DATA_ROOT = Path("/home/juliaoesterle/data/phenotyping_benchmark/IMMUcan")
OUT_DIR   = Path("/home/juliaoesterle/data/phenotyping_benchmark/IMMUcan_CellSighter")

RAW_DIR   = DATA_ROOT / "raw_images" / "multistack_tiffs"
SEG_DIR   = DATA_ROOT / "segmentation"
QUANT_CSV = DATA_ROOT / "quantification" / "processed" / "IMMUcan_quantification.csv"
FOLDS_JSON= DATA_ROOT / "CellTypes" / "folds.json"
MARKERS   = DATA_ROOT / "markers.txt"

CELL_TYPE_COL    = "cell_type"          # level 3 granularity
EXCLUDE_TYPES    = {"unlabelled", "undefined"}  # → -1
EXCLUDE_CHANNELS = {"DNA1", "DNA2", "HistoneH3"}  # nuclear/structural stains


#Build label map from quantification CSV

def build_label_map(quant_df: pd.DataFrame) -> tuple[dict, dict, int]:
    """
    Returns:
        str2int : cell_type string → integer label
        int2str : integer label → cell_type string
        n_classes : number of valid classes (excluding unlabelled)
    """
    valid_types = sorted(
        t for t in quant_df[CELL_TYPE_COL].unique()
        if t not in EXCLUDE_TYPES
    )
    str2int   = {t: i for i, t in enumerate(valid_types)}
    int2str   = {i: t for t, i in str2int.items()}
    n_classes = len(valid_types)
    return str2int, int2str, n_classes


#Resolve channel indices to keep

def get_channel_info(markers_path: Path) -> tuple[list[str], list[int]]:
    """
    Read markers.txt and return (kept_channel_names, kept_indices).
    Excludes EXCLUDE_CHANNELS by name.
    """
    with open(markers_path) as f:
        all_channels = [l.strip() for l in f if l.strip()]
    keep_names = [c for c in all_channels if c not in EXCLUDE_CHANNELS]
    keep_idx   = [i for i, c in enumerate(all_channels) if c not in EXCLUDE_CHANNELS]
    return keep_names, keep_idx


#Process one image: transpose + channel filter + save

def process_image(raw_path: Path, keep_idx: list[int], out_path: Path) -> None:
    img = io.imread(str(raw_path))          # (C, H, W) uint16
    img = img[keep_idx].astype(np.float32)  # (C_kept, H, W)
    img = np.transpose(img, (1, 2, 0))      # (H, W, C_kept)
    np.savez_compressed(out_path, data=img)

# Process one mask: save as int32 npz, return n_cells

def process_mask(sample_id: str, out_path: Path) -> int:
    """
    Find segmentation mask for sample_id (handles .tif / .tiff),
    save as int32 npz, return number of cells (seg.max()).
    """
    for ext in ("tiff", "tif"):
        candidate = SEG_DIR / f"{sample_id}.{ext}"
        if candidate.exists():
            seg = io.imread(str(candidate)).astype(np.int32)
            np.savez_compressed(out_path, data=seg)
            return int(seg.max())
    raise FileNotFoundError(f"No segmentation mask found for {sample_id}")

#Build cells2labels for one image (Marcel's reindexing fix)

def build_cells2labels(
    sample_id: str,
    sample_df: pd.DataFrame,
    n_cells: int,
    str2int: dict,
    out_path: Path,
) -> None:
    """
    Create a gap-free label file with exactly n_cells entries (one per line).
    Cell IDs not present in the quant CSV (segmentation gaps) → -1.
    Excluded cell types ('unlabelled') → -1.

    This is the critical fix: ndimage.find_objects() requires continuous
    integer cell IDs from 1 to n_cells with no gaps.
    """
    # Map cell_type string → integer (-1 for excluded)
    sample_df = sample_df.copy()
    sample_df["label_id"] = (
        sample_df[CELL_TYPE_COL]
        .map(str2int)
        .fillna(-1)
        .astype(int)
    )

    # Reindex to full range 1..n_cells — fills missing cell IDs with -1
    full_index = range(1, n_cells + 1)
    labels = (
        sample_df.set_index("cell_id")["label_id"]
        .reindex(full_index, fill_value=-1)
        .astype(int)
    )

    with open(out_path, "w") as f:
        f.write("\n".join(str(l) for l in labels))

# MAIN

def main():
    # Output directories 
    out_ct  = OUT_DIR / "CellTypes"
    out_img = out_ct / "data" / "images"
    out_seg = out_ct / "cells"
    out_c2l = out_ct / "cells2labels"
    for d in [out_img, out_seg, out_c2l]:
        d.mkdir(parents=True, exist_ok=True)

    # Load inputs 
    print("Loading quantification CSV...")
    quant_df = pd.read_csv(QUANT_CSV)
    quant_df[CELL_TYPE_COL] = quant_df[CELL_TYPE_COL].astype(str)
    print(f"  {len(quant_df):,} cells across {quant_df['sample_id'].nunique()} samples")

    str2int, int2str, n_classes = build_label_map(quant_df)
    print(f"  {n_classes} classes: {list(str2int.keys())}")

    keep_names, keep_idx = get_channel_info(MARKERS)
    print(f"  Channels: {len(keep_names)} kept, {len(EXCLUDE_CHANNELS)} excluded")

    with open(FOLDS_JSON) as f:
        folds = json.load(f)
    all_sample_ids = sorted(set(
        sid for key in folds for sid in folds[key]
    ))
    print(f"  {len(all_sample_ids)} samples from folds.json")

    #Process each sample 
    print("\nProcessing samples...")
    for sample_id in all_sample_ids:
        # Find raw image (may be .tif or .tiff)
        raw_path = None
        for ext in ("tiff", "tif"):
            candidate = RAW_DIR / f"{sample_id}.{ext}"
            if candidate.exists():
                raw_path = candidate
                break
        if raw_path is None:
            print(f"  [!] No image for {sample_id}, skipping")
            continue

        # Image
        process_image(raw_path, keep_idx, out_img / f"{sample_id}.npz")

        # Mask
        try:
            n_cells = process_mask(sample_id, out_seg / f"{sample_id}.npz")
        except FileNotFoundError as e:
            print(f"  [!] {e}, skipping")
            continue

        # Labels
        sample_df = quant_df[quant_df["sample_id"] == sample_id]
        build_cells2labels(sample_id, sample_df, n_cells, str2int,
                           out_c2l / f"{sample_id}.txt")

        print(f"  {sample_id}: {n_cells} cells")

    # Save label map and channels 
    pd.DataFrame({"phenotype": list(str2int.keys()),
                  "label":     list(str2int.values())}
    ).to_csv(OUT_DIR / "labels_cell_type.csv", index=False)

    with open(OUT_DIR / "channels.txt", "w") as f:
        f.write("\n".join(keep_names))

    # Copy folds.json unchanged
    import shutil
    shutil.copy(FOLDS_JSON, OUT_DIR / "folds.json")

    # Summary 
    print(f"\n{'='*60}")
    print(f"Done! → {OUT_DIR}")
    print(f"  Images:      {len(list(out_img.glob('*.npz')))}")
    print(f"  Masks:       {len(list(out_seg.glob('*.npz')))}")
    print(f"  Labels:      {len(list(out_c2l.glob('*.txt')))}")
    print(f"  Classes:     {n_classes}")
    print(f"  Channels:    {len(keep_names)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()