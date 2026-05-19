import os
import json
import argparse
import numpy as np
import pandas as pd
from skimage import io
import shutil
from sklearn.model_selection import KFold

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--working_dir", type=str, required=True)
    parser.add_argument("--cell_type_col", nargs='+', default=[])
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--exclude_celltypes", nargs='*', default=[])

    # Configuration file
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--swap_train_test", action="store_true")
    parser.add_argument("--split", type=float, default=None)
    parser.add_argument("--crop_input_size", type=int, default=60)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--epoch_max", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sample_batch", action="store_true")
    parser.add_argument("--to_pad", action="store_true")
    parser.add_argument("--aug_data", action="store_true")
    parser.add_argument("--folds_json", type=str, default="")

    return parser.parse_args()

def process_images(raw_dir, output_path, transpose):
    sample_ids = []
    sample_ids_clean = []
    os.makedirs(output_path, exist_ok=True)

    for fname in os.listdir(raw_dir):
        if fname.endswith((".tif", ".tiff", ".npz")):
            sample_id = fname.rsplit(".", 1)[0].replace("_stacked.ome", "")
            sample_ids.append(fname.rsplit(".", 1)[0])
            sample_ids_clean.append(sample_id)

            img_path = os.path.join(raw_dir, fname)
            img = io.imread(img_path)
            if transpose:
                img_t = img.transpose(1, 2, 0)
                np.savez(os.path.join(output_path, f"{sample_id}.npz"), data=img_t)
            else:
                np.savez(os.path.join(output_path, f"{sample_id}.npz"), data=img)
            print(f"Saved image {sample_id}")

    return sample_ids_clean

def process_masks(seg_dir, output_path, sample_ids):
    os.makedirs(output_path, exist_ok=True)
    n_cells_dict = {}

    for sample_id in sample_ids:
        seg_path = None
        for ext in ("tif", "tiff", "npz"):
            candidate = os.path.join(seg_dir, f"{sample_id}.{ext}")
            if os.path.exists(candidate):
                seg_path = candidate
                break
        if not seg_path:
            nested = os.path.join(seg_dir, sample_id, "H3_memSUM_noCD163_deepcell060_AutoHist_mpp1.75", "segmentationMap.tif")
            if os.path.exists(nested):
                seg_path = nested

        if not seg_path:
            print(f"[!] Segmentation not found for {sample_id}")
            continue

        seg_data = np.load(seg_path)["data"] if seg_path.endswith(".npz") else io.imread(seg_path)
        n_cells_dict[sample_id] = int(seg_data.max())
        np.savez(os.path.join(output_path, f"{sample_id}.npz"), data=seg_data)
        print(f"Saved mask for {sample_id} with {n_cells_dict[sample_id]} cells")

    return n_cells_dict

def process_labels(quant_path, label_path, output_path, cell_type_col, exclude_celltypes, n_cells_dict=None):
    df = pd.read_csv(quant_path)
    df[cell_type_col] = df[cell_type_col].astype(str)

    unique_phenos = pd.Series(df[cell_type_col].unique())
    if exclude_celltypes:
        unique_phenos = unique_phenos[~unique_phenos.isin(exclude_celltypes)]

    df_label_map = pd.DataFrame(unique_phenos.sort_values().reset_index(drop=True), columns=["phenotype"])
    df_label_map["label"] = df_label_map.index
    df = df.merge(df_label_map, how="left", left_on=cell_type_col, right_on="phenotype")
    df["label_id"] = df["label"].fillna(-1).astype(int)

    os.makedirs(output_path, exist_ok=True)
    for sample_id, group in df.groupby("sample_id"):
        clean_id = sample_id.rsplit(".", 1)[0]
        group = group.sort_values("cell_id").reset_index(drop=True)

        expected_n = n_cells_dict.get(clean_id, group["cell_id"].max())
        full_ids = list(range(1, expected_n + 1))

        group = group.set_index("cell_id").reindex(full_ids, fill_value=np.nan).reset_index()
        group["label_id"] = group["label_id"].fillna(-1).astype(int)

        path = os.path.join(output_path, f"{clean_id}.txt")
        group["label_id"].to_csv(path, index=False, header=False)
        print(f"Labels saved for {clean_id}, lvl: {cell_type_col}")

    return df_label_map

def create_kfold_splits(sample_ids, num_folds, swap_train_test):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    folds = {}
    sample_ids_sorted = sorted(sample_ids)
    for i, (train_idx, test_idx) in enumerate(kf.split(sample_ids_sorted)):

        if swap_train_test:
            folds[f"fold_{i}_train_set"] = [sample_ids_sorted[j] for j in test_idx]
            folds[f"fold_{i}_test_set"] = [sample_ids_sorted[j] for j in train_idx]
        else:
            folds[f"fold_{i}_train_set"] = [sample_ids_sorted[j] for j in train_idx]
            folds[f"fold_{i}_test_set"] = [sample_ids_sorted[j] for j in test_idx]
    return folds

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

def write_config(output_path, num_classes, crop_input_size, crop_size, epoch_max, lr, sample_batch, to_pad, aug_data, channel_path, folds):
    config = {
        "crop_input_size": crop_input_size,
        "crop_size": crop_size,
        "root_dir": output_path,
        "num_classes": num_classes,
        "epoch_max": epoch_max,
        "lr": lr,
        "blacklist": [],
        "channels_path": channel_path,
        "weight_to_eval": "",
        "sample_batch": sample_batch,
        "to_pad": to_pad,
        "aug_data": aug_data,
    }
    config.update(folds)
    config = convert_numpy(config)

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print(f"Config saved → {os.path.join(output_path, 'config.json')}")

def main():
    args = parse_args()

    output_root = os.path.join(args.working_dir, "datasets", args.dataset_name, "CellTypes")
    raw_dir = os.path.join(args.data_root, "raw_images", "multistack_tiffs")
    quant_path = os.path.join(args.data_root, "quantification", "processed", f"{args.dataset_name}_quantification.csv")
    label_path = os.path.join(args.data_root, "quantification", "processed", "labels.csv")
    seg_dir = os.path.join(args.data_root, "segmentation")
    channel_path = os.path.join(args.data_root, "markers.txt")

    sample_ids = process_images(raw_dir, os.path.join(output_root, "data", "images"), args.transpose)
    n_cells_dict = process_masks(seg_dir, os.path.join(output_root, "cells"), sample_ids)

    for cell_type_col in args.cell_type_col:
        df_label_map = process_labels(
            quant_path,
            label_path,
            os.path.join(output_root, "cells2labels", cell_type_col),
            cell_type_col,
            args.exclude_celltypes,
            n_cells_dict
        )
        label_out = os.path.join(output_root, "..", f"labels_{cell_type_col}.csv")
        if not os.path.exists(label_out):
            df_label_map.to_csv(label_out, index=False)

    channel_out = os.path.join(output_root, "..", "channels.txt")
    if not os.path.exists(channel_out):
        shutil.copyfile(channel_path, channel_out)

    if args.split:
        folds = create_kfold_splits(sample_ids, args.num_folds, args.swap_train_test)
        with open(os.path.join(output_root, "..", "folds.json"), "w") as f:
            json.dump(folds, f, indent=4)
        print("Folds saved")
    else:
        with open(args.folds_json, "r") as f:
            folds = json.load(f)

    num_classes = df_label_map["label"].max() + 1
    write_config(os.path.join(output_root, ".."), num_classes, args.crop_input_size, args.crop_size,
                 args.epoch_max, args.lr, args.sample_batch, args.to_pad, args.aug_data,
                 channel_out, folds)

if __name__ == "__main__":
    main()
