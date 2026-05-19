import sys
sys.path.append(".")
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import argparse
import numpy as np
import pandas as pd
from model import Model
from data.data import CellCropsDataset
from data.utils import load_crops
from data.transform import train_transform, val_transform
from torch.utils.data import DataLoader, WeightedRandomSampler
import json
import time

torch.multiprocessing.set_sharing_strategy('file_system')


def test_epoch(model, dataloader, device=None):
    with torch.no_grad():
        model.eval()
        predicted_labels = []
        pred_probs = []
        results = {'cell_id': [], 'image_id': [], 'true_labels': [], 'predicted_labels': [], 'pred_probs': []}

        for i, batch in enumerate(dataloader):
            x = batch['image']
            m = batch.get('mask', None)
            if m is not None:
                x = torch.cat([x, m], dim=1)
            x = x.to(device=device)
            # m = m.to(device=device)
            y_pred = model(x)

            pred_probs += y_pred.detach().cpu().numpy().tolist()
            predicted_labels += y_pred.detach().cpu().numpy().argmax(1).tolist()

            results['cell_id'].extend(batch['cell_id'].detach().cpu().numpy().tolist())
            results['image_id'].extend(batch['image_id'])
            results['true_labels'].extend(batch['label'].detach().cpu().numpy().tolist())
            results['predicted_labels'].extend(np.argmax(y_pred.detach().cpu().numpy(), axis=1).tolist())
            results['pred_probs'].extend(y_pred.detach().cpu().numpy().tolist())

            print(f"Eval {i} / {len(dataloader)}        ", end='\r')
        return np.array(predicted_labels), np.array(pred_probs), pd.DataFrame.from_dict(results)

def results_reformatting(result_path, fold_id):
    class_ids = {"0": "0", "1": "1", "2": "1", "3": "2", "4": "3", "5": "4", "6": "5", "7": "6", "8": "7", "9": "9",
                 "10": "8", "11": "10", "12": "1", "13": "11"}
    class_names = {"0": "B", "1": "CD4 T", "2": "CD4 Treg", "3": "CD8 T", "4": "DC", "5": "Endothelial", "6": "M1",
                   "7": "M2", "8": "NK", "9": "Neutrophil", "10": "Other", "11": "Tumor"}
    col_names = ["%s_prob" % class_names[key] for key in class_names.keys()]

    res = pd.read_csv(os.path.join(result_path, args.cell_type_col, fold_id, 'results.csv'))

    true_labels = [int(class_ids[str(label)]) for label in res['true_labels'].tolist()]
    predicted_labels = [int(class_ids[str(label)]) for label in res['predicted_labels'].tolist()]
    probs = res['pred_probs'].to_numpy()
    prob_matrix = np.zeros((probs.shape[0], len(col_names)))
    for i in range(probs.shape[0]):
        class_prob = np.array([np.float64(val) for val in probs[i][1:-1].split(',')])
        for key in class_ids.keys():
            prob_matrix[i, int(class_ids[key])] += class_prob[int(class_ids[key])]

    df = pd.DataFrame(prob_matrix, columns=col_names)
    df['predicted_label'] = predicted_labels
    df['true_label'] = true_labels
    os.makedirs(os.path.join(result_path, args.cell_type_col, fold_id), exist_ok=True)
    df.to_csv(os.path.join(result_path, args.cell_type_col, fold_id, 'results_test.csv'), index=False)

def subsample_const_size(crops, size):
    """
    sample same number of cell from each class
    """
    final_crops = []
    crops = np.array(crops)
    labels = np.array([c._label for c in crops])
    for lbl in np.unique(labels):
        indices = np.argwhere(labels == lbl).flatten()
        if (labels == lbl).sum() < size:
            chosen_indices = indices
        else:
            chosen_indices = np.random.choice(indices, size, replace=False)
        final_crops += crops[chosen_indices].tolist()
    return final_crops


def define_sampler(crops, hierarchy_match=None):
    """
    Sampler that sample from each cell category equally
    The hierarchy_match defines the cell category for each class.
    if None then each class will be category of it's own.
    """
    labels = np.array([c._label for c in crops])
    if hierarchy_match is not None:
        labels = np.array([hierarchy_match[str(l)] for l in labels])

    unique_labels = np.unique(labels)
    class_sample_count = {t: len(np.where(labels == t)[0]) for t in unique_labels}
    weight = {k: sum(class_sample_count.values()) / v for k, v in class_sample_count.items()}
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)
    return WeightedRandomSampler(samples_weight.double(), len(samples_weight))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validation')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--root',type=str,required=True, help='Data to the folder where all cellsighter data and the required folder structure is saved')
    parser.add_argument('--data_root',type=str,required=True, help='Path to where tha raw data of the benchmark is saved')
    parser.add_argument('--fold_id', type=str, default='fold_0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--columns', type=str, nargs='+', default=None)
    parser.add_argument('--cell_type_col', type=str, default='cell_type')
    parser.add_argument('--append', action="store_true")
    args = parser.parse_args()

    # Paths derived from dataset
    base_path = os.path.join(args.root,"datasets", args.dataset)
    result_path = os.path.join(args.root,"results", args.dataset, args.cell_type_col)
    if args.output_path:
        output_path = args.output_path
        os.makedirs(os.path.join(output_path), exist_ok=True)
    else:
        output_path = result_path

    writer = SummaryWriter(log_dir=os.path.join(result_path, "logs", args.fold_id))

    # Load config
    start_time = time.time()
    config_path = os.path.join(base_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    num_channels = sum(1 for _ in open(config["channels_path"])) + 1 - len(config["blacklist"])
    class_num = config["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_channels + 1, class_num)
    weights_path = os.path.join(result_path, args.fold_id, 'weights.pth')
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device).eval()

    config["train_set"] = config[f"{args.fold_id}_train_set"]
    config["test_set"] = config[f"{args.fold_id}_test_set"]

    _, test_crops = load_crops(
        config["root_dir"],
        config["channels_path"],
        config["crop_size"],
        [],
        config["test_set"],
        config["to_pad"],
        blacklist_channels=config["blacklist"],
        cell_type_col=args.cell_type_col
    )
    test_crops = [c for c in test_crops if c._label >= 0]
    crop_input_size = config.get("crop_input_size", 100)
    test_dataset = CellCropsDataset(test_crops, transform=val_transform(crop_input_size), mask=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    predicted_labels, pred_probs, results_df = test_epoch(model, test_loader, device=device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    results_df = results_df.rename(columns={"image_id": "sample_id"})

    # Load the label mapping
    labels_df = pd.read_csv(os.path.join(base_path, f"labels_{args.cell_type_col}.csv"))
    label_map = dict(zip(labels_df['label'], labels_df['phenotype']))


    # Map integer labels to names
    results_df["true_phenotype"] = results_df["true_labels"].map(label_map)
    results_df["predicted_phenotype"] = results_df["predicted_labels"].map(label_map)

    # Save the updated results

    print(f" Saved validation results for {args.dataset} - {args.fold_id}")

    if args.output_path:
        output_dir = args.output_path
    else:
        output_dir = os.path.join(result_path, args.fold_id)

    os.makedirs(output_dir, exist_ok=True)

    # get Cell_type_level from dic

    cell_type_levels = {
    "level_1_cell_type": "level1",
    "level_2_cell_type": "level2",
    "cell_type": "level3"
}
    level = cell_type_levels.get(args.cell_type_col)

    # Save full results
    if args.append:
        quant_path = os.path.join(args.data_root,args.dataset, "quantification", "processed", f"{args.dataset}_quantification.csv")
        df_quant = pd.read_csv(quant_path)
        df_quant["sample_id"] = df_quant["sample_id"].str.replace(r"\.csv$", "", regex=True)

        if args.output_path:
            final_path = os.path.join(output_dir, args.dataset, "cellsighter", level)
            os.makedirs(final_path, exist_ok=True)
            df_quant = df_quant.merge(
                results_df[["sample_id", "cell_id", "true_phenotype", "predicted_phenotype"]],
                on=["sample_id", "cell_id"],
                how="left"
)
            df_quant[args.cell_type_col] = df_quant["true_phenotype"]
            df_quant = df_quant.rename(columns={"true_phenotype": "true_phenotype_appended"})
            df_quant = df_quant.rename(columns={args.cell_type_col: "true_phenotype"})
            df_quant = df_quant.drop(columns=["true_phenotype_appended"])
            results_df = df_quant.dropna(subset=["true_phenotype"])
            results_df.to_csv(os.path.join(final_path, f"predictions_{args.fold_id}.csv"), na_rep="NaN")
            with open(os.path.join(final_path, f"val_time_{args.fold_id}.txt"), "w") as f:
                f.write(f"{args.fold_id} training_time: {elapsed_time:.2f}\n")
    else:
        if args.output_path:
            final_path = os.path.join(output_dir, args.dataset, "Cell_Sighter", level)
            os.makedirs(final_path, exist_ok=True)
            results_df.to_csv(os.path.join(final_path, f"predictions_{args.fold_id}.csv"), na_rep="NaN")
            with open(os.path.join(final_path, f"val_time_{args.fold_id}.txt"), "w") as f:
                f.write(f"{args.fold_id} training_time: {elapsed_time:.2f}\n")
        else:
            results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False, na_rep="NaN")
            with open(os.path.join(output_dir, f"val_time_{args.fold_id}.txt"), "w") as f:
                f.write(f"{args.fold_id} training_time: {elapsed_time:.2f}\n")


    # Optionally save selected columns
    if args.columns:
        missing_cols = [col for col in args.columns if col not in results_df.columns]
        if missing_cols:
            raise ValueError(f"Requested columns not in DataFrame: {missing_cols}")
        elif args.output_path:
            results_df[args.columns].to_csv(os.path.join(final_path, f"predictions_{args.fold_id}_selected.csv"), index=False, na_rep="NaN")
            print(f"Saved selected columns → {final_path}/predictions_{args.fold_id}_selected.csv")
        elif not args.output_path:
            results_df[args.columns].to_csv(os.path.join(output_dir, "results_selected.csv"), index=False, na_rep="NaN")
            print(f"Saved selected columns → {os.path.join(output_dir, 'results_selected.csv')}")
