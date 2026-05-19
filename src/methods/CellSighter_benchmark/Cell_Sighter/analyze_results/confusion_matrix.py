import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os


def metric(gt, pred, classes_for_cm, save_path, colorbar=True):
    sns.set(font_scale=2)
    cm_normed_recall = confusion_matrix(gt, pred, labels=classes_for_cm, normalize="true") * 100
    cm = confusion_matrix(gt, pred, labels=classes_for_cm)

    plt.figure(figsize=(50, 45))
    ax1 = plt.subplot2grid((50, 50), (0, 0), colspan=30, rowspan=30)
    cmap = LinearSegmentedColormap.from_list('', ['white', *plt.cm.Blues(np.arange(255))])
    annot_labels = cm_normed_recall.round(1).astype(str)
    annot_labels = pd.DataFrame(annot_labels) + "\n (" + pd.DataFrame(cm).astype(str) + ")"

    annot_mask = cm_normed_recall.round(1) <= 0.1
    annot_labels[annot_mask] = ""

    sns.heatmap(cm_normed_recall.T, ax=ax1, annot=annot_labels.T, fmt='', cbar=colorbar,
                cmap=cmap, linewidths=1, vmin=0, vmax=100, linecolor='black', square=True)

    ax1.xaxis.tick_top()
    ax1.set_xticklabels(classes_for_cm, rotation=90)
    ax1.set_yticklabels(classes_for_cm, rotation=0)
    ax1.tick_params(axis='both', which='major', labelsize=35)

    ax1.set_xlabel("Clustering and gating", fontsize=35)
    ax1.set_ylabel("CellSighter", fontsize=35)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a confusion matrix for one fold.")
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g. IMMUcan, cHL_2_MIBI)')
    parser.add_argument('--fold_id', required=True, help='Fold name (e.g. fold_0, fold_1, ..., fold_4)')
    parser.add_argument('--input_path', type=str, default="")
    parser.add_argument('--cell_type_col', type=str, default="cell_type")
    args = parser.parse_args()

    # get Cell_type_level from dic

    cell_type_levels = {
    "level_1_cell_type": "level1",
    "level_2_cell_type": "level2",
    "cell_type": "level3"
}
    level = cell_type_levels.get(args.cell_type_col)

    if args.input_path:
        fold_dir = os.path.join(args.input_path, args.dataset, "Cell_Sighter", level)
        result_file = os.path.join(fold_dir,f"predictions_{args.fold_id}.csv")
        output_file = os.path.join(fold_dir, f"confusion_matrix_{args.fold_id}.png")
    else:
        fold_dir = os.path.join("results", args.dataset, args.cell_type_col, args.fold_id)
        result_file = os.path.join(fold_dir, "results.csv")
        output_file = os.path.join(fold_dir, "confusion_matrix.png")

    if not os.path.isfile(result_file):
        print(f"Results file not found: {result_file}")
        exit(1)

    results = pd.read_csv(result_file)
    results = results.dropna(subset=["true_phenotypes", "predicted_phenotypes"])

    classes_for_cm = np.unique(np.concatenate([results["true_phenotypes"], results["predicted_phenotypes"]]))
    metric(results["true_phenotypes"], results["predicted_phenotypes"], classes_for_cm, output_file)
    print(f"Saved confusion matrix to {output_file}")
