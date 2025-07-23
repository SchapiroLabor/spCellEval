import scyan
import anndata as ad
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import random
import argparse
import os
import time


def run_scyan(
    dataset_path,
    seed,
    n_runs,
    granularity_level,
    remove_columns,
    remove_cell_types,
    preprocess,
    decision_matrix_path,
    accelerator,
    split_col,
    scaling,
    log1p,
    batch_key,
    prior_std,
    patience,
    remove_result_cell_types,
    output_path,
    save_results,
):

    random.seed(seed)
    np.random.seed(seed)

    data = pd.read_csv(dataset_path)

    target_col = granularity_level
    if target_col == "level3":
        target_col = "cell_type"
    elif target_col == "level2":
        target_col = "level_2_cell_type"
    elif target_col == "level1":
        target_col = "level_1_cell_type"
    print(f"selected granularity level {target_col}")
    # Remove columns if specified
    if remove_columns is not None:
        remove_columns = [col.strip() for col in remove_columns.split(",")]
        data = data.drop(columns=remove_columns)

    # Remove cell types if specified
    if remove_cell_types is not None:
        remove_cell_types = [
            cell_type.strip() for cell_type in remove_cell_types.split(",")
        ]
        data = data[~data[target_col].isin(remove_cell_types)]
    print("preparing andata object")
    X_cols = data.columns[: data.columns.get_loc(split_col)]
    obs_cols = data.columns[data.columns.get_loc(split_col) :]
    table = pd.read_csv(decision_matrix_path, index_col=0)

    if scaling is not None:
        data[X_cols] = data[X_cols] * scaling
    if log1p:
        data[X_cols] = np.log1p(data[X_cols])
    print(f"Starting SCyan with {n_runs} runs")
    train_times = []
    inference_times = []
    for n in range(n_runs):
        start_train = time.time()
        adata = ad.AnnData(
            X=data[X_cols], obs=data[obs_cols], var=pd.DataFrame(index=X_cols)
        )
        if preprocess:
            scyan.preprocess.scale(adata)
        if batch_key is not None:
            model = scyan.Scyan(adata, table, batch_key=batch_key, prior_std=prior_std)
        else:
            model = scyan.Scyan(adata, table, prior_std=prior_std)
        model.fit(patience=patience, accelerator=accelerator)
        end_train = time.time()
        start_inference = time.time()
        model.predict()
        end_inference = time.time()
        elapsed_train = end_train - start_train
        elapsed_inference = end_inference - start_inference
        train_times.append(elapsed_train)
        inference_times.append(elapsed_inference)

        adata.obs["scyan_pop"] = adata.obs["scyan_pop"].astype(str)
        adata.obs["scyan_pop"] = adata.obs["scyan_pop"].fillna("undefined")
        adata.obs["scyan_pop"] = adata.obs["scyan_pop"].replace("nan", "undefined")
        adata.obs[target_col] = adata.obs[target_col].astype(str)

        if remove_result_cell_types is not None:
            remove_result_cell_types = [
                cell_type.strip() for cell_type in remove_result_cell_types.split(",")
            ]
            adata = adata[~adata.obs[target_col].isin(remove_result_cell_types)]
            adata = adata[~adata.obs["scyan_pop"].isin(remove_result_cell_types)]

        df = adata.to_df()
        df = df.join(adata.obs)

        if save_results:
            cr = classification_report(
                df[target_col].astype(str), df["scyan_pop"].astype(str)
            )
            with open(
                os.path.join(output_path, f"classification_report_{n}.csv"), "w"
            ) as f:
                f.write(cr)

            cm = confusion_matrix(
                df[target_col].astype(str),
                df["scyan_pop"].astype(str),
                normalize="true",
            )
            class_labels = sorted(adata.obs[target_col].unique())
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=class_labels
            )
            fig, ax = plt.subplots(figsize=(12, 12))
            disp.plot(cmap="Blues", xticks_rotation="vertical", ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"confusion_matrix_{n}.png"))
            plt.close(fig)

        df.rename(
            columns={"scyan_pop": "predicted_phenotype", target_col: "true_phenotype"},
            inplace=True,
        )

        df.to_csv(os.path.join(output_path, f"predictions_{n}.csv"), index=False)
        with open(os.path.join(output_path, "fold_times.txt"), "w") as f:
            for i, (elapsed_train, elapsed_pred) in enumerate(
                zip(train_times, inference_times)
            ):
                f.write(f"Fold {i+1} training_time: {elapsed_train:.2f}\n")
                f.write(f"Fold {i+1} prediction_time: {elapsed_pred:.2f}\n")


def main():
    parser = argparse.ArgumentParser(description="Run SCyan on datasets")

    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset CSV file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of runs runs performed by SCyan",
    )

    parser.add_argument(
        "--granularity_level",
        type=str,
        default="level3",
        choices=["level1", "level2", "level3"],
        help="Granularity level for the dataset on which predictions are made",
    )
    parser.add_argument(
        "--remove_columns",
        type=str,
        help="Comma-separated list of columns to remove from the dataset (e.g., 'col1,col2,col3')",
        default=None,
    )
    parser.add_argument(
        "--remove_cell_types",
        type=str,
        help="Comma-separated list of cell types to remove from the groundtruth dataset (e.g., celltype1,celltype2)",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Whether to preprocess the dataset. Default is False",
    )
    parser.add_argument(
        "--decision_matrix_path",
        type=str,
        help="Path to the decision matrix CSV file",
    )
    parser.add_argument(
        "--split_col",
        type=str,
        help="Column name for splitting the dataset when creating the anndata object. It should be the column separating markers and other features",
    )
    parser.add_argument(
        "--scaling",
        type=float,
        default=None,
        help="Scaling factor for the dataset",
    )
    parser.add_argument(
        "--log1p",
        action="store_true",
        help="Whether to apply log1p transformation to the dataset. Default is False",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "tpu", "hpu", "auto"],
        help="Accelerator to use for training. Default is 'cpu'",
    )
    parser.add_argument(
        "--batch_key", type=str, help="Column name for batch key", default=None
    )
    parser.add_argument(
        "--prior_std",
        type=float,
        default=0.3,
        help="prior_std parameter for scyan",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="patience parameter for scyan",
    )
    parser.add_argument(
        "--remove_result_cell_types",
        type=str,
        help="Comma-separated list of cell types to remove from the predictions and groundtruth dataset after the predictions are made (e.g., celltype1,celltype2)",
        default=None,
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to save the predictions CSV file"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Whether to save the result plots. Default is False",
    )

    args = parser.parse_args()

    run_scyan(
        dataset_path=args.dataset_path,
        seed=args.seed,
        n_runs=args.n_runs,
        granularity_level=args.granularity_level,
        remove_columns=args.remove_columns,
        remove_cell_types=args.remove_cell_types,
        preprocess=args.preprocess,
        decision_matrix_path=args.decision_matrix_path,
        accelerator=args.accelerator,
        split_col=args.split_col,
        scaling=args.scaling,
        log1p=args.log1p,
        batch_key=args.batch_key,
        prior_std=args.prior_std,
        patience=args.patience,
        remove_result_cell_types=args.remove_result_cell_types,
        output_path=args.output_path,
        save_results=args.save_results,
    )


if __name__ == "__main__":
    main()
