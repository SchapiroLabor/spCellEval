import anndata as ad
import pandas as pd
import torch
import numpy as np
from starling import starling, utility
from lightning_lite import seed_everything
import pytorch_lightning as pl
import argparse
import time
import os


def run_starling(
    dataset_path: str,
    seed: int,
    split_col_name: str,
    transform: str,
    scaling: float,
    initial_clustering: str,
    k_cluster: int,
    dist_option: str,
    singlet_prop: float,
    model_cell_size: str,
    cell_size_col_name: str,
    model_regularizer: int,
    lr: float,
    n_runs: int,
    output_path: str,
):
    df = pd.read_csv(dataset_path)

    X_columns = df.columns[: df.columns.get_loc(split_col_name)]
    obs_columns = df.columns[df.columns.get_loc(split_col_name) :]
    adata = ad.AnnData(
        X=df[X_columns], obs=df[obs_columns], var=pd.DataFrame(index=X_columns)
    )
    if transform == "log1p":
        adata.X = np.log1p(adata.X)
    elif transform == "arcsinh":
        adata.X = np.arcsinh(adata.X)
    if scaling is not None:
        adata.X = adata.X * scaling

    train_times = []
    # inference_times = []
    for n in range(n_runs):
        seed_everything(seed + n)
        adata = utility.init_clustering(initial_clustering, adata, k=k_cluster)

        st = starling.ST(
            adata=adata,
            dist_option=dist_option,
            singlet_prop=singlet_prop,
            model_cell_size=(model_cell_size == "Y"),
            cell_size_col_name=cell_size_col_name,
            model_regularizer=model_regularizer,
            learning_rate=lr,
        )
        log_tb = pl.loggers.TensorBoardLogger(save_dir="log")
        cb_early_stopping = pl.callbacks.EarlyStopping(
            monitor="train_loss", mode="min", verbose=False
        )
        train_start = time.time()
        st.train_and_fit(
            callbacks=[cb_early_stopping],
            logger=[log_tb],
        )

        train_end = time.time()
        elapsed_train = train_end - train_start
        train_times.append(elapsed_train)

        result = st.result()

        intensities = result.to_df()
        obs = result.obs

        df_result = pd.concat([obs, intensities], axis=1)
        df_result.rename(
            columns={"st_label": "predicted_phenotype", "cell_type": "true_phenotype"},
            inplace=True,
        )

        # Save the results
        df_result.to_csv(
            os.path.join(output_path, f"predictions_fold_{n}.csv"), index=False
        )

    with open(os.path.join(output_path, "fold_times.txt"), "w") as f:
        for i, elapsed_train in enumerate(train_times):
            f.write(f"Fold {i+1} training_time: {elapsed_train:.2f}\n")


def main():
    parser = argparse.ArgumentParser(description="Run starling on datasets")

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
        "--split_col_name",
        type=str,
        help="Column name for splitting the dataset when creating the anndata object. It should be the column separating markers and other features",
    )
    parser.add_argument(
        "--transform",
        type=str,
        help="Transformation to apply to the dataset",
        choices=["log1p", "arcsinh"],
        default=None,
    )
    parser.add_argument(
        "--scaling",
        type=float,
        default=None,
        help="Scaling factor to apply to the dataset after transformation",
    )
    parser.add_argument(
        "--initial_clustering",
        type=str,
        help="Initial clustering method to use",
        choices=["KM", "GMM", "FS", "PG"],
        default="KM",
    )
    parser.add_argument(
        "--k_cluster",
        type=int,
        help="Number of clusters for the initial clustering",
        default=10,
    )
    parser.add_argument(
        "--dist_option",
        type=str,
        help="dist_option param of starling",
        choices=["T", "N"],
        default="T",
    )
    parser.add_argument(
        "--singlet_prop",
        type=float,
        help="Proportion of segmentation error free cells",
        default=0.6,
    )
    parser.add_argument(
        "--model_cell_size",
        type=str,
        help="Model cell size parameter",
        choices=["Y", "N"],
        default="Y",
    )
    parser.add_argument(
        "--cell_size_col_name",
        type=str,
        help="Column name for cell size information",
        default="area",
    )
    parser.add_argument(
        "--model_regularizer",
        type=int,
        help="Model regularizer parameter",
        default=1,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate for ADAM optimizer",
        default=0.001,
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        help="Number of runs performed by starling",
        default=1,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output directory",
    )

    args = parser.parse_args()
    run_starling(
        dataset_path=args.dataset_path,
        seed=args.seed,
        split_col_name=args.split_col_name,
        transform=args.transform,
        scaling=args.scaling,
        initial_clustering=args.initial_clustering,
        k_cluster=args.k_cluster,
        dist_option=args.dist_option,
        singlet_prop=args.singlet_prop,
        model_cell_size=args.model_cell_size,
        cell_size_col_name=args.cell_size_col_name,
        model_regularizer=args.model_regularizer,
        lr=args.lr,
        n_runs=args.n_runs,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
