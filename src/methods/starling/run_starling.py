import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)
import anndata as ad
import pandas as pd
import torch
import numpy as np
from starling import starling, utility
from lightning_fabric.utilities.seed import seed_everything
import pytorch_lightning as pl
import argparse
import time
import scanpy.external as sce
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from utils.greedy_f1_utils import greedy_f1_score, cluster_purity, purity_suffix

# k of starling's PG initial clustering (scanpy phenograph default), default k for PG subclustering
PG_K = 30


def subcluster_labels(X, labels, method, times, min_cells, sub_k, true_labels, purity):
    # Each round re-clusters every cluster with >= min_cells with the initial clustering method,
    # new labels are '<parent>,<child>'. KM/GMM k is capped by the cluster size.
    # PG uses sub_k as PhenoGraph k (larger k -> fewer subclusters) and keeps clusters with <= sub_k cells.
    # With purity, clusters whose cell_type purity is >= purity are kept.
    labels = labels.astype(str).astype(object)
    if method == "PG":
        min_cells = max(min_cells, sub_k + 1)
    for _ in range(times):
        values, counts = np.unique(labels, return_counts=True)
        for cluster in values[counts >= min_cells]:
            idx = np.where(labels == cluster)[0]
            if purity is not None and cluster_purity(true_labels[idx]) >= purity:
                continue
            if method == "KM":
                sub = KMeans(min(sub_k, len(idx))).fit_predict(X[idx])
            elif method == "GMM":
                sub = GaussianMixture(n_components=min(sub_k, len(idx)), covariance_type="diag").fit_predict(X[idx])
            elif method == "PG":
                sub, _, _ = sce.tl.phenograph(X[idx], k=sub_k)
            labels[idx] = [f"{cluster},{s}" for s in sub]
    return labels


def run_starling(
    dataset_path: str,
    seed: int,
    split_col_name: str,
    exclude_markers: list,
    transform: str,
    cofactor: float,
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
    subcluster_times: int,
    subcluster_min_cells: int,
    subcluster_k: int,
    subcluster_purity: float,
):
    df = pd.read_csv(dataset_path)

    # Markers are all columns before split_col_name, minus excluded ones (those are kept in obs)
    X_columns = [c for c in df.columns[: df.columns.get_loc(split_col_name)] if c not in exclude_markers]
    obs_columns = df.columns.drop(X_columns)
    adata = ad.AnnData(
        X=df[X_columns], obs=df[obs_columns], var=pd.DataFrame(index=X_columns)
    )
    if transform == "log1p":
        adata.X = np.log1p(adata.X)
    elif transform == "arcsinh":
        adata.X = np.arcsinh(adata.X / cofactor)
    if scaling is not None:
        adata.X = adata.X * scaling

    if subcluster_k is None:
        subcluster_k = PG_K if initial_clustering == "PG" else 2

    # Predictions go to <output_path>/starling/level3; subclustering runs to a separate method folder
    # (e.g. 'starling_subclust3'), for PG with the subclustering k appended if it differs from PG_K
    # (e.g. 'starling_subclust3_k60'), and the purity if set (e.g. 'starling_subclust3_purity0_9')
    method_name = f"starling_subclust{subcluster_times}" if subcluster_times > 0 else "starling"
    if subcluster_times > 0 and initial_clustering == "PG" and subcluster_k != PG_K:
        method_name += f"_k{subcluster_k}"
    if subcluster_times > 0:
        method_name += purity_suffix(subcluster_purity)
    method_dir = os.path.join(output_path, method_name)
    os.makedirs(os.path.join(method_dir, "level3"), exist_ok=True)
    train_times = []
    inference_times = []
    for n in range(n_runs):
        seed_everything(seed + n)
        # Training time = initial clustering (+ subclustering) + starling training, prediction time = cell assignment
        train_start = time.time()
        adata_st = utility.init_clustering(initial_clustering, adata, k=k_cluster)
        if subcluster_times > 0:
            labels = subcluster_labels(np.asarray(adata_st.X), adata_st.obs["init_label"].to_numpy(),
                                       initial_clustering, subcluster_times, subcluster_min_cells, subcluster_k,
                                       adata.obs["cell_type"].to_numpy(), subcluster_purity)
            # starling expects integer labels 0..K-1
            adata_st = utility.init_clustering("User", adata, labels=pd.factorize(labels)[0])

        st = starling.ST(
            adata=adata_st,
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
        st.train_and_fit(
            callbacks=[cb_early_stopping],
            logger=[log_tb],
        )

        train_end = time.time()
        elapsed_train = train_end - train_start
        train_times.append(elapsed_train)

        pred_start = time.time()
        result = st.result()
        inference_times.append(time.time() - pred_start)

        intensities = result.to_df()
        obs = result.obs

        df_result = pd.concat([obs, intensities], axis=1)
        df_result.rename(
            columns={"cell_type": "true_phenotype", "st_label": "cluster_label"},
            inplace=True,
        )
        greedy_output = greedy_f1_score(df_result, "true_phenotype", "cluster_label", tie_strategy="random")
        df_result['predicted_phenotype'] = greedy_output['mapped_predictions']

        # Save the results
        df_result.to_csv(
            os.path.join(method_dir, "level3", f"predictions_{n}.csv"), index=False
        )

        del st, adata_st, df_result, result
        torch.cuda.empty_cache()

    with open(os.path.join(method_dir, "level3", "fold_times.txt"), "w") as f:
        for i, (elapsed_train, elapsed_pred) in enumerate(zip(train_times, inference_times)):
            f.write(f"Fold {i} training_time: {elapsed_train:.2f}\n")
            f.write(f"Fold {i} prediction_time: {elapsed_pred:.2f}\n")


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
        "--exclude_markers",
        type=str,
        nargs="+",
        default=[],
        help="Marker names to exclude from clustering (e.g. DAPI)",
    )
    parser.add_argument(
        "--transform",
        type=str,
        help="Transformation to apply to the dataset",
        choices=["log1p", "arcsinh"],
        default=None,
    )
    parser.add_argument(
        "--cofactor",
        type=float,
        default=5.0,
        help="Cofactor for arcsinh, i.e. arcsinh(x / cofactor) (default: 5)",
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
        choices=["KM", "GMM", "PG"],
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
        help="Path to the output directory (dataset folder); results go to <output_path>/starling[_subclustN]/",
    )
    parser.add_argument(
        "--subcluster_times",
        type=int,
        default=0,
        help="Number of recursive subclustering rounds applied to the initial clusters before starling (default: 0 = off)",
    )
    parser.add_argument(
        "--subcluster_min_cells",
        type=int,
        default=10,
        help="Minimal number of cells an initial cluster needs to be subclustered; for PG always at least subcluster_k + 1 (default: 10)",
    )
    parser.add_argument(
        "--subcluster_k",
        type=int,
        default=None,
        help="KM/GMM: number of subclusters per cluster (default: 2); PG: PhenoGraph k for subclustering, larger k gives fewer subclusters (default: 30)",
    )
    parser.add_argument(
        "--subcluster_purity",
        type=float,
        default=None,
        help="Initial clusters whose majority cell_type fraction is >= this are not subclustered further, subcluster_times is then the max depth. Uses the ground truth (oracle setting) (default: None = off)",
    )

    args = parser.parse_args()
    run_starling(
        dataset_path=args.dataset_path,
        seed=args.seed,
        split_col_name=args.split_col_name,
        exclude_markers=args.exclude_markers,
        transform=args.transform,
        cofactor=args.cofactor,
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
        subcluster_times=args.subcluster_times,
        subcluster_min_cells=args.subcluster_min_cells,
        subcluster_k=args.subcluster_k,
        subcluster_purity=args.subcluster_purity,
    )


if __name__ == "__main__":
    main()
