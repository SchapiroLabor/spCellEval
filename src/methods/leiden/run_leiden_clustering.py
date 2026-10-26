import os
import sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
from utils.greedy_f1_utils import greedy_f1_score
import warnings
from anndata._core.aligned_df import ImplicitModificationWarning
import logging
import time
import argparse

#Parse arguments from command line
parser = argparse.ArgumentParser(description='Leiden clustering with greedy assignment and optional preprocessing.')
parser.add_argument('-i', '--input', dest='input', type = str, help='Path to input CSV file')
parser.add_argument('-o', '--output', dest='output_path', type = str, help='Path to output folder (usually ends with dataset name). Do not put a slash in the end!')
parser.add_argument('-s', '--split_col_name', dest='split_col_name', type=str, required=True, help='First non-marker column; all columns before it are used as markers (e.g. cell_id)')
parser.add_argument('-e', '--exclude_markers', dest='exclude_markers', nargs='+', default=[], help='Marker names to exclude from clustering (e.g. DAPI)')
parser.add_argument('-p', '--PCA', dest='PCA', action='store_true', help='Cluster on the first 30 PCs (default: False)')
parser.add_argument('-t', '--transform', dest='transform', default='none', choices=['none', 'log1p', 'arcsinh'], help='Transformation applied to the markers after normalization (default: none)')
parser.add_argument('-cf', '--cofactor', dest='cofactor', type=float, default=5.0, help='Cofactor for arcsinh, i.e. arcsinh(x / cofactor) (default: 5)')
parser.add_argument('-n', '--normalization', dest='normalization', action='store_true', help='Perform normalization (default: False)')
parser.add_argument('-ha', '--harmony', dest='harmony', action='store_true', help='Harmony integration on sample_id; on the PCs with -p, otherwise on the markers (default: False)')
parser.add_argument('-l', '--log', dest='log', required= False, default='off', choices=['short', 'long', 'off'], help='Logging level: short, long, or off (default: long)')
parser.add_argument('-it', '--iterations', dest='iterations', type=int, required= False, default=5, help='Number of iterations to run (default: 5)')
parser.add_argument('-r', '--resolution', dest='resolution', type=float, required= False, default=1.0, help='Resolution for Leiden clustering (default: 1.0)')
parser.add_argument('-sc', '--subcluster_times', dest='subcluster_times', type=int, default=0, help='Number of recursive subclustering rounds applied to the resulting clusters (default: 0 = off)')
parser.add_argument('-sm', '--subcluster_min_cells', dest='subcluster_min_cells', type=int, default=20, help='Minimal number of cells a cluster needs to be subclustered (default: 20)')
parser.add_argument('-sr', '--subcluster_resolution', dest='subcluster_resolution', type=float, default=1.0, help='Leiden resolution used for subclustering (default: 1.0)')
args = parser.parse_args()

# Suppress specific warning from anndata
warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
# Suppress specific FutureWarning from scanpy or general usage
warnings.filterwarnings("ignore", category=FutureWarning)

# Define logging setup function
def logging_setup(output_path):
    """
    Set up logging configuration.
    """
    # Create logs directory if it does not exist
    os.makedirs(f'{output_path}/logs', exist_ok=True)

    # Configure logging
    logging.basicConfig(
        filename=f'{output_path}/logs/{time.strftime("%Y-%m-%d")}_leiden.log',
        filemode="w",  # overwrite on each run; change to "a" to append
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

# Define preprocessing function
def preprocessing(input, split_col_name, exclude_markers, PCA, normalization, transform, cofactor, harmony):
    # Read in data
    df = pd.read_csv(input)
    # Markers are all columns before split_col_name, minus excluded ones
    markers = df.columns[:df.columns.get_loc(split_col_name)]
    features_list = [m for m in markers if m not in exclude_markers]
    # Subset DataFrame into features and observations
    X = df[features_list]
    obs = df.drop(columns=features_list)
    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs)

    if normalization == True:
        sc.pp.normalize_total(adata, target_sum=1)
    if transform == 'log1p':
        sc.pp.log1p(adata)
    elif transform == 'arcsinh':
        adata.X = np.arcsinh(adata.X / cofactor)

    # Representation for the kNN graph: markers, PCs (30, or n_markers - 1 if fewer markers) and/or Harmony on top
    rep = 'X'
    if PCA == True:
        sc.pp.pca(adata, n_comps=min(30, adata.n_vars - 1))
        rep = 'X_pca'
    if harmony == True:
        if rep == 'X':
            adata.obsm['X_markers'] = np.asarray(adata.X)
            rep = 'X_markers'
        # harmonypy needs the batch column as categories (fails with 'KeyError: unique' for numeric sample_id)
        adata.obs['sample_id'] = adata.obs['sample_id'].astype(str).astype('category')
        sc.external.pp.harmony_integrate(adata, 'sample_id', basis=rep, adjusted_basis='X_harmony')
        rep = 'X_harmony'
    sc.pp.neighbors(adata, n_neighbors=15, use_rep=rep)
    return adata, df

# Define recursive subclustering function
def subcluster(adata, key, times, min_cells, resolution, random_state):
    # Each round re-clusters every cluster with >= min_cells on a new kNN graph built from its cells only
    # (same representation as the top-level graph), new labels are '<parent>,<child>'
    rep = adata.uns['neighbors']['params']['use_rep']
    labels = adata.obs[key].astype(str).to_numpy()
    for _ in range(times):
        new_labels = labels.copy()
        for cluster, idx in pd.Series(labels).groupby(labels).indices.items():
            if len(idx) >= min_cells:
                sub = adata[idx].copy()
                sc.pp.neighbors(sub, n_neighbors=15, use_rep=rep)
                sc.tl.leiden(sub, resolution=resolution, random_state=random_state, flavor='igraph', n_iterations=2, directed=False)
                new_labels[idx] = cluster + ',' + sub.obs['leiden'].astype(str).to_numpy()
        labels = new_labels
    adata.obs[key] = pd.Categorical(labels)

# Define Leiden clustering with greedy assignment function
def leiden_with_greedy(adata, df, iterations, output_path, log, res, subcluster_times, subcluster_min_cells, subcluster_resolution, harmony):
    resolution = f'leiden_res{str(res).replace(".", "_")}'
    # Subclustering runs are saved in a separate method folder (e.g. '<method>_subclust3'), Harmony runs get '_harm' appended
    suffix = f'_subclust{subcluster_times}' if subcluster_times > 0 else ''
    if harmony:
        suffix += '_harm'
    for iteration in range(iterations):
        start = time.time()
        # Seed per iteration, otherwise all iterations give identical clusters (scanpy default random_state=0)
        sc.tl.leiden(adata, resolution=res, key_added=resolution, random_state=iteration, flavor='igraph', n_iterations=2, directed=False)
        subcluster(adata, resolution, subcluster_times, subcluster_min_cells, subcluster_resolution, iteration)
        elapsed = time.time() - start
        if log == 'long':
            logging.info(f'Iteration {iteration}: Leiden clustering with resolution={res} took {elapsed/60:.2f} minutes')
        if log == 'short':
            logging.info(f'Resolution={res}: {elapsed/60:.2f} min')
        output = df.copy()
        output['cluster_label'] = adata.obs[resolution].values
        #append time to the results
        text_path = os.path.join(output_path, resolution + suffix, 'level3', 'fold_times.txt')
        os.makedirs(os.path.dirname(text_path), exist_ok=True)
        mode = 'w' if iteration == 0 else 'a'
        with open(text_path, mode) as f:
            f.write(f"Fold {iteration} train_time, {elapsed:.2f}\n")

        ## greedy assignement
        results = greedy_f1_score(output, 'cell_type', 'cluster_label', tie_strategy='random')
        output['predicted_phenotype'] = results['mapped_predictions']
        output = output.rename(columns={'cell_type': "true_phenotype"})
        path = f"{output_path}/{resolution}{suffix}/level3/predictions_{iteration}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
        if log == 'long':
            logging.info(f"Saving predictions to {path}")
        output.to_csv(path, index=False)

def main():
    if args.log != 'off':
        logging_setup(args.output_path)
    adata, df = preprocessing(args.input, args.split_col_name, args.exclude_markers, args.PCA, args.normalization, args.transform, args.cofactor, args.harmony)
    leiden_with_greedy(adata, df, args.iterations, args.output_path, args.log, args.resolution, args.subcluster_times, args.subcluster_min_cells, args.subcluster_resolution, args.harmony)


if __name__ == "__main__":
    main()
