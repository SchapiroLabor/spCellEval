import os
import sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import re
from utils.greedy_f1_utils import greedy_f1_score
import warnings
from anndata._core.aligned_df import ImplicitModificationWarning
import logging
import time
import argparse

#Parse arguments from command line
parser = argparse.ArgumentParser(description='Leiden clustering with greedy assignment and optional preprocessing after Nimbus.')
parser.add_argument("-gt", "--ground_truth", dest = "ground_truth", required = True, help = "Path to input quantification GT CSV file")
parser.add_argument('-i', '--input', dest='input', type = str, help='Path to nimbus output folder containing one folder per iteration with quantification csv. EG immucan/nimbus/outputs')
parser.add_argument('-o', '--output', dest='output_path', type = str, help='Path to output folder (usually ends with dataset name). Do not put a slash in the end!')
parser.add_argument('-e', '--exclude_markers', dest='exclude_markers', nargs='+', default=[], help='Marker names to exclude from clustering (e.g. DAPI); all other Nimbus marker columns are used')
parser.add_argument("--remove_str", dest = "remove_str", required = False, default = "None", type = str, help = "String to remove from sample_id column of quantification data to match fov column of nimbus data.")
parser.add_argument('-p', '--PCA', dest='PCA', action='store_true', help='Cluster on the first 30 PCs (default: False)')
parser.add_argument('-n', '--normalization', dest='normalization', action='store_true', help='Perform normalization (default: False)')
parser.add_argument('-ha', '--harmony', dest='harmony', action='store_true', help='Harmony integration on sample_id; on the PCs with -p, otherwise on the markers (default: False)')
parser.add_argument('-l', '--log', dest='log', required= False, default='off', choices=['short', 'long', 'off'], help='Logging level: short, long, or off (default: long)')
parser.add_argument('-it', '--iterations', dest='iterations', type=int, required= False, default=5, help='Number of iterations to run (default: 5)')
parser.add_argument('-r', '--resolution', dest='resolution', type=float, required= False, default=1.0, help='Resolution for Leiden clustering (default: 1.0)')
parser.add_argument('-sc', '--subcluster_times', dest='subcluster_times', type=int, default=0, help='Number of recursive subclustering rounds applied to the resulting clusters (default: 0 = off)')
parser.add_argument('-sm', '--subcluster_min_cells', dest='subcluster_min_cells', type=int, default=10, help='Minimal number of cells a cluster needs to be subclustered (default: 10)')
parser.add_argument('-sr', '--subcluster_resolution', dest='subcluster_resolution', type=float, default=1.0, help='Leiden resolution used for subclustering (default: 1.0)')
args = parser.parse_args()

# Columns of the joined table that are not Nimbus marker columns
NON_MARKER_COLUMNS = ["cell_id", "sample_id", "level_1_cell_type", "level_2_cell_type", "cell_type", "fov", "label"]
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

# Define function to join ground truth with nimbus output
def join_ground_truth_with_nimbus(ground_truth,input, remove_str, iteration):
    # Read gt data
    gt = pd.read_csv(ground_truth)
    # sample_id as string to match the (string) fov column of nimbus, also for numeric sample ids
    gt['sample_id'] = gt['sample_id'].astype(str)
    # Remove substrings from sample_id if specified
    if remove_str != "None":
        gt['sample_id'] = gt['sample_id'].str.replace(remove_str, '', regex=True)

    # Read nimbus data
    nimbus_path = os.path.join(input, f"iteration_{iteration}", "nimbus_cell_table.csv")
    nimbus_df = pd.read_csv(nimbus_path)
    # Convert fov column to string type
    nimbus_df['fov'] = nimbus_df['fov'].astype(str)

    # Reduce nimbus data to only contain cells also present in gt, based on labels and fov
    nimbus_df = pd.merge(
        nimbus_df,
        gt[["sample_id", "cell_id"]],
        left_on=["fov", "label"],
        right_on=["sample_id", "cell_id"],
        how="inner"
    )
    nimbus_df = nimbus_df.drop(columns=["sample_id", "cell_id"])

    # Add celltype column to nimbus dataframe
    gt_subset = gt[["cell_id", "sample_id", "level_1_cell_type","level_2_cell_type", "cell_type"]]
    nimbus_df = pd.merge(
        gt_subset,
        nimbus_df,
        left_on=["sample_id", "cell_id"],
        right_on=["fov", "label"],
        how="right"
    )
    return nimbus_df

# Define preprocessing function
def preprocessing(nimbus_df, exclude_markers, PCA, normalization, harmony):
    # Markers are all Nimbus columns, minus excluded ones
    features_list = [c for c in nimbus_df.columns if c not in NON_MARKER_COLUMNS + exclude_markers]
    # Subset DataFrame into features and observations
    X = nimbus_df[features_list]
    obs = nimbus_df.drop(columns=features_list)
    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs)

    if normalization == True:
        sc.pp.normalize_total(adata, target_sum=1)

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
    return adata

# Define recursive subclustering function
def subcluster(adata, key, times, min_cells, resolution, random_state):
    # Each round re-clusters every cluster with >= min_cells on its subgraph of the kNN graph,
    # new labels are '<parent>,<child>'
    for _ in range(times):
        counts = adata.obs[key].value_counts()
        for cluster in counts[counts >= min_cells].index:
            sc.tl.leiden(adata, resolution=resolution, restrict_to=(key, [cluster]), key_added=key, random_state=random_state)

# Define Leiden clustering with greedy assignment function
def leiden_with_greedy(adata, nimbus_df, iteration, output_path, log, res, subcluster_times, subcluster_min_cells, subcluster_resolution):
    resolution = f'leiden_res{str(res).replace(".", "_")}'
    # Method folder 'nimbus_leiden_res<r>' (distinct from the Leiden-on-intensities folders);
    # subclustering runs are saved in a separate method folder (e.g. 'nimbus_leiden_res1_0_subclust3')
    suffix = f'_subclust{subcluster_times}' if subcluster_times > 0 else ''
    folder = f'nimbus_{resolution}{suffix}'
    start = time.time()
    # Seed per iteration, otherwise all iterations on the same input give identical clusters (scanpy default random_state=0)
    sc.tl.leiden(adata, resolution=res, key_added=resolution, random_state=iteration)
    subcluster(adata, resolution, subcluster_times, subcluster_min_cells, subcluster_resolution, iteration)
    elapsed = time.time() - start
    if log == 'long':
        logging.info(f'Iteration {iteration}: Leiden clustering with resolution={res} took {elapsed/60:.2f} minutes')
    if log == 'short':
        logging.info(f'Resolution={res}: {elapsed/60:.2f} min')
    output = nimbus_df.copy()
    output[resolution] = adata.obs[resolution].values
    #append time to the results
    text_path = os.path.join(output_path, folder, 'level3', 'fold_times.txt')
    os.makedirs(os.path.dirname(text_path), exist_ok=True)
    mode = 'w' if iteration == 0 else 'a'
    with open(text_path, mode) as f:
        f.write(f"Fold {iteration} train_time, {elapsed:.2f}\n")

    ## greedy assignement
    results = greedy_f1_score(output, 'cell_type', resolution, tie_strategy='random')
    output['predicted_phenotype'] = results['mapped_predictions']
    output = output.rename(columns={'cell_type': "true_phenotype"})
    path = f"{output_path}/{folder}/level3/predictions_{iteration}.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
    if log == 'long':
        logging.info(f"Saving predictions to {path}")
    output.to_csv(path, index=False)

def main():
    if args.log != 'off':
        logging_setup(args.output_path)
    for iteration in range(args.iterations):
        # Join ground truth with nimbus output (Nimbus writes iteration_1 ... iteration_N)
        nimbus_df = join_ground_truth_with_nimbus(args.ground_truth, args.input, args.remove_str, iteration + 1)
        adata = preprocessing(nimbus_df, args.exclude_markers, args.PCA, args.normalization, args.harmony)
        leiden_with_greedy(adata, nimbus_df, iteration, args.output_path, args.log, args.resolution,
                           args.subcluster_times, args.subcluster_min_cells, args.subcluster_resolution)


if __name__ == "__main__":
    main()


#example usage: python3 Nimbus_leiden_clustering.py -gt ../chl2/cHL_2_MIBI_quantification.csv -i ../chl2/nimbus/outputs -o ../chl2/CLI_test -e HistoneH3 -l off
