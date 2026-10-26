import os
import sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)
import glob
import re
import time
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from celllens.utils import drop_zero_variability_columns, cluster_refine, clean_cluster
from utils.greedy_f1_utils import greedy_f1_score, cluster_purity, purity_suffix

# Re-runs the CellLENS clustering (get_lens_clustering) on saved CellLENS embeddings, with optional
# recursive subclustering. Needs the outputs of the CellLENS run scripts:
#   <input>/<method>/it<i>/*_embedding.csv  and  <input>/<method>/level3/predictions_<i>.csv (for feature_labels)

parser = argparse.ArgumentParser(description='CellLENS clustering with optional recursive subclustering on saved CellLENS embeddings.')
parser.add_argument('-i', '--input', dest='input_path', type=str, default=None, help='Dataset folder containing the CellLENS method folder (default: same as -o)')
parser.add_argument('-o', '--output', dest='output_path', type=str, required=True, help='Dataset folder the subclustering method folder is written to')
parser.add_argument('-m', '--method', dest='method', required=True, choices=['CellLENS_Lite', 'CellLENS_Full'], help='CellLENS method folder to read from')
parser.add_argument('-sc', '--subcluster_times', dest='subcluster_times', type=int, default=0, help='Number of recursive subclustering rounds (default: 0 = only re-run the CellLENS clustering)')
parser.add_argument('-sm', '--subcluster_min_cells', dest='subcluster_min_cells', type=int, default=20, help='Minimal number of cells a cluster needs to be subclustered; always at least subcluster_small_neighbors + 1 (default: 20)')
parser.add_argument('-sr', '--subcluster_resolution', dest='subcluster_resolution', type=float, default=1.0, help='Leiden resolution used for subclustering (default: 1.0)')
parser.add_argument('-sk', '--subcluster_small_neighbors', dest='subcluster_small_neighbors', type=int, default=10, help='n_neighbors used for subclustering clusters with <= 15 cells (default: 10)')
parser.add_argument('-sp', '--subcluster_purity', dest='subcluster_purity', type=float, default=None, help='Clusters whose majority cell_type fraction is >= this are not subclustered further, -sc is then the max depth. Uses the ground truth (oracle setting) (default: None = off)')
args = parser.parse_args()

# get_lens_clustering parameters as used in the CellLENS run scripts
NEIGHBOR = 15
RESOLUTION = 1.0
ENTROPY_THRESHOLD = 0.75
CONCEN_THRESHOLD = 1
MAX_BREAKS = 3
SIZE_LIM = 50


def lens_clustering(embedding, feature_labels, neighbor, resolution, umap=False, **leiden_kwargs):
    # Same as CellLENS.get_lens_clustering with umap=True: its UMAP is unused, but sc.tl.umap prunes weak edges of the
    # kNN graph in place, which changes the Leiden result. leiden_kwargs are passed to sc.tl.leiden (default: scanpy's
    # leidenalg as in CellLENS)
    # Centering first does not change the scaled result, but sc.pp.scale computes var = E[x^2] - E[x]^2, which can turn
    # negative (-> NaN std) for nearly constant columns with an offset, e.g. an embedding dimension within a subcluster
    embedding = embedding - embedding.mean(axis=0)
    lens_adata = ad.AnnData(drop_zero_variability_columns(embedding))
    lens_adata.obs['input'] = feature_labels
    sc.pp.scale(lens_adata)
    sc.pp.neighbors(lens_adata, n_neighbors=neighbor, use_rep='X')
    if umap:
        sc.tl.umap(lens_adata)
    sc.tl.leiden(lens_adata, resolution=resolution, **leiden_kwargs)
    lens_refine = cluster_refine(label=lens_adata.obs['leiden'], label_ref=lens_adata.obs['input'],
                                 entropy_threshold=ENTROPY_THRESHOLD, concen_threshold=CONCEN_THRESHOLD,
                                 max_breaks=MAX_BREAKS, size_lim=SIZE_LIM)
    return np.array(clean_cluster(lens_refine), dtype=object)


def subcluster(embedding, feature_labels, labels, times, min_cells, resolution, small_neighbors, true_labels, purity):
    # Each round re-runs the CellLENS clustering on every cluster with >= min_cells (embedding kept fixed),
    # new labels are '<parent>,<child>'. Clusters with > NEIGHBOR cells use NEIGHBOR neighbors,
    # smaller ones small_neighbors, clusters with <= small_neighbors cells are kept.
    # With purity, clusters whose true label purity is >= purity are kept.
    # Subclusters use igraph's Leiden (much faster than leidenalg) and no UMAP (one per subcluster would be slow);
    # the top-level clustering stays as in CellLENS.
    min_cells = max(min_cells, small_neighbors + 1)
    for _ in range(times):
        values, counts = np.unique(labels, return_counts=True)
        for cluster in values[counts >= min_cells]:
            idx = np.where(labels == cluster)[0]
            if purity is not None and cluster_purity(true_labels[idx]) >= purity:
                continue
            neighbor = NEIGHBOR if len(idx) > NEIGHBOR else small_neighbors
            child = lens_clustering(embedding[idx], feature_labels[idx], neighbor, resolution,
                                    flavor='igraph', n_iterations=2, directed=False)
            labels[idx] = [f"{cluster},{c}" for c in child]
    return labels


def read_lens_times(path):
    # 'Fold <i> <event>, <seconds>' lines written by run_celllens.py (and the old CellLENS Lite scripts; old Full scripts did not write them)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"Fold (\d+) \w+, ([\d.]+)", f.read())}


def main():
    method_dir = os.path.join(args.input_path or args.output_path, args.method)
    out_dir = os.path.join(args.output_path, f"{args.method}_subclust{args.subcluster_times}{purity_suffix(args.subcluster_purity)}")
    lens_times = read_lens_times(os.path.join(method_dir, 'fold_times.txt'))
    iterations = sorted(int(d[2:]) for d in os.listdir(method_dir) if re.fullmatch(r"it\d+", d))

    times = []
    for out_idx, iteration in enumerate(iterations):
        embedding = pd.read_csv(glob.glob(os.path.join(method_dir, f"it{iteration}", "*_embedding.csv"))[0]).to_numpy()
        df = pd.read_csv(os.path.join(method_dir, 'level3', f"predictions_{iteration}.csv"))
        if len(df) != len(embedding):
            raise ValueError(f"Iteration {iteration}: {len(df)} cells in predictions but {len(embedding)} in embedding")
        feature_labels = df['feature_labels'].to_numpy()

        start = time.time()
        labels = lens_clustering(embedding, feature_labels, NEIGHBOR, RESOLUTION, umap=True)
        labels = subcluster(embedding, feature_labels, labels, args.subcluster_times, args.subcluster_min_cells,
                            args.subcluster_resolution, args.subcluster_small_neighbors,
                            df['true_phenotype'].to_numpy(), args.subcluster_purity)
        elapsed = time.time() - start

        ## greedy assignement
        output = df.drop(columns=['cluster_label', 'predicted_phenotype'])
        output['cluster_label'] = labels
        results = greedy_f1_score(output, 'true_phenotype', 'cluster_label', tie_strategy='random')
        output['predicted_phenotype'] = results['mapped_predictions']
        path = os.path.join(out_dir, 'level3', f"predictions_{out_idx}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        output.to_csv(path, index=False)
        times.append(elapsed + lens_times[iteration] if lens_times is not None else None)
        print(f"Iteration {iteration}: {len(np.unique(labels))} clusters, saved to {path}")

    # Train time = CellLENS training time (if recorded) + clustering/subclustering time
    if lens_times is None:
        print(f"No fold_times.txt in {method_dir}, so no fold_times.txt is written")
    else:
        with open(os.path.join(out_dir, 'level3', 'fold_times.txt'), 'w') as f:
            for out_idx, elapsed in enumerate(times):
                f.write(f"Fold {out_idx} train_time, {elapsed:.2f}\n")


if __name__ == "__main__":
    main()

#example usage: python3 cluster_celllens_embedding.py -i ../results/cHL2 -o ../results_subclustering/cHL2 -m CellLENS_Lite -sc 2
