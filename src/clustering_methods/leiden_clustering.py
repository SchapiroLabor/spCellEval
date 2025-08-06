import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import os
from greedy_f1_utils import greedy_f1_score
import warnings
from anndata._core.aligned_df import ImplicitModificationWarning
import logging
import time
import argparse

#Parse arguments from command line
parser = argparse.ArgumentParser(description='Leiden clustering with greedy assignment and optional preprocessing.')
parser.add_argument('-i', '--input', dest='input', type = str, help='Path to input CSV file')
parser.add_argument('-o', '--output', dest='output_path', type = str, help='Path to output folder (usually ends with dataset name). Do not put a slash in the end!')
parser.add_argument('-m', '--markers', dest='markers', nargs='+', help='List of marker names to use for clustering')
parser.add_argument('-p', '--PCA', dest='PCA', action='store_true', help='Perform PCA (default: False)')
parser.add_argument('-a', '--arcsine', dest='arcsine', action='store_true', help='Perform arcsine transformation (default: False)')
parser.add_argument('-n', '--normalization', dest='normalization', action='store_true', help='Perform normalization (default: False)')
parser.add_argument('-l', '--log', dest='log', required= False, default='off', choices=['short', 'long', 'off'], help='Logging level: short, long, or off (default: long)')
parser.add_argument('-it', '--iterations', dest='iterations', type=int, required= False, default=5, help='Number of iterations to run (default: 5)')
parser.add_argument('-r', '--resolutions', dest='resolutions', nargs='+', type=float, required= False, default=[0.5, 0.8, 1.0, 2.0], help='List of resolutions for Leiden clustering (default: [0.5, 0.8, 1.0, 2.0])')
args = parser.parse_args()

# define parameters
granularity = pd.Series({'level3':'cell_type', 'level1':'level_1_cell_type', 'level2':'level_2_cell_type'})
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
def preprocessing(input, markers, PCA, normalization, arcsine):
    # Read in data
    df = pd.read_csv(input)
    features_list = markers
    # Subset DataFrame into features and observations
    X = df[features_list]
    obs = df.drop(columns=features_list)
    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs)
    #if arcsine == True:
    #    #insert arcsine transformation here

    if normalization == True:
        sc.pp.normalize_total(adata, target_sum=1)

    if PCA == True:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_pca')
    else:
        sc.pp.neighbors(adata, n_neighbors=10)
    return adata, df

# Define Leiden clustering with greedy assignment function
def leiden_with_greedy(adata, df, iterations, output_path, log, resolutions):
    for iteration in range(1, iterations+1):
        for res in resolutions:
            start = time.time()
            sc.tl.leiden(adata, resolution=res, key_added=f'leiden_res{res}')
            elapsed = time.time() - start
            if log == 'long':
                logging.info(f'Iteration {iteration}: Leiden clustering with resolution={res} took {elapsed/60:.2f} minutes')
            if log == 'short':
                logging.info(f'Resolution={res}: {elapsed/60:.2f} min')
            output = df.copy()
            output[f'leiden_res{str(res).replace(".", "_")}'] = adata.obs[f'leiden_res{res}'].values
            resolution = f'leiden_res{str(res).replace(".", "_")}'
            #append time to the results
            text_path = os.path.join(output_path, resolution, 'fold_times.txt')
            os.makedirs(os.path.dirname(text_path), exist_ok=True)
            mode = 'w' if iteration == 1 else 'a'
            with open(text_path, mode) as f:
                f.write(f"Fold {iteration} train_time, {elapsed:.2f}\n")

            for k in granularity:
                ## greedy assignement
                results = greedy_f1_score(output, k, resolution, tie_strategy='random')
                output['predicted_phenotype'] = results['mapped_predictions']
                output = output.rename(columns={k: "true_phenotype"})
                name = granularity[granularity==k].index[0]
                path = f"{output_path}/{resolution}/{name}/predictions_{iteration}.csv"
                os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
                if log == 'long':
                    logging.info(f"Saving predictions to {path}")
                output.to_csv(path, index=False)

def main():
    if args.log != 'off':
        logging_setup(args.output_path)
    adata, df = preprocessing(args.input, args.markers, args.PCA, args.normalization, args.arcsine)
    leiden_with_greedy(adata, df, args.iterations, args.output_path, args.log, args.resolutions)


if __name__ == "__main__":
    main()


#example usage: python3 leiden_clustering.py -i ../chl2/cHL_2_MIBI_quantification.csv -o ../chl2/CLI_test -m CD163 CD11c CD28 -r 2 5 1 -l short
#MPO SMA CD16 CD38 HLADR CD27 CD15 CD45RA CD163 B2M CD20 CD68 Ido1 CD3 LAG3 CD11c PD1 PDGFRb CD7 GrzB PDL1 TCF7 CD45RO FOXP3 ICOS CD8a CarbonicAnhydrase CD33 Ki67 VISTA CD40 CD4 CD14 Ecad CD303 CD206 cleavedPARP ## nuclear: HistoneH3
#CD45 CD20 'pSLP-76' 'SLP-76' 'anti-H2AX (pS139)' CD163 CD45RO CD28 'CD153 (CD30L)' Lag3 CD4 CD11c CD56 FoxP3 GATA3 'Granzyme B' 'PD-L1' CD16 'Ki-67' 'PD-1' 'Pax-5' Tox CD161 CD68 'B2-Microglobulin' CD8 CD3 HLA1 CD15 Tbet CD14 CD123 CXCR5 CD45RA 'HLA-DR' CD57 'IL-10' CD30 TIM3 RORgT TCRgd CD86 CD25 'Na-K ATPase' ## nuclear: dsDNA 'Histone H3'
