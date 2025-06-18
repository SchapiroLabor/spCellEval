import pandas as pd
import scanpy as sc
import anndata as ad
import os
from greedy_f1_utils import greedy_f1_score
import warnings
from anndata._core.aligned_df import ImplicitModificationWarning
import logging
import time

#create logs directory if it does not exist
os.makedirs('../logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=f'../logs/{time.strftime("%Y-%m-%d")}_leiden_immucan_run.log',
    filemode="w",  # overwrite on each run; change to "a" to append
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Suppress specific warning from anndata
warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
# Suppress specific FutureWarning from scanpy or general usage
warnings.filterwarnings("ignore", category=FutureWarning)

## define parameters
granularity = pd.Series({'level3':'cell_type', 'level1':'level_1_cell_type', 'level2':'level_2_cell_type'})
clustering = pd.Series({'leiden_res1':'leiden_res1', 'leiden_res0_8':'leiden_res0_8', 'leiden_res0_5':'leiden_res0_5', 'leiden_res2':'leiden_res2'})

if __name__ == "__main__":
    ## Prep
    # Read in data
    logging.info("Loading data...")
    df = pd.read_csv('../immucan/IMMUcan_quantification.csv')
    features_list = [
    'MPO', 'SMA', 'CD16', 'CD38', 'HLADR', 'CD27', 'CD15',
    'CD45RA', 'CD163', 'B2M', 'CD20', 'CD68', 'Ido1', 'CD3', 'LAG3',
    'CD11c', 'PD1', 'PDGFRb', 'CD7', 'GrzB', 'PDL1', 'TCF7', 'CD45RO',
    'FOXP3', 'ICOS', 'CD8a', 'CarbonicAnhydrase', 'CD33', 'Ki67', 'VISTA',
    'CD40', 'CD4', 'CD14', 'Ecad', 'CD303', 'CD206', 'cleavedPARP'
    ]
    # Subset DataFrame into features (X) and observations (obs)
    X = df[features_list]
    obs = df.drop(columns=features_list)
    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs)
    # Optional: Normalize and reduce dimensionality
    logging.info("Normalizing data...")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    logging.info("Running PCA...")
    sc.pp.pca(adata)
    # Compute neighborhood graph (based on PCA or original data)
    logging.info("Computing neighborhood graph...")
    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_pca')  # or use 'X' for raw data

    # run 5 times
    for iteration in range(1, 6): 
        logging.info(f"Iteration {iteration} of 5")
        ## Leiden clustering
        logging.info("Running Leiden clustering with different resolutions")

        start = time.time()
        sc.tl.leiden(adata, resolution=1.0, key_added='leiden_res1', random_state=iteration)
        elapsed = time.time() - start
        logging.info(f'Leiden clustering with resolution=1.0 took {elapsed/60:.2f} minutes')
        df['leiden_res1'] = adata.obs['leiden_res1'].values

        start = time.time()
        sc.tl.leiden(adata, resolution=0.8, key_added='leiden_res0_8', random_state=iteration)
        elapsed = time.time() - start
        logging.info(f'Leiden clustering with resolution=0.8 took {elapsed/60:.2f} minutes')
        df['leiden_res0_8'] = adata.obs['leiden_res0_8'].values

        start = time.time()
        sc.tl.leiden(adata, resolution=0.5, key_added='leiden_res0_5', random_state=iteration)
        elapsed = time.time() - start
        logging.info(f'Leiden clustering with resolution=0.5 took {elapsed/60:.2f} minutes')
        df['leiden_res0_5'] = adata.obs['leiden_res0_5'].values

        start = time.time()
        sc.tl.leiden(adata, resolution=2.0, key_added='leiden_res2', random_state=iteration)
        elapsed = time.time() - start
        logging.info(f'Leiden clustering with resolution=2.0 took {elapsed/60:.2f} minutes')
        df['leiden_res2'] = adata.obs['leiden_res2'].values



        ## greedy assignement
        for c in clustering:
            for k in granularity:
                logging.info(f"Iteration {iteration} : Running greedy assignment of {c} to {k}")
                ## greedy assignement
                results = greedy_f1_score(df, k, c, tie_strategy='random')
                logging.info(f'F1 Macro: {results["f1_macro"]}')
                logging.info(f'F1 Weighted: {results["f1_weighted"]}')
                output = df.loc[:, ~(df.columns.str.startswith('leiden_') & (df.columns != c))].copy()
                output['predicted_phenotype'] = results['mapped_predictions']
                output = output.rename(columns={k: "true_phenotype"})
                name = granularity[granularity==k].index[0]
                path = f"../immucan/{c}/{name}/predictions_{iteration}.csv" #create output directory
                os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
                logging.info(f"Saving predictions to {path}")
                output.to_csv(path, index=False)