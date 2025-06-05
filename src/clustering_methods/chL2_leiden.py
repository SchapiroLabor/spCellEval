import pandas as pd
import scanpy as sc
import anndata as ad
import os
from greedy_f1_utils import greedy_f1_score
import warnings
from anndata._core.aligned_df import ImplicitModificationWarning

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
    print("Loading data...")
    df = pd.read_csv('../chl2/cHL_2_MIBI_quantification.csv')

    features_list = [
        'CD45', 'CD20', 'pSLP-76', 'SLP-76', 'anti-H2AX (pS139)',
        'CD163', 'CD45RO', 'CD28', 'CD153 (CD30L)', 'Lag3', 'CD4',
        'CD11c', 'CD56', 'FoxP3', 'GATA3', 'Granzyme B', 'PD-L1', 'CD16',
        'Ki-67', 'PD-1', 'Pax-5', 'Tox', 'CD161', 'CD68', 'B2-Microglobulin',
        'CD8', 'CD3', 'HLA1', 'CD15', 'Tbet', 'CD14', 'CD123', 'CXCR5',
        'CD45RA', 'HLA-DR', 'CD57', 'IL-10', 'CD30', 'TIM3', 'RORgT', 'TCRgd',
        'CD86', 'CD25', 'Na-K ATPase'
    ]
    # Subset DataFrame into features (X) and observations (obs)
    X = df[features_list]
    obs = df.drop(columns=features_list)
    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs)
    # Optional: Normalize and reduce dimensionality
    print("Normalizing data...")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    print("Running PCA...")
    sc.pp.pca(adata)
    # Compute neighborhood graph (based on PCA or original data)
    print("Computing neighborhood graph...")
    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_pca')  # or use 'X' for raw data

    # run 5 times
    for iteration in range(1, 6): 
        print(f"Iteration {iteration} of 5")
        ## Leiden clustering
        print("Running Leiden clustering with different resolutions")
        sc.tl.leiden(adata, resolution=1.0, key_added='leiden_res1', random_state=iteration)
        df['leiden_res1'] = adata.obs['leiden_res1'].values

        sc.tl.leiden(adata, resolution=0.8, key_added='leiden_res0_8', random_state=iteration)
        df['leiden_res0_8'] = adata.obs['leiden_res0_8'].values

        sc.tl.leiden(adata, resolution=0.5, key_added='leiden_res0_5', random_state=iteration)
        df['leiden_res0_5'] = adata.obs['leiden_res0_5'].values

        sc.tl.leiden(adata, resolution=2.0, key_added='leiden_res2', random_state=iteration)
        df['leiden_res2'] = adata.obs['leiden_res2'].values

        ## greedy assignement
        for c in clustering:
            for k in granularity:
                print(f"Iteration {iteration} : Running greedy assignment of {c} to {k}")
                ## greedy assignement
                f1_macro, f1_weighted, mapped_preds, mapping, ari, nmi, accuracy, mcc, kappa= greedy_f1_score(df, k, c, tie_strategy='random')
                print("F1 Macro:", f1_macro)
                print("F1 Weighted:", f1_weighted)
                output = df.loc[:, ~df.columns.str.startswith('leiden_')].copy()
                output['predicted_phenotype'] = mapped_preds
                output = output.rename(columns={k: "true_phenotype"})
                name = granularity[granularity==k].index[0]
                path = f"../chl2/{c}/{name}/predictions_{iteration}.csv" #create output directory
                os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
                print(f"Saving predictions to {path}")
                output.to_csv(path, index=False)