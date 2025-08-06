import pandas as pd
from celllens.utils import *
import os
from celllens.preprocessing import *
from celllens.datasets import *
from celllens.celllens import *
import torch
from greedy_f1_utils import greedy_f1_score
# for clearity
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')
import logging
import time
import argparse

#Parse arguments from command line
parser = argparse.ArgumentParser(description='Running CellLENS lite on chL2.')
parser.add_argument('-i', '--input', dest='input', type = str, help='Path to input CSV file')
parser.add_argument('-o', '--output', dest='output_path', type = str, help='Path to output folder (usually ends with dataset name). Do not put a slash in the end!')
parser.add_argument('-pc', '--path2crops', dest='path2crops', type = str, help='Path to folder with crops.')
parser.add_argument('-m', '--markers', dest='markers', nargs='+', help='List of marker names to use.')
parser.add_argument('-it', '--iterations', dest='iterations', type=int, required= False, default=5, help='Number of iterations to run (default: 5)')
parser.add_argument('-l', '--log', dest='log', required= False, default='off', choices=['short', 'long', 'off'], help='Logging level: short, long, or off (default: long)')
args = parser.parse_args()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

granularity = pd.Series({'level3':'cell_type', 'level1':'level_1_cell_type', 'level2':'level_2_cell_type'})
  
# Define logging setup function
def logging_setup(output_path):
    """
    Set up logging configuration.
    """
    # Create logs directory if it does not exist
    os.makedirs(f'{output_path}/logs', exist_ok=True)

    # Configure logging
    logging.basicConfig(
        filename=f'{output_path}/logs/{time.strftime("%Y-%m-%d")}_celllens_lite.log',
        filemode="w",  # overwrite on each run; change to "a" to append
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def lite_celllens(input, markers, iterations, output_path, log, path2crops):    
    # Set the device for PyTorch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if log == 'long':
        logging.info(f"Using device: {device}")
    
    ## Prep
    # Read in data
    df = pd.read_csv(input)

    #add a column called image_shift that contains a unique integer for each image
    df["image_shift"] = df["sample_id"].str.replace('.csv', '')
    df["image_shift"] = df["image_shift"].astype('int')
    df["image_shift"] = df["image_shift"] - 1

    # add shift value to x coordinates to 'stitch' images together
    df['x'] = df['image_shift'] * 3500 + df['x']

    # mirror coordinates to match image
    df['x_new'] = df['y']
    df['y_new'] = df['x']

    ## CellLENS
    features_list = markers

    # optionally, clean data Nans - in case data have fault
    df.fillna(0, inplace=True)

    # run 5 times
    for iteration in range(1, iterations + 1): 
        if log == 'long':
            logging.info(f"Running iteration {iteration} of 5")
        murine_dataset = LENS_Dataset(df,
                                    features_list=features_list,
                                    nbhd_composition=20, #adjusted to match default described on github
                                    feature_neighbor=15,
                                    spatial_neighbor=15,
                                    path2img=f'{path2crops}/', # location to store cropped images
                                    use_transform = False)


        murine_dataset.initialize(cent_x="x_new", # x loc of cells
                                cent_y="y_new", # y loc of cells
                                celltype="feature_labels", # default option - run leiden to initialize the clusters
                                pca_components=25, # PCA components to use, user decide
                                cluster_res= 0.2 # leiden initialization resolution
                                ) #, giving n_clusters threw an error, n_clusters = 10 


        murine_celllens = CellLENS(murine_dataset,
                                device,
                                cnn_model= 'LITE', ### NOTE here set to no cnn model
                                cnn_latent_dim=128,
                                gnn_latent_dim=32
                                ) # generally these parameters no need to change

        if log == 'long':
            logging.info('training lens embedding')
        #training
        start = time.time()
        murine_celllens.get_lens_embedding(round=5,
                                        k=32,
                                        learning_rate=1e-3,
                                        n_epochs=5000,
                                        loss_fn='MSELoss',
                                        OptimizerAlg='Adam',
                                        optimizer_kwargs={},
                                        SchedulerAlg=None,
                                        scheduler_kwargs={},
                                        verbose=True) # generally these parameters do not need to change
        elapsed = time.time() - start
        if log == 'long':
            logging.info(f"LENS training took {elapsed/60:.2f} minutes")
        if log == 'short':
            logging.info(f'Fold {iteration} LENS_time, {elapsed:.2f}')
        #append time to the results
        text_path = os.path.join(output_path, 'CellLENS_Lite', 'fold_times.txt')
        os.makedirs(os.path.dirname(text_path), exist_ok=True)
        mode = 'w' if iteration == 1 else 'a'
        with open(text_path, mode) as f:
            f.write(f"Fold {iteration} LENS_time, {elapsed:.2f}\n")

        #save embedding
        embedding = pd.DataFrame(murine_celllens.lens_embedding)
        os.makedirs(f'{output_path}/CellLENS_Lite/it{iteration}', exist_ok=True)  # Ensure the directory exists
        embedding.to_csv(f'{output_path}/CellLENS_Lite/it{iteration}/celllens_LITE_embedding.csv', index=False)

        #leiden clustering
        murine_celllens.get_lens_clustering(neighbor=15, # standard leiden parameter
                                            resolution=1.0, # resolution for leiden - specific
                                            entropy_threshold=0.75, # CellLENS parameter - generally no change needed
                                            concen_threshold=1, # CellLENS parameter - generally no change needed
                                            max_breaks=3, # CellLENS parameter - generally no change needed
                                            size_lim=50 # CellLENS parameter - smallest cluster size allowed
                                        )

        #save clustering
        cluster_labels = pd.Series(murine_celllens.lens_clustering, name="Cluster_Labels")

        cluster_df = murine_dataset.df.copy()  # Create a copy to avoid modifying the original
        cluster_df["Cluster_Labels"] = cluster_labels

        ## greedy assignement
        for k in granularity:
            if log == 'long':
                logging.info(f"Running greedy assignment for {k}")
            ## greedy assignement
            results = greedy_f1_score(cluster_df, k, 'Cluster_Labels', tie_strategy='random')
            output = cluster_df.copy()
            output['predicted_phenotype'] = results["mapped_predictions"]
            output = output.rename(columns={k: "true_phenotype"})
            name = granularity[granularity==k].index[0]
            path = f"{output_path}/CellLENS_Lite/{name}/predictions_{iteration}.csv" #path to output directory
            os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
            if log == 'long':
                logging.info(f"Saving predictions to {path}")            
            output.to_csv(path, index=False)


def main():
    if args.log != 'off':
        logging_setup(args.output_path)
    lite_celllens(args.input, args.markers, args.iterations, args.output_path, args.log, args.path2crops)

if __name__ == "__main__":
    main()

#example usage: python3 chL2_LITE_CellLENS.py -i ../../Maps_data/cHL_2_MIBI/quantification/processed/cHL_2_MIBI_quantification.csv -pc ../chl2/processed_images -o ../cHL2/ -m CD163 CD11c CD28 -l long -it 1
#immucan markers:
#MPO SMA CD16 CD38 HLADR CD27 CD15 CD45RA CD163 B2M CD20 CD68 Ido1 CD3 LAG3 CD11c PD1 PDGFRb CD7 GrzB PDL1 TCF7 CD45RO FOXP3 ICOS CD8a CarbonicAnhydrase CD33 Ki67 VISTA CD40 CD4 CD14 Ecad CD303 CD206 cleavedPARP ## nuclear: HistoneH3
#chl2 markers:
#CD45 CD20 'pSLP-76' 'SLP-76' 'anti-H2AX (pS139)' CD163 CD45RO CD28 'CD153 (CD30L)' Lag3 CD4 CD11c CD56 FoxP3 GATA3 'Granzyme B' 'PD-L1' CD16 'Ki-67' 'PD-1' 'Pax-5' Tox CD161 CD68 'B2-Microglobulin' CD8 CD3 HLA1 CD15 Tbet CD14 CD123 CXCR5 CD45RA 'HLA-DR' CD57 'IL-10' CD30 TIM3 RORgT TCRgd CD86 CD25 'Na-K ATPase' ## nuclear: dsDNA 'Histone H3'
