import pandas as pd
import numpy as np
from celllens.utils import *
import os
import glob
from skimage.io import imread
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
parser = argparse.ArgumentParser(description='Running CellLENS on chL2.')
parser.add_argument('-i', '--input', dest='input', type = str, help='Path to input CSV file')
parser.add_argument('-pi', '--path2img', dest='path2image', type = str, help='Path to folder with images.')
parser.add_argument('-pc', '--path2crops', dest='path2crops', type = str, help='Path to folder with crops.')
parser.add_argument('-o', '--output', dest='output_path', type = str, help='Path to output folder (usually ends with dataset name). Do not put a slash in the end!')
parser.add_argument('-m', '--markers', dest='markers', nargs='+', help='List of marker names to use.')
parser.add_argument('-it', '--iterations', dest='iterations', type=int, required= False, default=5, help='Number of iterations to run (default: 5)')
parser.add_argument('-l', '--log', dest='log', required= False, default='long', choices=['short', 'long', 'off'], help='Logging level: short, long, or off (default: long)')
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
        filename=f'{output_path}/logs/{time.strftime("%Y-%m-%d")}_celllens.log',
        filemode="w",  # overwrite on each run; change to "a" to append
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

## define functions 
#read all images from a folder in the same order as in the dataframe
def read_images_from_folder(folder, image_shift_map):
    images = []
    for shift in sorted(image_shift_map.keys()):
        img_name = image_shift_map[shift]
        img_path = os.path.join(folder, img_name.replace('.csv', '.tiff'))
        img = imread(img_path)
        if img is not None:
            images.append(img)
    return images

#pad images to have empy space around them
def pad_image(image, target_shape=(3500, 3500)):
    pad_height = target_shape[0] - image.shape[0]
    pad_width = target_shape[1] - image.shape[1]
    if pad_height < 0 or pad_width < 0:
        raise ValueError("Target shape must be larger than the original shape.")
    return np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)





def full_celllens(path2image, input, markers, iterations, path2crops, output_path, log):
    # Set the device for PyTorch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if log == 'long':
        logging.info(f"Using device: {device}")
    # Define the folder where the images are stored
    folder = path2image

    # Define parameters
    size = 512 #using any size smaller than 512 results in error
    truncation = 0.9
    if log == 'long':
        logging.info(f"Parameters set: size={size}, truncation={truncation}")

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

    #to read in images in same order as stitched CSV
    #create list of unique images and their corresponding shift values
    unique_images = df["sample_id"].unique()
    unique_shifts = df["image_shift"].astype('int').unique()

    # Create a dictionary to map image names to their shift values
    image_shift_map = dict(zip(unique_shifts, unique_images))

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
                                    nbhd_composition=15,
                                    feature_neighbor=15,
                                    spatial_neighbor=15,
                                    path2img=f'{path2crops}/', # location to store cropped images
                                    use_transform = False)


        murine_dataset.initialize(cent_x="x_new", # x loc of cells
                                cent_y="y_new", # y loc of cells
                                celltype="feature_labels", # default option - run leiden to initialize the clusters
                                pca_components=25, # PCA components to use, user decide
                                cluster_res= 0.2 #, giving n_clusters threw an error, n_clusters = 10
                                ) # leiden initialization resolution

        # Read and crop images from the folder
        # note this will create a folder with single crops images (same number of cells)
        # this step (on this data) takes ~ 10mins.
        if not glob.glob(f'{path2crops}/img_*.npy'):

            if log == 'long':
                logging.info('Processing images...')
            
            #load images from the folder
            images = read_images_from_folder(folder, image_shift_map)
            #pad images to have the same size
            padded_images = []
            for i, img in enumerate(images):  # img.shape = (C, H, W)
                #print(f"Padding image {i}")
                padded_channels = [pad_image(channel) for channel in img]  # Each channel is (H, W)
                padded_img = np.stack(padded_channels)  # Shape = (C, 1000, 1000), C may vary
                padded_images.append(padded_img)

            #change order of dimensions to put channels in the back.
            for i, img in enumerate(padded_images): 
                padded_images[i] = np.transpose(img, (1, 2, 0))

            # Check all padded_images have the same number of channels
            channel_counts = [img.shape[2] for img in padded_images]
            if len(set(channel_counts)) != 1:
                raise ValueError("All images must have the same number of channels to tile like this.")

            # Stack vertically along height
            stitched_image = np.concatenate(padded_images, axis=1)  # axis=1 = width  

            start = time.time()
            murine_dataset.prepare_images(stitched_image,
                                        size,
                                        truncation,
                                        aggr = [[0], [1]], 
                                        pad=1000, # pad size at boundary 
                                        verbose=False)
            elapsed = time.time() - start
            if log == 'long':
                logging.info(f'Image preparation took {elapsed/60:.2f} minutes') ### TODO put if log != 'off'
        else:
            if log == 'long':
                logging.info('Images already processed, skipping this step')
        # note saving images of 50k cells will take up ~27gb space - after CNN training feel free to delete it.
        # chl2 has 230k cells, so this will take up ~ 120gb, expecting 50 minutes to save

        murine_celllens = CellLENS(murine_dataset,
                                device,
                                cnn_model= 'CNN', ### NOTE here set to no cnn model
                                cnn_latent_dim=128,
                                gnn_latent_dim=32
                                ) # generally these parameters no need to change
       
        if log == 'long':
            logging.info('Training CNN model')
        #I/O is the limiting step to training speed
        start = time.time()
        murine_celllens.fit_lens_cnn(
                            batch_size=64,
                            learning_rate=1e-4,
                            n_epochs=200, 
                            loss_fn='MSELoss',
                            OptimizerAlg='Adam',
                            optimizer_kwargs={},
                            SchedulerAlg=None,
                            scheduler_kwargs={},
                            num_workers=8, # number of workers for data loading, 8 should be safe, might crash with 12
                            use_amp = True,
                            cnn_model='CNN',
                            print_every=1000) # generally other parameters dont need to be changed - unless for specfic reason
        elapsed = time.time() - start
        if log == 'long':
            logging.info(f'Fold {iteration} CNN training done in {elapsed/60:.2f} minutes')
        if log == 'short':
            logging.info(f'Fold {iteration} CNN_time, {elapsed:.2f}')

        #get embedding
        murine_celllens.get_cnn_embedding(batch_size=512, # generally these parameters no need to change
                                        path2result=f'{output_path}/CellLENS_Full/it{iteration}/CellLENS_saveouts')
        if log == 'long':
            logging.info('CNN embedding done, now training lens embedding')
        #training
        start = time.time()        
        murine_celllens.get_lens_embedding(round=5, #took 103 minutes
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

        #save embedding
        embedding = pd.DataFrame(murine_celllens.lens_embedding)
        embedding.to_csv(f'{output_path}/CellLENS_Full/it{iteration}/celllens_FULL_embedding.csv', index=False)

        #leiden clustering
        if log == 'long':
            logging.info('Running clustering')      
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
            path = f"{output_path}/CellLENS_Full/{name}/predictions_{iteration}.csv" #create output directory
            os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
            if log == 'long':
                logging.info(f"Saving predictions to {path}")
            output.to_csv(path, index=False)

def main():
    if args.log != 'off':
        logging_setup(args.output_path)
    full_celllens(args.path2image, args.input, args.markers, args.iterations, args.path2crops, args.output_path, args.log)

if __name__ == "__main__":
    main()

#example usage: python3 chL2_Full_CellLENS.py -i ../../Maps_data/cHL_2_MIBI/quantification/processed/cHL_2_MIBI_quantification.csv -pi ../../Maps_data/cHL_2_MIBI/raw_image/max_projections -pc ../chl2/processed_images -o ../cHL2/ -m CD163 CD11c CD28 -l long -it 1
#immucan markers:
#MPO SMA CD16 CD38 HLADR CD27 CD15 CD45RA CD163 B2M CD20 CD68 Ido1 CD3 LAG3 CD11c PD1 PDGFRb CD7 GrzB PDL1 TCF7 CD45RO FOXP3 ICOS CD8a CarbonicAnhydrase CD33 Ki67 VISTA CD40 CD4 CD14 Ecad CD303 CD206 cleavedPARP ## nuclear: HistoneH3
#chl2 markers:
#CD45 CD20 'pSLP-76' 'SLP-76' 'anti-H2AX (pS139)' CD163 CD45RO CD28 'CD153 (CD30L)' Lag3 CD4 CD11c CD56 FoxP3 GATA3 'Granzyme B' 'PD-L1' CD16 'Ki-67' 'PD-1' 'Pax-5' Tox CD161 CD68 'B2-Microglobulin' CD8 CD3 HLA1 CD15 Tbet CD14 CD123 CXCR5 CD45RA 'HLA-DR' CD57 'IL-10' CD30 TIM3 RORgT TCRgd CD86 CD25 'Na-K ATPase' ## nuclear: dsDNA 'Histone H3'
