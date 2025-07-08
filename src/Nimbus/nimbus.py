# import required packages
import warnings
warnings.simplefilter("ignore")
import os
#from IPython.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))
from nimbus_inference.nimbus import Nimbus#, prep_naming_convention
from nimbus_inference.utils import MultiplexDataset
from alpineer import io_utils
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd

from greedy_f1_utils import greedy_f1_score
import logging
import time
import argparse


### TODO write instructions for how data needs to be structured
#Parse arguments from command line TODO change descriptions to match
parser = argparse.ArgumentParser(description='Leiden clustering with greedy assignment and optional preprocessing.')
parser.add_argument('-i', '--input', dest='input', type = str, help='Path to input files') ### TODO change help
parser.add_argument('-s', '--segmentation', dest='seg_mask_dir', type = str, help='Path segmentation masks') ### TODO change help
parser.add_argument('-o', '--output', dest='output_path', type = str, help='Path to output folder (usually ends with dataset name). Do not put a slash in the end!')
parser.add_argument('-m', '--markers', dest='markers', nargs='+', help='List of marker names to use for clustering')
parser.add_argument('--fov', dest='fovs', nargs='+',default=None, type=str, help='List of specific fovs to use for clustering. If not provided, all fovs in the input directory will be used.')
parser.add_argument('--input_type', dest='input_type',required= False, default='single', choices=['single', 'ome'], help='Is input single channel tiff folders (single) or .ome.tiff files (ome)? Default: single channel tiff folders')
parser.add_argument('--mask_type', dest='mask_type',required= False, default='tiff', choices=['tiff', 'tif'], help='Are masks in tiff or tif format? Default: tiff')
parser.add_argument('--image_suffix', dest='image_suffix',required= False, default='tiff', choices=['tiff', 'tif', 'ome.tif', 'ome.tiff'], help='Suffix of image files. Choices: tiff, tif, ome.tif, ome.tiff. Default: tiff')
parser.add_argument('-l', '--log', dest='log', required= False, default='long', choices=['short', 'long', 'off'], help='Logging level: short, long, or off (default: long)')
parser.add_argument('-it', '--iterations', dest='iterations', type=int, required= False, default=5, help='Number of iterations to run (default: 5)')
parser.add_argument('-lc', '--leiden_clustering', dest='leiden_clustering', type=bool, required= False, default=True, help='Whether to perform Leiden clustering (default: True)')
args = parser.parse_args()

# define parameters
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
        filename=f'{output_path}/logs/{time.strftime("%Y-%m-%d")}_nimbus.log',
        filemode="w",  # overwrite on each run; change to "a" to append
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

# Define function to prepare naming convention for segmentation data
def prep_naming_convention(mask_dir):
    """Prepares the naming convention for the segmentation data.

    Args:
        mask_dir (str): path to directory where segmentation data is saved
    Returns:
        function: function that returns the path to the
            segmentation data for a given fov
    """

    def segmentation_naming_convention(fov_path):
        """Prepares the path to the segmentation data for a given fov

        Args:
            fov_path (str): path to fov
        Returns:
            str: paths to segmentation fovs
        """
        if args.input_type == 'ome':
            fov_name = os.path.basename(fov_path).replace(".ome.tiff", "")
            if args.mask_type == 'tif':
                return os.path.join(mask_dir, fov_name + ".tif")
            elif args.mask_type == 'tiff':
                return os.path.join(mask_dir, fov_name + ".tiff")
        else:  # single channel tiffs
            fov_name = os.path.basename(fov_path)
            if args.mask_type == 'tif':
                return os.path.join(mask_dir, fov_name + ".tif")
            elif args.mask_type == 'tiff':
                return os.path.join(mask_dir, fov_name + ".tiff")
    
    return segmentation_naming_convention

# Define Leiden clustering with greedy assignment function
def leiden_with_greedy(adata, df, output_base, log, iteration):
    start = time.time()
    sc.tl.leiden(adata, resolution=0.9, key_added='leiden_res_0_9')
    elapsed = time.time() - start
    if log == 'long':
        logging.info(f'Leiden clustering with resolution = 0.9 took {elapsed/60:.2f} minutes')
    if log == 'short':
        logging.info(f'Resolution=0.9: {elapsed/60:.2f} min')
    output = df.copy()
    output[f'leiden_res_0_9'] = adata.obs[f'leiden_res_0_9'].values
    resolution = f'leiden_res_0_9'

    for k in granularity:
        results = greedy_f1_score(output, k, resolution, tie_strategy='random')
        output['predicted_phenotype'] = results['mapped_predictions']
        output = output.rename(columns={k: "true_phenotype"})
        name = granularity[granularity==k].index[0]
        path = f"{output_base}/{name}/predictions_{iteration}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
        if log == 'long':
            logging.info(f"Saving predictions to {path}")
        output.to_csv(path, index=False)

# Define function to run nimbus inference
def run_nimbus(input_dir, mask_dir, output_base, markers, fov, img_suffix, iterations, log, leiden_clustering):
    """Function to run nimbus inference on the input data."""

    # Check if paths exist
    print("checking input and mask paths")
    io_utils.validate_paths([input_dir, mask_dir])

    # define the channels to include
    print("defining channels to include")
    include_channels = markers
    include_channels = ['CD14', 'HLA1', 'GATA3', 'HLA-DR', 'CXCR5', 'CD15', 'Pax-5', 'anti-H2AX (pS139)', 'CD45', 
                    'RORgT', 'CD86', 'CD123', 'CD28', 'Granzyme B', 'CD68', 'Tbet', 'CD163', 'FoxP3', 'CD11c', 
                    'Tox', 'CD25', 'TCRgd', 'PD-1', 'CD45RO', 'CD30', 'CD8', 'CD4', 'SLP-76', 'Histone H3', 
                    'CD45RA', 'pSLP-76', 'CD161', 'CD153 (CD30L)', 'Lag3', 'IL-10', 'CD16', 'Na-K ATPase', 'B2-Microglobulin', 
                    'CD57', 'CD20', 'TIM3', 'Ki-67', 'PD-L1', 'CD56', 'CD3']

    # either get all fovs in the folder...
    print("getting fovs")
    if fov is not None:
        fov_names = fov
    else:
        fov_names = os.listdir(input_dir)
    # make sure to filter paths out that don't lead to FoVs, e.g. .DS_Store files.
    fov_names = [fov_name for fov_name in fov_names if not fov_name.startswith(".")]
    # construct paths for fovs
    fov_paths = [os.path.join(input_dir, fov_name) for fov_name in fov_names]

    # Prepare segmentation naming convention that maps a fov_path to the according segmentation label map
    segmentation_naming_convention = prep_naming_convention(mask_dir)

    # Test segmentation_naming_convention
    print("checking segmentation naming convention")
    if not os.path.exists(segmentation_naming_convention(fov_paths[0])):
        logging.info(f"Segmentation data does not exist for {fov_paths[0]} or naming convention is incorrect")
        raise FileNotFoundError(
            f"Segmentation data does not exist for {fov_paths[0]} or naming convention is incorrect"
        )
    
    for iteration in range(1, iterations+1):
        # Create nimbus output directory
        print(f"Creating output directory for iteration {iteration}")

        output_dir = os.path.join(output_base, f"outputs/iteration_{iteration}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"Output directory created: {output_dir}")
        start = time.time()
        print("Creating MultiplexDataset")
        dataset = MultiplexDataset(
            fov_paths=fov_paths,
            suffix=f'.{img_suffix}', # or .png, .jpg, .jpeg, .tif or .ome.tiff
            include_channels=include_channels,
            segmentation_naming_convention=segmentation_naming_convention,
            output_dir=output_dir
        )
        print("running Nimbus")
        nimbus = Nimbus(
            dataset=dataset,
            output_dir=output_dir,
            save_predictions=True,
            batch_size=4,
            test_time_aug=True,
            device="auto",
            input_shape=[1024,1024]
        )

        # check if all inputs are valid
        print("Checking inputs")
        nimbus.check_inputs() 
        print("Preparing normalization dictionary")
        dataset.prepare_normalization_dict(
            quantile=0.999,
            n_subset=50,
            clip_values=(0, 2),
            multiprocessing=True,
            overwrite=True
        )
        print("Training Nimbus")
        cell_table = nimbus.predict_fovs()

        elapsed = time.time() - start
        if log == 'long':
            logging.info(f'Nimbus training took {elapsed/60:.2f} minutes')
        if log == 'short':
            logging.info(f'Nimbus: {elapsed/60:.2f} min')
        print("Nimbus training completed")
        cell_table.to_csv(os.path.join(output_dir, f"nimbus_cell_table.csv"), index=False) 

        if leiden_clustering:
            print("Creating AnnData object and clustering with Leiden and greedy assignment")
             # Create AnnData object
            table = cell_table[include_channels]
            print(table.head())
            obs = cell_table.drop(columns=include_channels)
            print(obs.head())
            adata = ad.AnnData(X=table, obs=obs)
            # Run Leiden clustering with greedy assignment
            leiden_with_greedy(adata, cell_table, output_base, log, iteration)



def main():
    if args.log != 'off':
        logging_setup(args.output_path)
    run_nimbus(args.input, args.seg_mask_dir, args.output_path, args.markers, args.fovs, args.image_suffix, args.iterations, args.log, args.leiden_clustering)


if __name__ == "__main__":
    main()


#example usage: python3 nimbus.py -i /Volumes/Gut_Project/pheno_benchmark/Maps_data/cHL_2_MIBI/raw_image -o /Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/nimbus/nimbus_test -s /Volumes/Gut_Project/pheno_benchmark/Maps_data/cHL_2_MIBI/segmentation/masks -m CD163 CD11c CD28 -l short --mask_type tif -it 1 -lc False

