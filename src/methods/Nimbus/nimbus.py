# import required packages
import warnings
warnings.simplefilter("ignore")
import os
from nimbus_inference.nimbus import Nimbus
from nimbus_inference.utils import MultiplexDataset
from alpineer import io_utils
import logging
import time
import argparse


### TODO write instructions for how data needs to be structured
parser = argparse.ArgumentParser(description='Nimbus Inference of marker expression in multiplexed imaging data. This script runs Nimbus on the input images and outputs new images as well as a quantification table.')
parser.add_argument('-i', '--input', dest='input', type = str, help='Path to input folder. Should contain either multi-channel .ome.tiffs or folders with single-channel tiffs for each FOV.')
parser.add_argument('-s', '--segmentation', dest='seg_mask_dir', type = str, help='Path to segmentation masks. This should be a folder with segmentation masks for each fov. The masks should be named the same as the fovs in the input folder, e.g. if the input folder contains a fov called "fov1.tiff", the mask should be called "fov1.tiff" or "fov1.tif" depending on the mask_type argument.')
parser.add_argument('-o', '--output', dest='output_path', type = str, help='Path to output folder, usually ends with dataset name. Nimbus folder and log folder will be created here.')
parser.add_argument('-m', '--markers', dest='markers', nargs='+', help='List of channels to apply Nimbus to')
parser.add_argument('--fov', dest='fovs', nargs='+',default=None, type=str, help='List of specific fovs to apply Nimbus to. If not provided, all fovs in the input directory will be used.')
parser.add_argument('--input_type', dest='input_type',required= False, default='single', choices=['single', 'ome'], help='Is input single-channel tiff folders (single) or .ome.tiff files (ome)? Default: single-channel tiff folders')
parser.add_argument('--mask_type', dest='mask_type',required= False, default='tiff', choices=['tiff', 'tif'], help='Are masks in tiff or tif format? Default: tiff')
parser.add_argument('--image_suffix', dest='image_suffix',required= False, default='tiff', choices=['tiff', 'tif', 'ome.tif', 'ome.tiff'], help='Suffix of image files. Choices: tiff, tif, ome.tif, ome.tiff. Default: tiff')
parser.add_argument('-l', '--log', dest='log', required= False, default='off', choices=['short', 'long', 'off'], help='Logging level: short, long, or off (default: long)')
parser.add_argument('-it', '--iterations', dest='iterations', type=int, required= False, default=5, help='Number of iterations to run (default: 5)')
args = parser.parse_args()

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

# Define function to run nimbus inference
def run_nimbus(input_dir, mask_dir, output_base, markers, fov, img_suffix, iterations, log):
    """Function to run nimbus inference on the input data."""
    # Check if paths exist
    io_utils.validate_paths([input_dir, mask_dir])

    # define the channels to include
    include_channels = markers

    # either get all fovs in the folder...
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
    if not os.path.exists(segmentation_naming_convention(fov_paths[0])):
        raise FileNotFoundError(
            f"Segmentation data does not exist for {fov_paths[0]} or naming convention is incorrect"
        )
    
    for iteration in range(1, iterations+1):
        # Create nimbus output directory
        output_dir = os.path.join(output_base, f"nimbus/outputs/iteration_{iteration}")
        os.makedirs(output_dir, exist_ok=True)

        start = time.time()
        dataset = MultiplexDataset(
            fov_paths=fov_paths,
            suffix=f'.{img_suffix}',
            include_channels=include_channels,
            segmentation_naming_convention=segmentation_naming_convention,
            output_dir=output_dir
        )

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
        nimbus.check_inputs() 
        dataset.prepare_normalization_dict(
            quantile=0.999,
            n_subset=50,
            clip_values=(0, 2),
            multiprocessing=True,
            overwrite=True
        )
        cell_table = nimbus.predict_fovs()

        elapsed = time.time() - start
        if log == 'long':
            logging.info(f'Nimbus training took {elapsed/60:.2f} minutes')
        if log == 'short':
            logging.info(f'Nimbus: {elapsed/60:.2f} min')

        text_path = os.path.join(output_base, "nimbus", 'fold_times.txt')
        os.makedirs(os.path.dirname(text_path), exist_ok=True)
        mode = 'w' if iteration == 1 else 'a'
        with open(text_path, mode) as f:
            f.write(f"Fold {iteration} prediction_time, {elapsed:.2f}\n")
        cell_table.to_csv(os.path.join(output_dir, f"nimbus_cell_table.csv"), index=False) 

def main():
    if args.log != 'off':
        logging_setup(args.output_path)
    run_nimbus(args.input, args.seg_mask_dir, args.output_path, args.markers, args.fovs, args.image_suffix, args.iterations, args.log)

if __name__ == "__main__":
    main()

#example usage: python3 nimbus.py -i /Volumes/Gut_Project/pheno_benchmark/Maps_data/cHL_2_MIBI/raw_image -o /Users/margotchazotte/Documents/uni/PhD/pheno_benchmark/chl2 -s /Volumes/Gut_Project/pheno_benchmark/Maps_data/cHL_2_MIBI/segmentation/masks -m CD163 CD11c CD28 -l long --mask_type tif -it 1
#immucan markers:
#MPO SMA CD16 CD38 HLADR CD27 CD15 CD45RA CD163 B2M CD20 CD68 Ido1 CD3 LAG3 CD11c PD1 PDGFRb CD7 GrzB PDL1 TCF7 CD45RO FOXP3 ICOS CD8a CarbonicAnhydrase CD33 Ki67 VISTA CD40 CD4 CD14 Ecad CD303 CD206 cleavedPARP ## nuclear: HistoneH3
#chl2 markers:
#CD45 CD20 'pSLP-76' 'SLP-76' 'anti-H2AX (pS139)' CD163 CD45RO CD28 'CD153 (CD30L)' Lag3 CD4 CD11c CD56 FoxP3 GATA3 'Granzyme B' 'PD-L1' CD16 'Ki-67' 'PD-1' 'Pax-5' Tox CD161 CD68 'B2-Microglobulin' CD8 CD3 HLA1 CD15 Tbet CD14 CD123 CXCR5 CD45RA 'HLA-DR' CD57 'IL-10' CD30 TIM3 RORgT TCRgd CD86 CD25 'Na-K ATPase' ## nuclear: dsDNA 'Histone H3'

