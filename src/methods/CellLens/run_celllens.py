import os
import sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)
import glob
import time
import random
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from skimage.io import imread
from celllens import graph as lens_graph
from celllens.datasets import LENS_Dataset
from celllens.celllens import CellLENS
from utils.greedy_f1_utils import greedy_f1_score
warnings.filterwarnings('ignore')

#Parse arguments from command line
parser = argparse.ArgumentParser(description='CellLENS (Lite or Full) with greedy assignment.')
parser.add_argument('-mo', '--mode', dest='mode', required=True, choices=['lite', 'full'], help='lite: markers + cell locations; full: additionally a CNN on per-cell image crops')
parser.add_argument('-i', '--input', dest='input', type=str, required=True, help='Path to input quantification CSV file')
parser.add_argument('-o', '--output', dest='output_path', type=str, required=True, help='Path to output folder (usually ends with dataset name). Do not put a slash in the end!')
parser.add_argument('-s', '--split_col_name', dest='split_col_name', type=str, required=True, help='First non-marker column; all columns before it are used as markers (e.g. cell_id)')
parser.add_argument('-e', '--exclude_markers', dest='exclude_markers', nargs='+', default=[], help='Marker names to exclude (e.g. DAPI)')
parser.add_argument('-ic', '--image_col', dest='image_col', type=str, default='sample_id', help='Column with the image of each cell; the image file is <value without extension><image_ext> (default: sample_id)')
parser.add_argument('-nc', '--init_n_clusters', dest='init_n_clusters', type=int, default=10, help='Number of initial Leiden clusters on the markers (CellLENS searches the resolution), recommended 8-15 (default: 10)')
parser.add_argument('-pi', '--path2img', dest='path2img', type=str, help='full only: folder with one (2, H, W) nuclear/membrane max projection per image')
parser.add_argument('-ie', '--image_ext', dest='image_ext', type=str, default='.tiff', help='full only: file extension of the images (default: .tiff)')
parser.add_argument('-pc', '--path2crops', dest='path2crops', type=str, help='full only: folder for the per-cell crops (~0.5 MB per cell); reused if it holds one crop per cell')
parser.add_argument('-tr', '--truncation', dest='truncation', type=float, default=0.9, help='full only: quantile used to binarize each image channel (default: 0.9)')
parser.add_argument('-ce', '--cnn_epochs', dest='cnn_epochs', type=int, default=200, help='full only: training epochs of the LENS-CNN (default: 200)')
parser.add_argument('-ge', '--gnn_epochs', dest='gnn_epochs', type=int, default=5000, help='Training epochs of each of the 5 LENS-GNN rounds (default: 5000)')
parser.add_argument('-it', '--iterations', dest='iterations', type=int, default=5, help='Number of iterations to run (default: 5)')
parser.add_argument('-sd', '--seed', dest='seed', type=int, default=0, help='Random seed, iteration i uses seed + i (default: 0)')
parser.add_argument('-l', '--log', dest='log', default='off', choices=['short', 'long', 'off'], help='Logging level: short, long, or off (default: off)')
args = parser.parse_args()

# CellLENS parameters as used in the previous CellLENS run scripts
CROP_SIZE = 512  # the LENS-CNN architecture only works with 512 x 512 crops
NBHD_COMPOSITION = 20
NUM_WORKERS = 8

# Define logging setup function
def logging_setup(output_path, mode):
    os.makedirs(f'{output_path}/logs', exist_ok=True)
    logging.basicConfig(
        filename=f'{output_path}/logs/{time.strftime("%Y-%m-%d")}_celllens_{mode}.log',
        filemode="w",  # overwrite on each run; change to "a" to append
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

# Define preprocessing function
def preprocessing(input, split_col_name, exclude_markers, image_col, mode, path2img, image_ext):
    df = pd.read_csv(input)
    # Markers are all columns before split_col_name, minus excluded ones
    markers = df.columns[:df.columns.get_loc(split_col_name)]
    features_list = [m for m in markers if m not in exclude_markers]
    df.fillna(0, inplace=True)

    # CellLENS works on one image, so all images are placed side by side (in image_col category order)
    # with a gap of CROP_SIZE pixels, so that neither spatial neighbors nor crops reach into the next image
    image_ids = df[image_col].astype('category')
    stitched = None
    if mode == 'full':
        images = [imread(os.path.join(path2img, os.path.splitext(i)[0] + image_ext)) for i in image_ids.cat.categories]
        for name, img in zip(image_ids.cat.categories, images):
            if img.ndim != 3 or img.shape[0] != 2:
                raise ValueError(f"{name}: expected a (2, H, W) nuclear/membrane image, got shape {img.shape}")
        width = max(img.shape[2] for img in images)
        slot = width + CROP_SIZE
        stitched = np.zeros((max(img.shape[1] for img in images), slot * len(images), 2), dtype=images[0].dtype)
        for k, img in enumerate(images):
            stitched[:img.shape[1], k * slot:k * slot + img.shape[2]] = np.transpose(img, (1, 2, 0))
    else:
        slot = int(np.ceil(df['x'].max())) + CROP_SIZE
    # CellLENS locations are (row, column) of the stitched image: row = y, column = x + image offset
    df['x_new'] = df['y']
    df['y_new'] = image_ids.cat.codes.astype(int) * slot + df['x']  # codes are int8/int16, would overflow
    return df, features_list, stitched

# Define seeding function
_graph_clustering = lens_graph.graph_clustering
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CellLENS' initialize() runs its initial Leiden clustering with seed=None (different on every run), so the seed is injected
    lens_graph.graph_clustering = lambda *a, **kw: _graph_clustering(*a, **{**kw, 'seed': seed})

# Define crop preparation function (once per crop folder, the crops are the same for all iterations)
def prepare_crops(dataset, stitched, path2crops, truncation, log):
    n_crops = len(glob.glob(os.path.join(path2crops, 'img_*.npy')))
    if n_crops == dataset.n_cells:
        if log == 'long':
            logging.info(f'Reusing {n_crops} crops in {path2crops}')
        return
    if n_crops > 0:
        raise ValueError(f'{path2crops} holds {n_crops} crops but the dataset has {dataset.n_cells} cells, use an empty folder')
    start = time.time()
    dataset.prepare_images(stitched, CROP_SIZE, truncation, aggr=[[0], [1]], pad=1000, verbose=False)
    if log == 'long':
        logging.info(f'Image cropping took {(time.time() - start)/60:.2f} minutes')

# Define CellLENS with greedy assignment function
def celllens_with_greedy(df, features_list, stitched, mode, iterations, output_path, init_n_clusters, path2crops, truncation,
                         cnn_epochs, gnn_epochs, seed, log):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if log == 'long':
        logging.info(f"Using device: {device}")
    method_dir = f"{output_path}/CellLENS_{mode.capitalize()}"
    for iteration in range(1, iterations + 1):
        if log == 'long':
            logging.info(f"Running iteration {iteration} of {iterations} (seed {seed + iteration})")
        set_seed(seed + iteration)
        dataset = LENS_Dataset(df,
                               features_list=features_list,
                               nbhd_composition=NBHD_COMPOSITION,
                               feature_neighbor=15,
                               spatial_neighbor=15,
                               path2img=f'{path2crops}/' if mode == 'full' else None,
                               use_transform=False)
        # Initial Leiden clustering on the markers, its labels define the neighborhood composition; the resolution
        # is searched (bisection with tolerance 0.05), so the reached number can differ slightly from init_n_clusters
        dataset.initialize(cent_x="x_new",
                           cent_y="y_new",
                           celltype="feature_labels",
                           pca_components=min(25, len(features_list) - 1),
                           cluster_res=None,
                           n_clusters=init_n_clusters)
        if log == 'long':
            logging.info(f"{len(np.unique(dataset.feature_labels))} initial clusters (target {init_n_clusters})")
        if mode == 'full' and iteration == 1:
            prepare_crops(dataset, stitched, path2crops, truncation, log)

        model = CellLENS(dataset,
                         device,
                         cnn_model='CNN' if mode == 'full' else 'LITE',
                         cnn_latent_dim=128,
                         gnn_latent_dim=32)
        # Train time = CNN training + CNN embedding (full only) + LENS-GNN training
        start = time.time()
        if mode == 'full':
            model.fit_lens_cnn(batch_size=64,
                               learning_rate=1e-4,
                               n_epochs=cnn_epochs,
                               loss_fn='MSELoss',
                               OptimizerAlg='Adam',
                               optimizer_kwargs={},
                               SchedulerAlg=None,
                               scheduler_kwargs={},
                               num_workers=NUM_WORKERS,  # might crash with 12
                               use_amp=True,
                               cnn_model='CNN',
                               print_every=1000)
            model.get_cnn_embedding(batch_size=512, path2result=f'{method_dir}/it{iteration}/CellLENS_saveouts')
        model.get_lens_embedding(round=5,
                                 k=32,
                                 learning_rate=1e-3,
                                 n_epochs=gnn_epochs,
                                 loss_fn='MSELoss',
                                 OptimizerAlg='Adam',
                                 optimizer_kwargs={},
                                 SchedulerAlg=None,
                                 scheduler_kwargs={},
                                 verbose=True)
        elapsed = time.time() - start
        if log == 'long':
            logging.info(f"Iteration {iteration}: training took {elapsed/60:.2f} minutes")
        if log == 'short':
            logging.info(f'Fold {iteration} train_time, {elapsed:.2f}')
        #append time to the results
        os.makedirs(method_dir, exist_ok=True)
        with open(os.path.join(method_dir, 'fold_times.txt'), 'w' if iteration == 1 else 'a') as f:
            f.write(f"Fold {iteration} train_time, {elapsed:.2f}\n")

        #save embedding (read by cluster_celllens_embedding.py for subclustering)
        os.makedirs(f'{method_dir}/it{iteration}', exist_ok=True)
        pd.DataFrame(model.lens_embedding).to_csv(f'{method_dir}/it{iteration}/celllens_{mode.upper()}_embedding.csv', index=False)

        #leiden clustering on the embedding; its UMAP is kept although unused: sc.tl.umap prunes weak edges of the
        #kNN graph in place, so the Leiden clustering would differ without it
        model.get_lens_clustering(neighbor=15,
                                  resolution=1.0,
                                  entropy_threshold=0.75,
                                  concen_threshold=1,
                                  max_breaks=3,
                                  size_lim=50)
        cluster_df = dataset.df.copy()
        cluster_df["cluster_label"] = model.lens_clustering

        ## greedy assignement on cell_type, saved under level3
        results = greedy_f1_score(cluster_df, 'cell_type', 'cluster_label', tie_strategy='random')
        output = cluster_df.copy()
        output['predicted_phenotype'] = results["mapped_predictions"]
        output = output.rename(columns={'cell_type': "true_phenotype"})
        path = f"{method_dir}/level3/predictions_{iteration}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if log == 'long':
            logging.info(f"Saving predictions to {path}")
        output.to_csv(path, index=False)

def main():
    if args.mode == 'full' and (args.path2img is None or args.path2crops is None):
        parser.error('--mode full needs --path2img and --path2crops')
    if args.log != 'off':
        logging_setup(args.output_path, args.mode)
    df, features_list, stitched = preprocessing(args.input, args.split_col_name, args.exclude_markers, args.image_col,
                                                args.mode, args.path2img, args.image_ext)
    celllens_with_greedy(df, features_list, stitched, args.mode, args.iterations, args.output_path, args.init_n_clusters,
                         args.path2crops, args.truncation, args.cnn_epochs, args.gnn_epochs, args.seed, args.log)


if __name__ == "__main__":
    main()

#example usage (cHL_2_MIBI, full):
#python3 run_celllens.py -mo full -i .../cHL_2_MIBI_quantification.csv -o .../results/cHL_2_MIBI -s cell_id -e dsDNA "Histone H3" \
#    -pi .../cHL_2_MIBI/raw_images/max_projection -pc .../crops/cHL_2_MIBI -l long
#example usage (IMMUcan, lite):
#python3 run_celllens.py -mo lite -i .../IMMUcan_quantification.csv -o .../results/IMMUcan -s image -e HistoneH3 DNA1 DNA2 -ic image
