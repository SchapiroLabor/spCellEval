import deepcell_types
import pandas as pd
import numpy as np
import os
import tifffile as tiff
import time
import argparse

def run_ceepcelltypes(image_seg_pairs, marker_path, quant_path, mpp, model_name, output_dir, device, num_data_loader_threads, n_runs):

    try:
        with open('/Users/lukashat/Documents/PhD_Schapiro/Projects/phenotype_benchmark/deepcelltypes/markers.txt') as f:
            chnames = f.read().splitlines()
    except Exception as e:
        print(f"Error reading marker file: {e}")
        return

    quant = pd.read_csv(quant_path)

    for n in range(n_runs):
        # UNDER CONSTRUCTION
    
                
        



def main():
    parser = argparse.ArgumentParser(description='Run DeepCellTypes on a dataset')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_dirs', nargs=2, metavar=('IMAGE_DIR', 'SEG_DIR'),
                       help='Paths to the directories for input images and segmentation masks Should be executed from command line like this: --input_dirs /path/to/images /path/to/segmentations')
    group.add_argument('--data_paths', type=str,
                       help='Path to a CSV file with "image_path" and "seg_path" columns listing all pairs of images and masks.')
    parser.add_argument('--marker_path', type=str, required=True, help='Path to the marker txt file')
    parser.add_argument('--quant_path', type=str, required=True, help='Path to the output quantification CSV file')
    parser.add_argument('--mpp', type=float, required=True, help='Microns per pixel (mpp) for the images')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the DeepCellTypes model to use')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (default: cuda)')
    parser.add_argument('--num_data_loader_threads', type=int, default=4, help='Number of threads for the data loader (default: 4)')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs for the model (default: 1)')
    args = parser.parse_args()

    image_seg_pairs = []
    if args.input_dirs:
        image_dir, seg_dir = args.input_dirs
        for f in sorted(os.listdir(image_dir)):
            if f.endswith('.tif') or f.endswith('.tiff'):
                image_path = os.path.join(image_dir, f)
                seg_path = os.path.join(seg_dir, f)
                if os.path.exists(seg_path):
                    image_seg_pairs.append((image_path, seg_path))
                else:
                    print(f"Segmentation file for {f} not found in {seg_dir}")
    
    elif args.data_paths:
        try:
            df = pd.read_csv(args.data_paths)
            if 'image_path' not in df.columns or 'seg_path' not in df.columns:
                parser.error("CSV file must contain 'image_path' and 'seg_path' columns.")
            image_seg_pairs = list(zip(df['image_path'], df['seg_path']))
        except Exception as e:
            parser.error(f"Error reading CSV file: {e}")
        

    run_ceepcelltypes(
        image_seg_pairs=image_seg_pairs,
        marker_path=args.marker_path,
        quant_path=args.quant_path,
        mpp=args.mpp,
        model_name=args.model_name,
        output_dir=args.output_dir,
        device=args.device,
        num_data_loader_threads=args.num_data_loader_threads,
        n_runs=args.n_runs
    )

if __name__ == "__main__":
    main()