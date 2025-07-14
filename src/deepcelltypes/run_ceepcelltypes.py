import deepcell_types
import pandas as pd
import numpy as np
import os
import tifffile as tiff
import time
import argparse


def run_ceepcelltypes(
    image_seg_pairs,
    marker_path,
    quant_path,
    mpp,
    model_name,
    output_dir,
    device,
    num_data_loader_threads,
    n_runs,
):

    try:
        with open(marker_path) as f:
            chnames = f.read().splitlines()
    except Exception as e:
        print(f"Error reading marker file: {e}")
        return

    master_quant = pd.read_csv(quant_path)
    master_quant["sample_id"] = master_quant["sample_id"].astype(str)
    master_quant["cell_id"] = master_quant["cell_id"].astype(int)

    reformatting_logged = False
    inference_times = []

    for n in range(n_runs):
        quant = master_quant.copy()
        start = time.time()
        prediction_results_list = []

        for image, segmask in image_seg_pairs:
            img = tiff.imread(image)
            seg = tiff.imread(segmask)
            filename_with_ext = os.path.basename(image)
            img_name, _ = os.path.splitext(filename_with_ext)

            # Check for shape and otherwise reformat
            if img.ndim != 3:
                raise ValueError(
                    f"Image {image} is not a 3D image. Expected shape (c, h, w), got {img.shape}."
                )
            dim1, dim2, dim3 = img.shape
            if dim3 < dim1 and dim3 < dim2:
                img = img.transpose(2, 0, 1)
                if not reformatting_logged:
                    print(
                        f"Reformatted image {image} from shape {img.shape} to (c, h, w)."
                    )
                    reformatting_logged = True

            cell_types = deepcell_types.predict(
                img,
                seg,
                chnames,
                mpp,
                model_name=model_name,
                device_num=device,
                num_workers=num_data_loader_threads,
            )

            cell_types_df = pd.DataFrame(
                {"predicted_phenotype": cell_types, "batch_id": img_name}
            )
            cell_types_df.index = range(1, len(cell_types_df) + 1)
            cell_types_df.reset_index(inplace=True)
            cell_types_df.rename(columns={"index": "cell_index"}, inplace=True)
            prediction_results_list.append(cell_types_df)

        end = time.time()
        elapsed_time = end - start
        inference_times.append(elapsed_time)

        combined_predictions = pd.concat(prediction_results_list, ignore_index=True)

        combined_predictions["batch_id"] = combined_predictions["batch_id"].astype(str)
        combined_predictions["cell_index"] = combined_predictions["cell_index"].astype(
            int
        )

        final_quant_table = pd.merge(
            quant,
            combined_predictions,
            how="left",
            left_on=["sample_id", "cell_id"],
            right_on=["batch_id", "cell_index"],
        )
        final_quant_table.drop(columns=["batch_id", "cell_index"], inplace=True)
        final_quant_table["predicted_phenotype"] = final_quant_table[
            "predicted_phenotype"
        ].replace(
            {
                "Tumor": "Cancer",
                "CD8T": "CD8+_T_cell",
                "CD4T": "CD4+_T_cell",
                "BloodVesselEndothelial": "Endothelial",
                "NK": "NK_cell",
                "Dendritic": "Dendritic_cell",
                "Plasma": "Plasma_cell",
                "Bcell": "B_cells",
                "Mast": "Mast_cell",
                "LymphaticEndothelial": "Lymphatic",
                "Myofibroblast": "Myofibroblasts",
                "SmoothMuscle": "muscle",
                "Nerve": "Neuronal",
                "Erythrocyte": "Blood",
                "EVT": "Trophoblast",
            }
        )
        final_quant_table.to_csv(
            os.path.join(output_dir, f"predictions_{n}.csv"), index=False
        )
        print(
            f"Run {n+1}/{n_runs} completed in {elapsed_time:.2f} seconds. Predictions saved to {output_dir}."
        )

    with open(os.path.join(output_dir, "fold_times.txt"), "w") as f:
        for i, elapsed in enumerate(inference_times):
            f.write(f"Fold {i+1} inference_time: {elapsed:.2f}\n")


def main():
    parser = argparse.ArgumentParser(description="Run DeepCellTypes on a dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_dirs",
        nargs=2,
        metavar=("IMAGE_DIR", "SEG_DIR"),
        help="Paths to the directories for input images and segmentation masks Should be executed from command line like this: --input_dirs /path/to/images /path/to/segmentations",
    )
    group.add_argument(
        "--data_paths",
        type=str,
        help='Path to a CSV file with "image_path" and "seg_path" columns listing all pairs of images and masks.',
    )
    parser.add_argument(
        "--marker_path", type=str, required=True, help="Path to the marker txt file"
    )
    parser.add_argument(
        "--quant_path",
        type=str,
        required=True,
        help="Path to the output quantification CSV file. IMPORTANT: A image_id column needs to be present that contains the image file basenames for mapping predictions",
    )
    parser.add_argument(
        "--mpp",
        type=float,
        required=True,
        help="Microns per pixel (mpp) for the images",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the DeepCellTypes model to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (default: cuda)",
    )
    parser.add_argument(
        "--num_data_loader_threads",
        type=int,
        default=4,
        help="Number of threads for the data loader (default: 4)",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of runs for the model (default: 1)",
    )
    args = parser.parse_args()

    image_seg_pairs = []
    if args.input_dirs:
        image_dir, seg_dir = args.input_dirs
        for f in sorted(os.listdir(image_dir)):
            if f.endswith(".tif") or f.endswith(".tiff"):
                image_path = os.path.join(image_dir, f)
                seg_path = os.path.join(seg_dir, f)
                if os.path.exists(seg_path):
                    image_seg_pairs.append((image_path, seg_path))
                else:
                    print(f"Segmentation file for {f} not found in {seg_dir}")

    elif args.data_paths:
        try:
            df = pd.read_csv(args.data_paths)
            if "image_path" not in df.columns or "seg_path" not in df.columns:
                parser.error(
                    "CSV file must contain 'image_path' and 'seg_path' columns."
                )
            image_seg_pairs = list(zip(df["image_path"], df["seg_path"]))
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
        n_runs=args.n_runs,
    )


if __name__ == "__main__":
    main()