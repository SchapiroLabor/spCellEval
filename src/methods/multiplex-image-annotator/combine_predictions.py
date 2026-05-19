import pandas as pd
import argparse
import os
import re
import json


def combine_results(results_dir, batch_csv_path, quantification_path, image_sample_col, output_dir, rename_rules=None, strip_ext=False):
    batch_csv = pd.read_csv(batch_csv_path)
    quantification = pd.read_csv(quantification_path)
    if rename_rules:
        with open(rename_rules, 'r') as f:
            rename_dict = json.load(f)
        quantification[image_sample_col] = quantification[image_sample_col].astype(str).replace(rename_dict)
    n_expected = len(batch_csv)
    dfs = []
    found_indices = []
    for anot_filename in os.listdir(results_dir):
        match = re.search(r'_annotation_(\d+)\.csv$', anot_filename)
        if match:
            index_from_file = int(match.group(1))
            print(f"Processing annotation file: {anot_filename} for index: {index_from_file}")
            annotation = pd.read_csv(os.path.join(results_dir, anot_filename))
            annotation = annotation.rename(columns={'Cell Index': 'cell_id', 'Cell Type': 'predicted_phenotype'})
            annotation['cell_id'] = annotation['cell_id'].astype(int)
            image_path_string = batch_csv['image_path'][index_from_file]
            image_name = os.path.basename(image_path_string)
            stem = os.path.splitext(image_name)[0]
            image_key = os.path.splitext(stem)[0] if strip_ext else image_name
            annotation['unique_id'] = image_key + '_' + annotation['cell_id'].astype(str)
            annotation['predicted_phenotype'] = annotation['predicted_phenotype'].replace({'B cell': 'B_cell', 'Others': 'undefined', 'Dendritic cell': 'Dendritic_cell', 'CD8 T cell': 'CD8+_T_cell', 'CD4 T cell': 'CD4+_T_cell',
                                                'M2 macrophage cell': 'M2_Macrophage', 'M1 macrophage cell': 'M1_Macrophage', 'Natural killer cell': 'NK_cell',
                                                'Plasma cell': 'Plasma_cell', 'Regulatory T cell': 'Treg', 'Granulocyte cell': 'Neutrophil', 'Mast cell': 'Mast_cell',
                                                'Endothelial cell': 'Endothelial', 'Epithelial cell': 'Epithelial', 'Proliferating/tumor cell': 'Cancer', 'Smooth muscle': 'Stromal',
                                                'Stroma cell': 'Stromal'})
            dfs.append(annotation[['unique_id', 'predicted_phenotype']])
            found_indices.append(index_from_file)

    found_indices_sorted = sorted(found_indices)
    expected_indices = list(range(n_expected))
    if found_indices_sorted != expected_indices:
        missing = sorted(set(expected_indices) - set(found_indices))
        extra = sorted(set(found_indices) - set(expected_indices))
        raise ValueError(
            f"Annotation file indices do not match batch CSV rows.\n"
            f"  Expected indices: {expected_indices}\n"
            f"  Found indices:    {found_indices_sorted}\n"
            f"  Missing: {missing}  |  Unexpected: {extra}\n"
            f"RIBCA likely skipped one or more images — cell-to-image mapping would be corrupted."
        )
    print(f"Sanity check passed: found {len(found_indices)} annotation files matching {n_expected} batch CSV rows.")

    df = pd.concat(dfs, ignore_index=True)
    quant_image_key = quantification[image_sample_col].apply(lambda x: os.path.splitext(x)[0]) if strip_ext else quantification[image_sample_col]
    quantification['unique_id'] = quant_image_key + '_' + quantification['cell_id'].astype(str)
    result = pd.merge(quantification, df, on='unique_id', how='left')
    result = result.rename(columns={'cell_type': 'true_phenotype'})
    result.drop(columns=['unique_id'], inplace=True)
    result['predicted_phenotype'] = result['predicted_phenotype'].fillna('undefined')
    result.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description="Combine results from ribca output files.")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the directory containing ribca annotation results.")
    parser.add_argument("--batch_csv_path", type=str, required=True, help="Path to the batch CSV file.")
    parser.add_argument("--quantification_path", type=str, required=True, help="Path to the quantification CSV file.")
    parser.add_argument("--image_sample_col", type=str, required=True, help="Column name in the quantification table that contains image sample identifiers for matching with the actual image basenames used by annotation file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for combined results.")
    parser.add_argument("--rename_image_sample_col", type=str, help="Path to a json file with renaming rules for image sample identifiers. If provided, will rename the image sample identifiers in the quantification table before merging with annotation results.")
    parser.add_argument("--strip_ext", action="store_true", help="Strip file extensions from image identifiers on both sides before matching. Useful when the quantification table uses a different extension than the image files (e.g., .csv vs .tiff).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    combine_results(args.results_dir, args.batch_csv_path, args.quantification_path, args.image_sample_col, args.output_dir, args.rename_image_sample_col, args.strip_ext)

if __name__ == "__main__":
    main()