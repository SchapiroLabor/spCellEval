import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import os
import tribus
from visualization import z_score
import argparse
import time


def run_tribus(dataset_path, seed, n_runs, seed_stability_mode, granularity_level, columns_to_use, remove_cell_types,scaling,log1p, decision_matrix_path, normalization, tuning, sigma, learning_rate,
               clustering_threshold, undefined_threshold, other_threshold, depth, remove_result_cell_types, output_path):
    
    sample_data = pd.read_csv(dataset_path)

    target_col = granularity_level
    if target_col == 'level3':
        target_col = 'cell_type'
    elif target_col == 'level2':
        target_col = 'level_2_cell_type'
    elif target_col == 'level1':
        target_col = 'level_1_cell_type'
    print(f'selected granularity level {target_col}')

    colums_to_use_clean = columns_to_use.strip()
    if ',' in colums_to_use_clean:
        columns_to_use = [s.strip() for s in colums_to_use_clean.split(',')]
    else:
        columns_to_use = colums_to_use_clean
    cols = list(columns_to_use)
    Q = sample_data[cols].quantile(0.999)
    sample_data = sample_data[~((sample_data[cols] > Q)).any(axis=1)]
    if scaling is not None:
        sample_data[cols] = sample_data[cols] * scaling
    if log1p:
        sample_data[cols] = np.log1p(sample_data[cols])

    if remove_cell_types is not None:
        remove_cell_types = [cell_type.strip() for cell_type in remove_cell_types.split(',')]
        sample_data = sample_data[~sample_data[target_col].isin(remove_cell_types)]
    
    df = pd.ExcelFile(decision_matrix_path)
    logic = pd.read_excel(df, df.sheet_names, index_col=0)
    if normalization == 'z_score':
        normalization = z_score
    else:
        normalization = None
    fold_times = []
    for i in range(n_runs):
        start_time_loop = time.perf_counter()
        if seed_stability_mode:
            current_seed = seed + i
        else:
            current_seed = seed
        labels,_ = tribus.run_tribus(sample_data[cols], logic, depth=depth, normalization=normalization, tuning=tuning, sigma=sigma, learning_rate=learning_rate,
                                    clustering_threshold=clustering_threshold, undefined_threshold=undefined_threshold, other_threshold=other_threshold, random_state=current_seed)
        end_time_loop = time.perf_counter()
        elapsed_time_loop = end_time_loop - start_time_loop
        fold_times.append(elapsed_time_loop)
        result_data = sample_data.join(labels)
        if remove_result_cell_types is not None:
            remove_result_cell_types = [cell_type.strip() for cell_type in remove_result_cell_types.split(',')]
            result_data = result_data[~result_data[target_col].isin(remove_result_cell_types)]
        result_data.loc[result_data["final_label"].str.contains('other|undefined', case=False, na=False), "final_label"] = 'undefined'
        result_data.rename(columns={"final_label": 'predicted_phenotype', target_col: 'true_phenotype'}, inplace=True)
        result_data.to_csv(os.path.join(output_path, f'predictions_{i}.csv'))
        cr = classification_report(result_data['true_phenotype'], result_data['predicted_phenotype'])
        report_with_time = f"{cr}\nLoop iteration {i} processing time: {elapsed_time_loop:.4f} seconds\n"
        with open(os.path.join(output_path, f'classification_report_{i}.csv'), 'w') as f:
                f.write(report_with_time)
    with open(os.path.join(output_path, "fold_times.txt"), "w") as f:
        for i, elapsed in enumerate(fold_times):
            f.write(f"Fold {i+1} inference_time: {elapsed:.2f}\n")

def main():
    parser = argparse.ArgumentParser(description="Run Tribus on datasets")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility. If n_runs > 1, the seed will be incremented by the run number")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--seed_stability_mode", action='store_true', help="If set, the seed will be incremented by the run number for each run")
    parser.add_argument("--granularity_level", type=str, default='level3', help="Granularity level for cell types")
    parser.add_argument("--columns_to_use", type=str, default='gene1,gene2,gene3', help="Columns to use for tribus celltyping. Comma separated")
    parser.add_argument("--remove_cell_types", type=str, default=None, help="Cell types to remove from the dataset before running tribus, Comma separated")
    parser.add_argument("--decision_matrix_path", type=str, required=True, help="Path to the decision matrix")
    parser.add_argument("--scaling", type=int, default=None, help="Scaling factor for the data")
    parser.add_argument("--log1p", action='store_true', help="Apply log1p transformation to the data")
    parser.add_argument("--normalization", type=str, default=None, choices=['z_score'], help="Normalization method")
    parser.add_argument("--tuning", type=str, default=None, help='If hyperparameter tuning is needed, set to 1 default is 0')
    parser.add_argument("--sigma", type=float, default=1.0, help="Sigma. If tuning is performed, this value will be estimated automatically")
    parser.add_argument("--learning_rate", type=float, default=1.0, help="Learning rate. If tuning is performed, this value will be estimated automatically")
    parser.add_argument("--clustering_threshold", type=float, default=100.0, help="Clustering threshold")
    parser.add_argument("--undefined_threshold", type=float, default=0.0005, help="Undefined threshold")
    parser.add_argument("--other_threshold", type=float, default=0.4, help="Other threshold")
    parser.add_argument("--depth", type=int, default=3, help="Depth of the decision matrix")
    parser.add_argument("--remove_result_cell_types", type=str, default=None, help="Cell types to remove from the result, Comma separated")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the results")
    args = parser.parse_args()

    run_tribus(
        dataset_path=args.dataset_path,
        seed=args.seed,
        n_runs=args.n_runs,
        seed_stability_mode=args.seed_stability_mode,
        granularity_level=args.granularity_level,
        columns_to_use=args.columns_to_use,
        remove_cell_types=args.remove_cell_types,
        scaling=args.scaling,
        log1p=args.log1p,
        decision_matrix_path=args.decision_matrix_path,
        normalization=args.normalization,
        tuning=args.tuning,
        sigma=args.sigma,
        learning_rate=args.learning_rate,
        clustering_threshold=args.clustering_threshold,
        undefined_threshold=args.undefined_threshold,
        other_threshold=args.other_threshold,
        depth=args.depth,
        remove_result_cell_types=args.remove_result_cell_types,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()