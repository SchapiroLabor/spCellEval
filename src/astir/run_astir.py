import os
import sys
from astir.data import from_csv_yaml
import argparse
import pandas as pd
import torch
import time

def run_astir(
    quant_path,
    decision_matrix_path,
    max_epochs,
    n_init,
    n_init_epochs,
    delta_loss,
    learning_rate,
    batch_size,
    random_seed,
    n_runs,
    output_path
):
    df = pd.read_csv(quant_path)
    df.to_csv(os.path.join(output_path, "quantification_astir.csv"), index=True)

    inference_times = []
    for n in range(n_runs):
        seed = random_seed + n
        ast = from_csv_yaml(os.path.join(output_path, "quantification_astir.csv"), marker_yaml=decision_matrix_path, random_seed=seed)

        start_time = time.time()
        ast.fit_type(max_epochs=max_epochs, n_init=n_init, n_init_epochs=n_init_epochs, delta_loss=delta_loss, learning_rate=learning_rate, batch_size=batch_size)
        end_time = time.time()
        elapsed_time = end_time - start_time

        inference_times.append(elapsed_time)
        cts = ast.get_celltypes()

        df_cts = df.copy()
        df_cts['predicted_phenotype'] = cts

        df_cts.to_csv(os.path.join(output_path, f"predictions_{n}.csv"))
        with open(os.path.join(output_path, "fold_times.txt"), "w") as f:
            for i, elapsed in enumerate(inference_times):
                f.write(f"Fold {i+1} inference_time: {elapsed:.2f}\n")

def main():
    parser = argparse.ArgumentParser(description="Run ASTIR with specified parameters.")
    parser.add_argument('--quant_path', type=str, required=True, help='Path to the quantification tableCSV file.')
    parser.add_argument('--decision_matrix_path', type=str, required=True, help='Path to the decision matrix YAML file.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run ASTIR on (default: cuda).')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs for training (default: 50).')
    parser.add_argument('--n_init', type=int, default=5, help='Number of initializations for training (default: 5).')
    parser.add_argument('--n_init_epochs', type=int, default=5, help='Number of epochs for each initialization (default: 5).')
    parser.add_argument('--delta_loss', type=float, default=0.001, help='Delta loss for convergence (default: 0.001).')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training (default: 0.001).')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128).')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility (default: 42).')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of runs for the ASTIR algorithm (default: 5).')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the ASTIR results.')
    args = parser.parse_args()

    torch.device(args.device)  
    run_astir(
        quant_path=args.quant_path,
        decision_matrix_path=args.decision_matrix_path,
        max_epochs=args.max_epochs,
        n_init=args.n_init,
        n_init_epochs=args.n_init_epochs,
        delta_loss=args.delta_loss,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
        n_runs=args.n_runs,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()