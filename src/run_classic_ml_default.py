from models_classes.default_classic_ml_models_kfolds import ClassicMLDefault
import argparse
import os
import json

def run_on_datasets(main_dir, model, kfold_method, granularity_level, random_state, n_jobs_model, model_kwargs, verbose, scaling, dumb_columns):
    """
    This function runs the selected model on all datasets in the main directory using the argparse arguments.
    """
    # Extracting the keywords from json files
    if model_kwargs: #optional input
        with open(model_kwargs, 'r') as f:
            model_kwargs_dict = json.load(f)
    else:
        model_kwargs_dict = {}

    if scaling == 'Yes':
        scaling = True
    else:
        scaling = False
    
    if dumb_columns is not None:
        dumb_columns_clean = dumb_columns.strip()
        if ',' in dumb_columns_clean:
            dumb_columns = [s.strip() for s in dumb_columns_clean.split(',')]
        else:
            dumb_columns = dumb_columns_clean

    # Starting the loop through datasets
    for dataset_dir in os.listdir(os.path.join(main_dir, 'datasets')):
        if not os.path.isdir(os.path.join(main_dir, 'datasets', dataset_dir)):
            continue
        if dataset_dir.startswith('.'):
            continue
        print(f"Processing dataset {dataset_dir}")
        datasets_path = os.path.join(main_dir, "datasets", dataset_dir)
        kfold_dir = os.path.join(datasets_path, f'quantification/processed/kfolds_{kfold_method}_{granularity_level}')
        save_dir = os.path.join(main_dir, 'results', dataset_dir, f'{model}_default_{kfold_method}', granularity_level)
        os.makedirs(save_dir, exist_ok=True)
        label_dir = os.path.join(datasets_path, f'quantification/processed/labels_{kfold_method}_{granularity_level}.csv')

        data_object = ClassicMLDefault(random_state=random_state, model = model, n_jobs = n_jobs_model, **model_kwargs_dict)
        data_object.train_tune_evaluate(kfold_dir, label_dir, verbose, scaling, dumb_columns)
        data_object.save_results(save_dir, label_dir, kfold_dir)


def main():

    parser = argparse.ArgumentParser(
        description='Run classic ML models on the benchmark datasets'
        )
    parser.add_argument(
        "--main_dir",
        type=str,
        help="Path to the main directory holding the folders 'datasets' and 'results'. A specific directory structure is expected. See README for more information.",
        required=True
        )
    parser.add_argument(
        "--model",
        type=str,
        choices=['logistic_regression', 'random_forest', 'xgboost','most_frequent','stratified'],
        help="Select a model to run, either 'logistic_regression', 'random_forest', 'xgboost', 'most_frequent', or 'stratified'.", required=True
        )
    parser.add_argument(
        "--kfold_method",
        type=str,
        choices=['StratifiedGroupKFold', 'StratifiedKFold', 'GroupShuffleSplit'],
        default='StratifiedGroupKFold',
        help="Which fold creation method was used. Default is StratifiedGroupKFold."
        )
    parser.add_argument(
        "--granularity_level",
        type=str,
        choices=['level1', 'level2', 'level3'],
        default='level3',
        help="On which celltype granularity level the method is applied to. Granularity level 3 represents the finest granularity and equals cell subtypes. Default is level3."
        )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility"
        )
    parser.add_argument(
        "--n_jobs_model",
        type=int,
        default=-1,
        help="Number of jobs passed to the model object itself" 
        )
    parser.add_argument(
        "--model_kwargs",
        type=str,
        help="""Path to the json file containing the model kwargs, otherwise standard values are used. please consider using class_weight parameter as 'balanced, also for xgboost. Example:
        {
        "min_child_weight": 2,
        "learning_rate": 0.05,
        "class_weight": "balanced",
        }
        In XGBoost, the class_weight parameter is not directly available. We implemented it with compute_sample_weight parameter from sklearn. In Logistic Regression and Random Forest, the class_weight parameter is directly available.
        """
        )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="Verbosity level for xgboost training and evluation"
        )
    parser.add_argument(
        '--scaling',
        type=str,
        choices=['Yes', 'No'],
        default='Yes',
        help="Whether to scale the data using scikit-learn StandardScaler or not"
        )
    parser.add_argument(
        "--dumb_columns",
        type=str,
        help="Columns to drop from the data. Comma-seperated strings. Example: column1,column2",
        default=None
        )

    args = parser.parse_args()
    
    print(f"Selected model: {args.model}")
    
    run_on_datasets(
        main_dir=args.main_dir,
        model=args.model,
        kfold_method=args.kfold_method,
        granularity_level=args.granularity_level,
        random_state=args.random_state,
        n_jobs_model=args.n_jobs_model,
        model_kwargs=args.model_kwargs,
        verbose=args.verbose,
        scaling=args.scaling,
        dumb_columns=args.dumb_columns
    )

if __name__ == '__main__':
    main()
