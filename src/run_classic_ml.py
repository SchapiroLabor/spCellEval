import pandas as pd
import numpy as np
from models_classes.gridsearch import ClassicMLTuner
import argparse
import os
import json

def run_on_datasets(main_dir, model, param_grid, random_state, n_jobs_model, n_jobs_gridsearch, model_kwargs, verbose, scoring, scaling, dumb_columns,  sample_weight, xgb_earlystopping):
    """
    This function runs the selected model on all datasets in the main directory using the argparse arguments.
    """
    # Extracting the keywords from json files
    if model_kwargs: #optional input
        with open(model_kwargs, 'r') as f:
            model_kwargs_dict = json.load(f)
    else:
        model_kwargs_dict = {}

    with open(param_grid, 'r') as f: #required input
        param_grid_dict = json.load(f)
    
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
    
    if sample_weight == 'balanced':
        sample_weight = 'balanced'
    elif os.path.isfile(sample_weight):
        try:
            with open(sample_weight, 'r') as f:
                sample_weight = json.load(f)
        except Exception as e:
            raise ValueError(f"Could not load the sample weight file. Error: {e}")
    else:
        sample_weight = None
    
    if isinstance(scoring, str):
        scoring_clean = scoring.strip()
        if ',' in scoring_clean:
            scoring = [s.strip() for s in scoring_clean.split(',')]
        else:
            scoring = scoring_clean


    # Starting the loop through datasets
    for dataset_dir in os.listdir(os.path.join(main_dir, 'datasets')):
        if not os.path.isdir(os.path.join(main_dir, 'datasets', dataset_dir)):
            continue
        if dataset_dir.startswith('.'):
            continue
        print(f"Processing dataset {dataset_dir}")
        datasets_path = os.path.join(main_dir, "datasets", dataset_dir)
        kfold_dir = os.path.join(datasets_path, 'quantification/processed/kfolds')
        save_dir = os.path.join(main_dir, 'results', dataset_dir, model)
        os.makedirs(save_dir, exist_ok=True)
        label_dir = os.path.join(datasets_path, 'quantification/processed/labels.csv')

        data_object = ClassicMLTuner(random_state=random_state, model = model, n_jobs = n_jobs_model, **model_kwargs_dict)
        data_object.train_tune_evaluate(kfold_dir, param_grid_dict, n_jobs_gridsearch, verbose, scoring,
                                        scaling, dumb_columns, sample_weight = sample_weight, early_stopping_rounds=xgb_earlystopping)
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
        choices=['logistic_regression', 'random_forest', 'xgboost'],
        help="Select a model to run, either 'logistic_regression', 'random_forest' or 'xgboost'", required=True
        )
    parser.add_argument(
        "--param_grid",
        type=str,
        help="""Path to the json file containing the hyperparameter grid to search. Has to be in a py-dictionary format. Example:
        {
        "n_estimators": [100, 250, 500],
        "max_depth": [5, 7, 10],
        }
        """,
        required=True
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
        default=1,
        help="Number of jobs passed to the model object itself" 
        )
    parser.add_argument(
        "--n_jobs_gridsearch",
        type=int,
        default=-1,
        help="Number of jobs to run in parallel during GridSearchCV"
        )
    parser.add_argument(
        "--model_kwargs",
        type=str,
        help="""Path to the json file containing the model kwargs, otherwise standard values are used. Example:
        {
        "min_child_weight": 2,
        "learning_rate": 0.05,
        }
        """
        )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="Verbosity level for the GridSearchCV"
        )
    parser.add_argument(
        "--scoring",
        type=str,
        default='accuracy',
        help="Scoring metric for GridSearchCV. Can either be one string or several strings seperated by comma. "
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
    parser.add_argument(
        "--sample_weight",
        type=str,
        help=" Either 'balanced' or path to the json sample weight dictionary"
        )
    parser.add_argument(
        "--xgb_earlystopping",
        type=int,
        default=10,
        help="Early stopping rounds for the best parameter xgboost model on final refit"
        )

    args = parser.parse_args()
    
    print(f"Selected model: {args.model}")
    
    run_on_datasets(args.main_dir, args.model, args.param_grid, args.random_state,
                    args.n_jobs_model, args.n_jobs_gridsearch, args.model_kwargs, args.verbose,
                    args.scoring, args.scaling, args.dumb_columns, args.sample_weight, args.xgb_earlystopping)

if __name__ == '__main__':
    main()
