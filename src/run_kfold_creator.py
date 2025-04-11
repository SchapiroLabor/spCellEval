import argparse
import os
from models_classes.data_handler import DataSetHandler

def run_fold_creation(main_dir, dataset_name, dropna, impute_value, phenotype_column, batch_identifier_column, drop_columns, drop_non_numerical, n_splits, method, group_shuffle_split_size, random_state, percentage_validation):
    """
    This function executes the DataSetHandler class to create cleaned kfolds and labels including translation csvs.
    """    
    if impute_value is not None:
        impute_value = float(impute_value)
    
    if drop_columns is not None:
        drop_columns_clean = drop_columns.strip()
        if ',' in drop_columns_clean:
            drop_columns = [s.strip() for s in drop_columns_clean.split(',')]
        else:
            drop_columns = drop_columns_clean

    # Loop through datasets
    if dataset_name is None:
        for dataset in os.listdir(os.path.join(main_dir, 'datasets')):
            if dataset.startswith('.'):
                continue
            if not os.path.isdir(os.path.join(main_dir,'datasets',dataset)):
                continue

            print(f"Processing dataset {dataset}")
            dataset_path = os.path.join(main_dir, 'datasets', dataset, f'quantification/processed/{dataset}_quantification.csv')
            save_dir = os.path.join(main_dir, 'datasets', dataset, 'quantification/processed')
            data_handler = DataSetHandler(dataset_path, random_state=random_state)
            data_handler.preprocess(dropna, impute_value, phenotype_column, batch_identifier_column, drop_columns = drop_columns, drop_non_numerical = drop_non_numerical)
            data_handler.createFolds(n_splits, method, batch_identifier_column, group_shuffle_split_size)
            data_handler.save_labels(save_dir)
            data_handler.save_folds(save_dir)
            data_handler.create_validation_set_from_fold(save_path=os.path.join(save_dir, f'kfolds_{method}'), percentage_validation=percentage_validation)
    else:
        if os.path.isdir(os.path.join(main_dir, 'datasets', dataset_name)):
            print(f"Processing {dataset_name}")
            dataset_path = os.path.join(main_dir, 'datasets', dataset_name, f'quantification/processed/{dataset_name}_quantification.csv')
            save_dir = os.path.join(main_dir, 'datasets', dataset_name, 'quantification/processed')
            data_handler = DataSetHandler(dataset_path, random_state=random_state)
            data_handler.preprocess(dropna, impute_value, phenotype_column, batch_identifier_column, drop_columns = drop_columns, drop_non_numerical = drop_non_numerical)
            data_handler.createFolds(n_splits, method, batch_identifier_column, group_shuffle_split_size)
            data_handler.save_labels(save_dir)
            data_handler.save_folds(save_dir)
            data_handler.create_validation_set_from_fold(save_path=os.path.join(save_dir, 'kfolds'), percentage_validation=percentage_validation)
        else:
            raise ValueError(f"{dataset_name} is not present among the datasets")
def main():
    
    parser = argparse.ArgumentParser(
        description = 'clean data, label target variable and create training, validation and test kfolds'
    )
    
    parser.add_argument(
        '--main_dir',
        type = str,
        help = """ Path to the main directory holding the folders 'datasets' and 'results'.
        . An explicit directory structure is required. See README for more information""",
        required = True,
    )
    parser.add_argument(
        '--dataset_name',
        type = str,
        default=None,
        help = "If method should be run on only 1 dataset, specify the name of the dataset. Default is None",
    )
    parser.add_argument(
        '--dropna',
        action='store_true',
        help='drop rows with missing values. Default is False',
    )
    parser.add_argument(
        '--impute_value',
        type=float,
        help='value to impute missing values. Default is None',
        default=None,
    )
    parser.add_argument(
        '--phenotype_column',
        type=str,
        help="""name of the column containing the target variable. Default is cell_type, which corresponds to level1 granularity.
        """,
        choices=['cell_type', 'level_2_cell_type', 'level_1_cell_type'],
        default='cell_type',
    )
    parser.add_argument(
        '--batch_identifier_column',
        type=str,
        help='Name of the column containing the batch identifier. Default is None',
        default=None,
    )
    parser.add_argument(
        '--drop_columns',
        type = str,
        help = 'columns to drop from the data. Comma-seperated strings. Example: column1,column2',
        default = None,
    )
    parser.add_argument(
        '--drop_non_numerical',
        action='store_true',
        help = 'drop non-numerical columns. Default is False',
    )
    parser.add_argument(
        '--n_splits',
        type = int,
        help = 'number of splits for the kfold. Default is 5',
        default = 5,
    )
    parser.add_argument(
        '--method',
        type = str,
        help = 'method to use for creating folds. Default is StratifiedKFold',
        choices=['StratifiedKFold', 'StratifiedGroupKFold', 'GroupShuffleSplit'],
        default = 'StratifiedKFold',
    )
    parser.add_argument(
        '--group_shuffle_split_size',
        type = float,
        help = 'size of the group shuffle split if GroupShuffleSplit was selected as method. Default is 0.5',
        default = 0.5,
    )
    parser.add_argument(
        '--random_state',
        type = int,
        help = 'random state for reproducibility. Default is 42',
        default = 42,
    )
    parser.add_argument(
        '--percentage_validation',
        type = float,
        help = 'percentage of data to be used as validation set. Default is 0.15',
        default = 0.15,
    )

    args = parser.parse_args()
    run_fold_creation(args.main_dir, args.dataset_name, args.dropna, args.impute_value, args.phenotype_column,
                        args.batch_identifier_column, args.drop_columns, args.drop_non_numerical, args.n_splits,
                        args.method, args.group_shuffle_split_size, args.random_state,
                        args.percentage_validation
                        )
    print("Done.")
if __name__ == '__main__':
    main()