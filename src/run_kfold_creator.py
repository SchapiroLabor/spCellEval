import argparse
import os
from models_classes.data_handler import DataSetHandler

def run_kfold_creation(main_dir, dataset_name, dropna, impute_value, phenotype_column, drop_columns, drop_non_numerical, n_splits, random_state, percentage_validation):
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
        for dataset in os.path.join(main_dir, 'datasets'):
            if not os.path.isdir(dataset):
                continue
            print(f"Processing dataset {dataset}")
            dataset_path = os.path.join(main_dir, 'datasets', dataset, f'quantification/processed/{dataset}_quantification.csv')
            save_dir = os.path.join(main_dir, 'datasets', dataset, 'quantification/processed')
            data_handler = DataSetHandler(dataset_path, random_state=random_state)
            data_handler.preprocess(dropna, impute_value, phenotype_column, drop_columns = drop_columns, drop_non_numerical = drop_non_numerical)
            data_handler.save_labels(save_dir)
            data_handler.createKfold(n_splits)
            data_handler.save_folds(save_dir)
            data_handler.create_validation_set_from_fold(save_path=os.path.join(save_dir, 'kfolds'), percentage_validation=percentage_validation)
    else:
        if os.path.isdir(os.path.join(main_dir, 'datasets', dataset_name)):
            print(f"Processing {dataset_name}")
            dataset_path = os.path.join(main_dir, 'datasets', dataset_name, f'quantification/processed/{dataset_name}_quantification.csv')
            save_dir = os.path.join(main_dir, 'datasets', dataset_name, 'quantification/processed')
            data_handler = DataSetHandler(dataset_path, random_state=random_state)
            data_handler.preprocess(dropna, impute_value, phenotype_column, drop_columns = drop_columns, drop_non_numerical = drop_non_numerical)
            data_handler.save_labels(save_dir)
            data_handler.createKfold(n_splits)
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
        . An explicit directory structure is required. If mode=batch, needs to be the main directory holding directories 
         'datasets' and 'results. If mode=single, needs to point to the specific dataset. See README for more information""",
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
        help='name of the column containing the target variable. Default is cell_type',
        default='cell_type',
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
    run_kfold_creation(args.main_dir, args.dataset_name, args.dropna, args.impute_value, args.phenotype_column,
                       args.drop_columns, args.drop_non_numerical, args.n_splits, args.random_state,
                       args.percentage_validation
                       )
    print("Done.")
if __name__ == '__main__':
    main()