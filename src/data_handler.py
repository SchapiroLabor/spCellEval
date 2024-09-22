import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder

class DataSetHandler:
    def __init__(self, path, random_state):
        self.file_path = path
        self.data = pd.read_csv(self.file_path)
        self.random_state = random_state
        self.X = None
        self.Y = None
        self.labels = None
        self.group_identifier = None
        self.kfolds = None
        self.fold_indices = None
        self.fold_data = None
        print("DataSetHandler initialized successfully")
    
    
    def preprocess(self, dropna: bool, impute_value: float = None, phenotype_column: str = 'cell_type', group_identifier_column: str = None) -> None:
        """
        This function processes the laoded dataframe. If NA values are present, they can be dropped or imputed with a specified value. A group identifier can be
        specified so the kfolds are created with the group identifier in mind. The phenotype column is encoded and the data is split into X and Y. 
        """

        if not isinstance(dropna, bool):
            raise TypeError("dropna must be a boolean value")
        if not isinstance(impute_value, (float, int, type(None))):
            raise TypeError("impute_value must be a number")
        
        if dropna:
            self.data.dropna(inplace=True)
        if impute_value is not None:
            self.data.fillna(impute_value, inplace=True)

        label_encoder = LabelEncoder()
        self.Y = label_encoder.fit_transform(self.data[phenotype_column])
        self.labels = pd.DataFrame({
            'label': range(len(label_encoder.classes_)),
            'phenotype': label_encoder.classes_
            })
        self.X = self.data.drop(columns=[phenotype_column])
        if group_identifier_column is not None:
            self.group_identifier = self.data[group_identifier_column]

        print("Data successfully preprocessed")
    
    def save_labels(self, save_path: str) -> None:
        if self.labels is None:
            raise ValueError("No labels have been created. Call preprocess first.")
        
        if save_path is None:
            save_path = os.getcwd()
        
        self.labels.to_csv(os.path.join(save_path, 'labels.csv'), index=False)
        print(f"Labels saved in: {save_path} as labels.csv")

    def createKfold(self, k: int) -> None:
        """
        Creates folds, depending if the user has specified a group identifier or not. If a group identifier is present, GroupKFold is used, otherwise KFold is used.
        Folds will be carried by the fold_data attribute.
        """
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        
        if self.group_identifier is not None:
            self.kfolds = GroupKFold(n_splits=k)
            fold_generator = self.kfolds.split(self.X, self.Y, self.group_identifier)
        else:
            self.kfolds = KFold(n_splits=k, shuffle=True, random_state=self.random_state)
            fold_generator = self.kfolds.split(self.X)

        self.fold_indices = list(fold_generator)
        
        self.fold_data = []
        for train_index, test_index in self.fold_indices:
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            Y_train, Y_test = self.Y[train_index], self.Y[test_index]
            
            fold = {
                'X_train': X_train,
                'X_test': X_test,
                'Y_train': Y_train,
                'Y_test': Y_test
            }
            self.fold_data.append(fold)
        
        print(f"{k} folds created. To save the folds, call save_folds method.")


    def save_folds(self, save_path: str) -> None:

        if self.fold_indices is None:
            raise ValueError("No folds have been created. Call createKfold first.")
        
        if save_path is None:
            save_path = os.getcwd()

        kfolds_dir = os.path.join(save_path, 'kfolds')
        os.makedirs(kfolds_dir, exist_ok=True)

        fold_data = {
            'random_state': self.random_state,
            'folds': [{'train': train.tolist(), 'test': test.tolist()} for train, test in self.fold_indices]
        }

        with open(os.path.join(kfolds_dir, 'fold_indices.json'), 'w') as f:
            json.dump(fold_data, f)

        for i, (train_index, test_index) in enumerate(self.fold_indices):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]
            
            train_data.to_csv(os.path.join(kfolds_dir, f'fold_{i+1}_train.csv'), index=False)
            test_data.to_csv(os.path.join(kfolds_dir, f'fold_{i+1}_test.csv'), index=False)

        print(f"Folds saved in: {kfolds_dir}")

            
    @classmethod
    def load_folds(cls, load_path: str = None, data_path: str = None) -> 'DataSetHandler':
        if load_path is None:
            load_path = os.getcwd()
        
        kfolds_dir = os.path.join(load_path, 'kfolds')
        
        with open(os.path.join(kfolds_dir, 'fold_indices.json'), 'r') as f:
            fold_data = json.load(f)
        
        if data_path is None:
            first_fold_file = [f for f in os.listdir(kfolds_dir) if f.startswith('fold_1_train_')][0]
            data_path = os.path.join(kfolds_dir, first_fold_file.replace('fold_1_train_', ''))
        
        handler = cls(data_path, fold_data['random_state'])
        handler.data = pd.read_csv(data_path)
        handler.fold_indices = [(np.array(fold['train']), np.array(fold['test'])) for fold in fold_data['folds']]
        handler.kfolds = KFold(n_splits=len(handler.fold_indices), shuffle=True, random_state=handler.random_state)
        
        return handler


