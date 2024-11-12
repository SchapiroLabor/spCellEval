import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as rfc
from joblib import Parallel, delayed
from tqdm import tqdm
import json
import csv
import pickle
import itertools

class Tune_Eval:
    def __init__(self, random_state: int, model: str, class_weight: dict = None) -> None:
        if model == 'logistic_regression':
            self.model_name = 'logistic_regression'
            self.model = LogisticRegression(random_state=random_state, penalty='l2')
        elif model == 'random_forest':
            self.model_name = 'random_forest'
            self.model = rfc(random_state=random_state, criterion='log_loss', class_weight=class_weight, n_jobs= 2)

        else:
            raise ValueError("Invalid model. Please choose either 'logistic_regression' or 'random_forest' as model parameter.")
         
        self.random_state = random_state
        self.data_handler = None
        self.scaler = StandardScaler()
        self.fold_accuracies = []
        self.fold_f1_scores = []
        self.fold_weighted_f1_scores = []
        self.fold_precisions = []
        self.fold_recalls = []
        self.confusion_matrices = []
        self.classification_reports = []
        self.average_accuracy = None
        self.average_f1_score = None
        self.average_weighted_f1_score = None
        self.average_precision = None
        self.average_recall = None
        self.best_models = []
        self.best_params = []

        print('GirdSearcher class initialized successfully with the following model parameters:')
        print(f'Random State: {self.random_state}')
        print(f'Model: {self.model}')

    @staticmethod
    def evaluate_params(args):
        model, X_train, y_train, X_val, y_val, params = args
        model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy, model, params
    
    def train_tune_evaluate(self, path: str, param_grid: dict, dumb_nonnumericals: bool = True, n_processes: int = 1, drop_xy: bool = True) -> None:
        """
        This function implements manual kfold cross-validation using a custom parameter grid and evaluates the models based on predefined and saved kfolds. 
        Input is the path pointing to the folder containing the train, validation and test csv kfold files.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} does not exist.")
        
        fold_dict = {
            'train': [],
            'validation': [],
            'test': []
        }
        for file in os.listdir(path):
            if file.endswith('.csv'):
                if 'train' in file:
                    fold_dict['train'].append(file)
                elif 'validation' in file:
                    fold_dict['validation'].append(file) 
                elif 'test' in file:
                    fold_dict['test'].append(file)
                else:
                    print(f"skipping {file}")

        fold_dict['train'].sort()
        fold_dict['validation'].sort()
        fold_dict['test'].sort()

        if len(fold_dict['train']) != len(fold_dict['test']) or len(fold_dict['train']) != len(fold_dict['validation']):
            raise ValueError("The number of train, validation and test files do not match.")
        

        for i, (train_file, validation_file, test_file) in enumerate(zip(fold_dict['train'], fold_dict['validation'], fold_dict['test'])):
            
            if dumb_nonnumericals:
                train_data = pd.read_csv(os.path.join(path, train_file)).select_dtypes(include=[np.number]) 
                validation_data = pd.read_csv(os.path.join(path, validation_file)).select_dtypes(include=[np.number])
                test_data = pd.read_csv(os.path.join(path, test_file)).select_dtypes(include=[np.number])
            else:
                train_data = pd.read_csv(os.path.join(path, train_file))
                validation_data = pd.read_csv(os.path.join(path, validation_file))
                test_data = pd.read_csv(os.path.join(path, test_file))

            non_numeric_df = train_data.select_dtypes(exclude=[np.number])
            if train_data.isnull().values.any() or test_data.isnull().values.any() or validation_data.isnull().values.any():
                raise ValueError("NaN values found in the fold data. Please handle missing values before training.")
            elif not non_numeric_df.empty:
                raise ValueError(f"Non-numeric values found in the fold data. Please encode the non-numeric values before training. Found columns: {non_numeric_df.columns}")
            else:
                print(f"Taking data from {train_file} and {test_file} for fold {i+1}. No NANs found")
            
            if drop_xy:
                train_data.drop(columns=['x', 'y'], inplace=True)
                validation_data.drop(columns=['x', 'y'], inplace=True)
                test_data.drop(columns=['x', 'y'], inplace=True)

            X_train = train_data.drop(columns='encoded_phenotype')
            y_train = train_data['encoded_phenotype']
            X_val = validation_data.drop(columns='encoded_phenotype')
            y_val = validation_data['encoded_phenotype']
            X_test = test_data.drop(columns='encoded_phenotype')
            y_test = test_data['encoded_phenotype']
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)

            # Perform manual Hyperparameter tuning 
            best_score = 0
            best_model = None
            best_params = None
            param_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
            args = [(self.model, X_train_scaled, y_train, X_val_scaled, y_val, param) for param in param_combinations]
            results = Parallel(n_jobs=n_processes)(
                delayed(self.evaluate_params)(arg) for arg in tqdm(args, desc=f"Fold {i+1}")
                )
            
            for accuracy, model, params in results:
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
                    best_params = params

            y_pred_test = best_model.predict(X_test_scaled)              
            accuracy = accuracy_score(y_test, y_pred_test)
            f1 = f1_score(y_test, y_pred_test, average='macro')
            weighted_f1 = f1_score(y_test, y_pred_test, average='weighted')
            precision = precision_score(y_test, y_pred_test, average='macro')
            recall = recall_score(y_test, y_pred_test, average='macro')
            cm = confusion_matrix(y_test, y_pred_test)
            cr = classification_report(y_test, y_pred_test, output_dict=False)

            self.fold_accuracies.append(accuracy)
            self.fold_f1_scores.append(f1)
            self.fold_weighted_f1_scores.append(weighted_f1)
            self.fold_precisions.append(precision)
            self.fold_recalls.append(recall)
            self.confusion_matrices.append(cm)
            self.classification_reports.append(cr)
            self.best_models.append(best_model)
            self.best_params.append(best_params)
            print(f"Outer Fold {i+1} completed")
            print(f"Best parameters for Fold {i+1}: {best_params}")
            print(f"classification report: {cr}")

        print('Calculating average results...')
        self.average_accuracy = np.mean(self.fold_accuracies)
        self.average_f1_score = np.mean(self.fold_f1_scores)
        self.average_weighted_f1_score = np.mean(self.fold_weighted_f1_scores)
        self.average_precision = np.mean(self.fold_precisions)
        self.average_recall = np.mean(self.fold_recalls)
        print(f'Average Accuracy: {self.average_accuracy}')
        print(f'Average F1 Score: {self.average_f1_score}')
        print(f'Average Weighted F1 Score: {self.average_weighted_f1_score}')



    def save_results(self, save_path: str, label_path: str, save_model: bool = True) -> None:
            """
            This function saves the results of the model and translates the labels to their respective phenotypes.
            """
            if label_path is None:
                raise ValueError("Please provide the path to the label file.")
            
            if not os.path.exists(save_path):
                print(f"The path {save_path} does not exist. Creating directory 'results' in the current working directory.")
                save_path = os.path.join(os.getcwd(), 'results')
                os.makedirs(save_path, exist_ok=True)
            else:
                print(f"The path {save_path} exists. Saving results in the specified directory.")

            # This part saves the average results in a .json file
            avg_results = {
                'average_accuracy': self.average_accuracy,
                'average_f1_score': self.average_f1_score,
                'average_weighted_f1_score': self.average_weighted_f1_score,
                'average_precision': self.average_precision,
                'average_recall': self.average_recall,
                'best_params': self.best_params
            }

            print("Saving average results...")
            if self.model_name == 'random_forest':
                json_name = 'average_rfc_results.json'
            elif self.model_name == 'logistic_regression':
                json_name = 'average_logreg_results.json'
            with open(os.path.join(save_path, json_name), 'w') as f:
                json.dump(avg_results, f, indent=4)


            labels = pd.read_csv(label_path)
            label_dict = dict(zip(labels['label'], labels['phenotype']))

            # This part saves the confusion matrices
            def __create_labeled_cm(cm, label_dict):
                df = pd.DataFrame(cm, columns=label_dict.values(), index=label_dict.values())
                df.index.name = 'Actual'
                df.columns.name = 'Predicted'
                return df

            print("Saving confusion matrices...")

            for fold, cm in enumerate(self.confusion_matrices, 1):
                labeled_cm = __create_labeled_cm(cm, label_dict)
                csv_filename = f'confusion_matrix_fold_{fold}.csv'
                labeled_cm.to_csv(os.path.join(save_path, csv_filename))
            
            # This part saves the classification reports
            def __translate_and_save_report(report_str, label_dict, fold):
                lines = report_str.strip().split('\n')

                header = lines[0].split()
                header.insert(0, 'label')

                data_rows = []
                for line in lines[2:]:  # Skip the header and the empty line
                    if line.strip() and not line.startswith('accuracy') and not line.startswith('macro avg') and not line.startswith('weighted avg'):
                        parts = line.split()
                        if len(parts) == 5:  # Ensure it's a data row
                            label_num = int(parts[0])
                            label_name = label_dict.get(label_num, str(label_num))
                            data_rows.append([label_name] + parts[1:])

                summary_rows = []
                for line in lines[-3:]:
                    parts = line.split()
                    if parts[0] == 'accuracy':
                        summary_rows.append(['accuracy', '', '', parts[1], parts[2]])  # Add an empty field for alignment
                    elif parts[0] == 'macro' or parts[0] == 'weighted':
                        summary_rows.append([f"{parts[0]}avg", parts[2], parts[3], parts[4], parts[5]])

                all_rows = data_rows + summary_rows
                if self.mode
                output_file = os.path.join(save_path, f'classification_report_fold_{fold}.csv')
                with open(output_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header)
                    writer.writerows(all_rows)

            print("Saving classification reports...")
            for i, cr in enumerate(self.classification_reports, 1):
                __translate_and_save_report(cr, label_dict, i)

            # This part saves the model
            if save_model:
                models_path = os.path.join(save_path, 'models')
                os.makedirs(models_path, exist_ok=True)
                print(f"Saving models in {models_path}...")
                for i, model in enumerate(self.best_models):
                    model_filename = os.path.join(models_path, f'{self.model_name}_model_fold_{i+1}.pkl')
                    with open(model_filename, 'wb') as f:
                        pickle.dump(model, f)
                print(f"Best models for all folds saved successfully in {models_path}.")
            else:
                print(f"Results saved successfully in {save_path}. Models not saved.")
            