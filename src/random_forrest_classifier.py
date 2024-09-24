import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import json
import csv
import pickle
from sklearn.ensemble import RandomForestClassifier as rfc

class MultiClassRandomForestClassifier:
    def __init__(self,
                 n_estimators: int = 100,
                 criterion: str = 'log_loss',
                 max_depth: int = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_weight_fraction_leaf: float = 0.0,
                 max_features: str = 'sqrt',
                 max_leaf_nodes: int = None,
                 min_impurity_decrease: float = 0.0,
                 bootstrap: bool = True,
                 oob_score: bool = False,
                 n_jobs: int = 2,
                 random_state: int = None,
                 verbose: int = 0,
                 warm_start: bool = False,
                 class_weight: dict = None,
                 ccp_alpha: float = 0.0,
                 max_samples: float = None) -> None:
        
        if criterion not in ['gini', 'entropy', 'log_loss']:
            raise ValueError("Criterion must be 'gini', 'log_loss or 'entropy'.")
        
        self.random_state = random_state
        self.data_handler = None
        self.model = rfc(n_estimators=n_estimators,
                         criterion=criterion,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease,
                         bootstrap=bootstrap,
                         oob_score=oob_score,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         verbose=verbose,
                         warm_start=warm_start,
                         class_weight=class_weight,
                         ccp_alpha=ccp_alpha,
                         max_samples=max_samples)
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

        print('MultiClassRandomForestClassifier class initialized successfully with the following parameters:')
        print(f'  n_estimators: {n_estimators}')
        print(f'  criterion: {criterion}')
        print(f'  max_depth: {max_depth}')
        print(f'  min_samples_split: {min_samples_split}')
        print(f'  min_samples_leaf: {min_samples_leaf}')
        print(f'  min_weight_fraction_leaf: {min_weight_fraction_leaf}')
        print(f'  max_features: {max_features}')
        print(f'  max_leaf_nodes: {max_leaf_nodes}')
        print(f'  min_impurity_decrease: {min_impurity_decrease}')
        print(f'  bootstrap: {bootstrap}')
        print(f'  oob_score: {oob_score}')
        print(f'  n_jobs: {n_jobs}')
        print(f'  random_state: {random_state}')
        print(f'  verbose: {verbose}')
        print(f'  warm_start: {warm_start}')
        print(f'  class_weight: {class_weight}')
        print(f'  ccp_alpha: {ccp_alpha}')
        print(f'  max_samples: {max_samples}')

    def train_and_evaluate(self, datahandler) -> None:
        """
        This function trains and evaluates the rfc model. Input is the .folds_data attribute from the DataSetHandler class. 
        """
        self.data_handler = datahandler
        if self.data_handler.fold_data is None:
            raise ValueError("No fold data found. Call create_folds on the DataSetHandler first.")
        
        for i, fold in enumerate(self.data_handler.fold_data):
            if fold['X_train'].isnull().values.any() or fold['X_test'].isnull().values.any() or fold['Y_train'].isnull().values.any() or fold['Y_test'].isnull().values.any():
                raise ValueError("NaN values found in the fold data. Please handle missing values before training.")
            else:
                print(f"Data in fold {i+1} is clean.")

        for i, fold in enumerate(self.data_handler.fold_data):
            X_train = self.scaler.fit_transform(fold['X_train'])
            X_test = self.scaler.transform(fold['X_test'])
            Y_train = fold['Y_train']
            Y_test = fold['Y_test']
            
            print(f"Training fold {i+1}...")
            self.model.fit(X_train, Y_train)
            predictions = self.model.predict(X_test)

            accuracy = accuracy_score(Y_test, predictions)
            f1 = f1_score(Y_test, predictions, average='macro')
            weighted_f1 = f1_score(Y_test, predictions, average='weighted')
            cm = confusion_matrix(Y_test, predictions)
            precision = precision_score(Y_test, predictions, average='weighted')
            recall = recall_score(Y_test, predictions, average='weighted')
            cr = classification_report(Y_test, predictions)

            self.fold_accuracies.append(accuracy)
            self.fold_f1_scores.append(f1)
            self.fold_weighted_f1_scores.append(weighted_f1)
            self.confusion_matrices.append(cm)
            self.classification_reports.append(cr)
            self.fold_precisions.append(precision)
            self.fold_recalls.append(recall)

            print(f"Fold {i+1} results:")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print("Accuracy:", accuracy)
            print("F1 Score:", f1)
            print("Weighted F1 Score:", weighted_f1)
            print("="*50)
        
        self.average_accuracy = np.mean(self.fold_accuracies)
        self.average_f1_score = np.mean(self.fold_f1_scores)
        self.average_weighted_f1_score = np.mean(self.fold_weighted_f1_scores)
        self.average_precision = np.mean(self.fold_precisions)
        self.average_recall = np.mean(self.fold_recalls)

        print(f"Average Accuracy across all folds: {self.average_accuracy}")
        print(f"Average F1 Score across all folds: {self.average_f1_score}")
        print(f"Average Weighted F1 Score across all folds: {self.average_weighted_f1_score}")
        print(f"Average Precision across all folds: {self.average_precision}")
        print(f"Average Recall across all folds: {self.average_recall}")

    def train_and_evaluate_manual(self, path: str) -> None:
        """
        This function trains and evaluates the rfc model. Input is the path pointing to the folder containing the train and test csv kfold files.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} does not exist.")
        
        fold_dict = {
            'train': [],
            'test': []
        }
        for file in os.listdir(path):
            if file.endswith('.csv'):
                if 'train' in file:
                    fold_dict['train'].append(file)
                elif 'test' in file:
                    fold_dict['test'].append(file)
                else:
                    print(f"skipping {file}")

        fold_dict['train'].sort()
        fold_dict['test'].sort()

        if len(fold_dict['train']) != len(fold_dict['test']):
            raise ValueError("The number of train and test files do not match.")
        
        for train_file, test_file in zip(fold_dict['train'], fold_dict['test']):
            train_data = pd.read_csv(os.path.join(path, train_file))
            test_data = pd.read_csv(os.path.join(path, test_file))
            if train_data.isnull().values.any() or test_data.isnull().values.any():
                raise ValueError("NaN values found in the fold data. Please handle missing values before training.")
            else:
                print(f"Data in {train_file} and {test_file} is clean.")

        for i, (train_file, test_file) in enumerate(zip(fold_dict['train'], fold_dict['test'])):
            train_data = pd.read_csv(os.path.join(path, train_file))
            test_data = pd.read_csv(os.path.join(path, test_file))

            X_train = train_data.drop(columns='encoded_phenotype')
            Y_train = train_data['encoded_phenotype']
            X_test = test_data.drop(columns='encoded_phenotype')
            Y_test = test_data['encoded_phenotype']


            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            print(f"Training fold {i+1}...")

            self.model.fit(X_train_scaled, Y_train)
            predictions = self.model.predict(X_test_scaled)

            accuracy = accuracy_score(Y_test, predictions)
            f1 = f1_score(Y_test, predictions, average='macro')
            weighted_f1 = f1_score(Y_test, predictions, average='weighted')
            cm = confusion_matrix(Y_test, predictions)
            precision = precision_score(Y_test, predictions, average='weighted')
            recall = recall_score(Y_test, predictions, average='weighted')
            cr = classification_report(Y_test, predictions)

            self.fold_accuracies.append(accuracy)
            self.fold_f1_scores.append(f1)
            self.fold_weighted_f1_scores.append(weighted_f1)
            self.confusion_matrices.append(cm)
            self.classification_reports.append(cr)
            self.fold_precisions.append(precision)
            self.fold_recalls.append(recall)

            print(f"Fold {i+1} results:")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print("Accuracy:", accuracy)
            print("F1 Score:", f1)
            print("Weighted F1 Score:", weighted_f1)
            print("="*50)
        
        self.average_accuracy = np.mean(self.fold_accuracies)
        self.average_f1_score = np.mean(self.fold_f1_scores)
        self.average_weighted_f1_score = np.mean(self.fold_weighted_f1_scores)
        self.average_precision = np.mean(self.fold_precisions)
        self.average_recall = np.mean(self.fold_recalls)

        print(f"Average Accuracy across all folds: {self.average_accuracy}")
        print(f"Average F1 Score across all folds: {self.average_f1_score}")
        print(f"Average Weighted F1 Score across all folds: {self.average_weighted_f1_score}")
        print(f"Average Precision across all folds: {self.average_precision}")
        print(f"Average Recall across all folds: {self.average_recall}")

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
        }

        print("Saving average results...")
        with open(os.path.join(save_path, 'average_rfc_results.json'), 'w') as f:
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
            print("Saving model...")
            model_filename = os.path.join(save_path, 'rfc_model.pkl')
            with open(model_filename, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Results and model saved successfully in {save_path}.")
        else:
            print("Results saved successfully in {save_path}. Model not saved.")
        
        