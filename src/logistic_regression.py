import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

class MultinomialLogisticRegression:
    def __init__(self, random_state: int, max_iter: int = 1000) -> None:
        self.random_state = random_state
        self.data_handler = None
        self.model = LogisticRegression(multi_class='multinomial', max_iter=max_iter, random_state=random_state)
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

    def train_and_evaluate(self, datahandler) -> None:
        """
        This function trains and evaluates the logistic regression model. Input is the .folds_data attribute from the DataSetHandler class. 
        """
        self.data_handler = datahandler
        if self.data_handler.fold_data is None:
            raise ValueError("No fold data found. Call create_folds on the DataSetHandler first.")
        
        for i, fold in enumerate(self.data_handler.fold_data):
            X_train = self.scaler.fit_transform(fold['X_train'])
            X_test = self.scaler.transform(fold['X_test'])
            Y_train = fold['Y_train']
            Y_test = fold['Y_test']
            
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

            print(f"Fold {i+1} results:")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print("Accuracy:", accuracy)
            print("F1 Score:", f1)
            print("Weighted F1 Score:", weighted_f1)
            print("="*30)
        
        self.average_accuracy = np.mean(self.fold_accuracies)
        self.average_f1_score = np.mean(self.fold_f1_scores)
        self.average_weighted_f1_score = np.mean(self.fold_weighted_f1_scores)
        self.average_precision = np.mean(self.fold_precisions)
        self.average_recall = np.mean(self.fold_recalls)

        print(f"Average Accuracy across all folds: {self.accuracy}")
        print(f"Average F1 Score across all folds: {self.average_f1_score}")
        print(f"Average Weighted F1 Score across all folds: {self.average_weighted_f1_score}")
        print(f"Average Precision across all folds: {self.average_precision}")
        print(f"Average Recall across all folds: {self.average_recall}")


            
            
# NEEEEDS TO BE ADJUSTED, currently its just a copy of the above function
    def train_and_evaluate_manual(self, path: str) -> None:
        """
        This function trains and evaluates the logistic regression model for multiclasses. Input is the path pointing to the folder containing the train and test csv kfold files.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} does not exist.")
        
        fold_dict = {
            'train': [],
            'test': []
        }
        for fold in os.listdir(path):
            if fold.endswith('.csv') and 'train' in fold:
                fold_dict['train'].append(fold)
            elif fold.endswith('.csv') and 'test' in fold:
                fold_dict['test'].append(fold)
        # Check that we always match the correct test and train fold, furhter we need to create X_train, X_test, Y_train, Y_test from the fold csv tab;es
        ##....
        for i, fold in enumerate(fold_data):
            X_train = self.scaler.fit_transform(fold['X_train'])
            X_test = self.scaler.transform(fold['X_test'])
            Y_train = fold['Y_train']
            Y_test = fold['Y_test']
            
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

            print(f"Fold {i+1} results:")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print("Accuracy:", accuracy)
            print("F1 Score:", f1)
            print("Weighted F1 Score:", weighted_f1)
            print("="*30)
        
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

        