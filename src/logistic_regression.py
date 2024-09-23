import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self, random_state: int, data_handler) -> None:
        self.random_state = random_state
        self.data_handler = data_handler
        self.model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=random_state)
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

    def train_and_evaluate(self) -> None:
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
            confusion_matrix = confusion_matrix(Y_test, predictions)
            precision = precision_score(Y_test, predictions, average='weighted')
            recall = recall_score(Y_test, predictions, average='weighted')
            classification_report = classification_report(Y_test, predictions)

            self.fold_accuracies.append(accuracy)
            self.fold_f1_scores.append(f1)
            self.fold_weighted_f1_scores.append(weighted_f1)
            self.confusion_matrices.append(confusion_matrix)
            self.classification_reports.append(classification_report)

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
            confusion_matrix = confusion_matrix(Y_test, predictions)
            precision = precision_score(Y_test, predictions, average='weighted')
            recall = recall_score(Y_test, predictions, average='weighted')
            classification_report = classification_report(Y_test, predictions)

            self.fold_accuracies.append(accuracy)
            self.fold_f1_scores.append(f1)
            self.fold_weighted_f1_scores.append(weighted_f1)
            self.confusion_matrices.append(confusion_matrix)
            self.classification_reports.append(classification_report)

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