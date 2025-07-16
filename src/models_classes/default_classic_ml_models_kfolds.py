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
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as rfc
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.dummy import DummyClassifier
import json
import csv
import pickle
import time


class ClassicMLDefault:
    def __init__(self, random_state: int, model: str, n_jobs: str, **kwargs) -> None:
        if model == "logistic_regression":
            self.model_name = "logistic_regression"
            self.model = LogisticRegression(
                n_jobs=n_jobs, random_state=random_state, **kwargs
            )
        elif model == "random_forest":
            self.model_name = "random_forest"
            self.model = rfc(
                n_jobs=n_jobs, random_state=random_state, criterion="log_loss", **kwargs
            )
        elif model == "xgboost":
            self.class_weight = kwargs.pop("class_weight", None)
            self.model_name = "xgboost"
            self.model = XGBClassifier(
                objective="multi:softmax",
                eval_metric="mlogloss",
                booster="gbtree",
                n_jobs=n_jobs,
                random_state=random_state,
                **kwargs,
            )
        elif model == "most_frequent":
            self.model_name = "most_frequent"
            self.model = DummyClassifier(
                strategy="most_frequent", random_state=random_state, **kwargs
            )
        elif model == "stratified":
            self.model_name = "stratified"
            self.model = DummyClassifier(
                strategy="stratified", random_state=random_state, **kwargs
            )
        else:
            raise ValueError(
                "Invalid model. Pleaseso choose either 'logistic_regression', 'xgboost', 'random_forest', 'most_frequent or 'stratified' as model parameter."
            )

        self.kwargs = kwargs
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
        self.predictions = {}
        self.train_times = []
        self.pred_times = []

        print("Class initialized successfully with the following model parameters:")
        print(f"Random State: {self.random_state}")
        print(f"Model: {self.model}")

    def train_tune_evaluate(
        self,
        path: str,
        label_path: str,
        verbose: int = 2,
        scaling: bool = True,
        dumb_columns: str | list = None,
        dumb_nonnumericals: bool = True,
    ) -> None:
        """
        This function implements manual kfold cross-validation using a custom parameter grid and evaluates the models based on predefined and saved kfolds.
        Input is the path pointing to the folder containing the train, validation and test csv kfold files.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} does not exist.")

        fold_dict = {"train": [], "validation": [], "test": []}
        for file in os.listdir(path):
            if file.endswith(".csv"):
                if "train" in file:
                    fold_dict["train"].append(file)
                elif "validation" in file:
                    fold_dict["validation"].append(file)
                elif "test" in file:
                    fold_dict["test"].append(file)
                else:
                    print(f"skipping {file}")

        fold_dict["train"].sort()
        fold_dict["validation"].sort()
        fold_dict["test"].sort()

        if len(fold_dict["train"]) != len(fold_dict["test"]) or len(
            fold_dict["train"]
        ) != len(fold_dict["validation"]):
            raise ValueError(
                "The number of train, validation and test files do not match."
            )

        labels = pd.read_csv(label_path)
        label_dict = dict(zip(labels["label"], labels["phenotype"]))
        all_labels = list(label_dict.keys())

        train_times = []
        pred_times = []
        print("Starting the integration of the predefined Kfolds...")
        for c, (train_file, validation_file, test_file) in enumerate(
            zip(fold_dict["train"], fold_dict["validation"], fold_dict["test"])
        ):
            if self.model_name == "logistic_regression":
                self.model = LogisticRegression(
                    n_jobs=self.model.n_jobs,
                    random_state=self.random_state,
                    **self.kwargs,
                )
            elif self.model_name == "random_forest":
                self.model = rfc(
                    n_jobs=self.model.n_jobs,
                    random_state=self.random_state,
                    criterion="log_loss",
                    **self.kwargs,
                )
            elif self.model_name == "xgboost":
                self.model = XGBClassifier(
                    objective="multi:softmax",
                    eval_metric="mlogloss",
                    booster="gbtree",
                    n_jobs=self.model.n_jobs,
                    random_state=self.random_state,
                    **self.kwargs,
                )
            elif self.model_name == "most_frequent":
                self.model = DummyClassifier(
                    strategy="most_frequent",
                    random_state=self.random_state,
                    **self.kwargs,
                )
            elif self.model_name == "stratified":
                self.model = DummyClassifier(
                    strategy="stratified", random_state=self.random_state, **self.kwargs
                )
            if dumb_columns is not None:
                train_data = pd.read_csv(os.path.join(path, train_file))
                validation_data = pd.read_csv(os.path.join(path, validation_file))
                test_data = pd.read_csv(os.path.join(path, test_file))
                # check if columns in dumb_columns are in the dataframes
                dumb_columns = [
                    col for col in dumb_columns if col in train_data.columns
                ]
                train_data = train_data.drop(columns=dumb_columns)
                validation_data = validation_data.drop(columns=dumb_columns)
                test_data = test_data.drop(columns=dumb_columns)
            else:
                train_data = pd.read_csv(os.path.join(path, train_file))
                validation_data = pd.read_csv(os.path.join(path, validation_file))
                test_data = pd.read_csv(os.path.join(path, test_file))
            if dumb_nonnumericals:
                train_data = train_data.select_dtypes(include=[np.number])
                validation_data = validation_data.select_dtypes(include=[np.number])
                test_data = test_data.select_dtypes(include=[np.number])

            non_numeric_df = train_data.select_dtypes(exclude=[np.number])
            if (
                train_data.isnull().values.any()
                or validation_data.isnull().values.any()
            ):
                raise ValueError(
                    "NaN values found in the fold data. Please handle missing values before training."
                )
            elif not non_numeric_df.empty:
                raise ValueError(
                    f"Non-numeric values found in the fold data. Please encode the non-numeric values before training. Found columns: {non_numeric_df.columns}"
                )
            else:
                print(
                    f"Taking data from {train_file} and {validation_file} for fold {c+1}. No NANs found"
                )

            X_train = train_data.drop(columns="encoded_phenotype")
            y_train = train_data["encoded_phenotype"].values
            X_val = validation_data.drop(columns="encoded_phenotype")
            y_val = validation_data["encoded_phenotype"].values
            X_test = test_data.drop(columns="encoded_phenotype")
            y_test = test_data["encoded_phenotype"].values
            if scaling:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
                X_test_scaled = self.scaler.transform(X_test)
                X_train = pd.DataFrame(
                    X_train_scaled,
                    columns=train_data.drop(columns="encoded_phenotype").columns,
                )
                X_val = pd.DataFrame(
                    X_val_scaled,
                    columns=validation_data.drop(columns="encoded_phenotype").columns,
                )
                X_test = pd.DataFrame(
                    X_test_scaled,
                    columns=test_data.drop(columns="encoded_phenotype").columns,
                )

            # We need to adjust to xgboost case where we dont have the same labels in training than testing:
            y_train_to_fit = y_train
            y_val_to_fit = y_val
            reverse_label_map = None

            if self.model_name == "xgboost":
                unique_train_labels = np.unique(y_train)

                # Check whether our unique training labels adhere to the XGBoost desired labeli indices, starting from 0 with consecutive integers
                if not np.array_equal(
                    unique_train_labels, np.arange(len(unique_train_labels))
                ):
                    print(
                        f"Fold {c+1}: Labels are not zero-indexed/consecutive. Re-mapping for XGBoost."
                    )
                    label_map = {
                        label: i for i, label in enumerate(unique_train_labels)
                    }
                    reverse_label_map = {label: i for i, label in label_map.items()}
                    y_train_to_fit = np.array([label_map[label] for label in y_train])
                    y_val_to_fit = np.array([label_map[label] for label in y_val])

            print(f"Fold {c+1} integrated successfully")

            train_start_time = time.time()
            if self.model_name == "xgboost":
                if self.class_weight is not None:
                    sample_weights = compute_sample_weight(
                        class_weight=self.class_weight, y=y_train_to_fit
                    )
                    self.model.fit(
                        X_train,
                        y_train_to_fit,
                        eval_set=[(X_val, y_val_to_fit)],
                        verbose=verbose,
                        sample_weight=sample_weights,
                    )
                else:
                    self.model.fit(
                        X_train,
                        y_train_to_fit,
                        eval_set=[(X_val, y_val_to_fit)],
                        verbose=verbose,
                    )
            else:
                self.model.fit(X_train, y_train)
            train_end_time = time.time()
            train_time = train_end_time - train_start_time

            pred_start_time = time.time()
            y_pred_test = self.model.predict(X_test)
            pred_end_time = time.time()
            pred_time = pred_end_time - pred_start_time
            self.train_times.append(train_time)
            self.pred_times.append(pred_time)

            if reverse_label_map:
                print(reverse_label_map)
                print("\n")
                print(label_map)
                y_pred_test = np.array(
                    [reverse_label_map[int(label)] for label in y_pred_test]
                )

            accuracy = accuracy_score(y_test, y_pred_test)
            f1 = f1_score(y_test, y_pred_test, average="macro")
            weighted_f1 = f1_score(y_test, y_pred_test, average="weighted")
            precision = precision_score(y_test, y_pred_test, average="macro")
            recall = recall_score(y_test, y_pred_test, average="macro")
            cm = confusion_matrix(y_test, y_pred_test, labels=all_labels)
            cr = classification_report(y_test, y_pred_test, output_dict=False)

            self.predictions[f"fold_{c+1}"] = y_pred_test
            self.fold_accuracies.append(accuracy)
            self.fold_f1_scores.append(f1)
            self.fold_weighted_f1_scores.append(weighted_f1)
            self.fold_precisions.append(precision)
            self.fold_recalls.append(recall)
            self.confusion_matrices.append(cm)
            self.classification_reports.append(cr)
            self.best_models.append(self.model)
            print(f"Test Fold {c+1} completed")
            print(f"classification report: {cr}")

            del X_train, X_val, y_train, y_val, X_test, y_test

        print("Calculating average results...")
        self.average_accuracy = np.mean(self.fold_accuracies)
        self.average_f1_score = np.mean(self.fold_f1_scores)
        self.average_weighted_f1_score = np.mean(self.fold_weighted_f1_scores)
        self.average_precision = np.mean(self.fold_precisions)
        self.average_recall = np.mean(self.fold_recalls)
        print(f"Average Accuracy: {self.average_accuracy}")
        print(f"Average F1 Score: {self.average_f1_score}")
        print(f"Average Weighted F1 Score: {self.average_weighted_f1_score}")

    def save_results(
        self, save_path: str, label_path: str, data_path: str, save_model: bool = True
    ) -> None:
        """
        This function saves the results of the model and translates the labels to their respective phenotypes.
        """
        if label_path is None:
            raise ValueError("Please provide the path to the label file.")

        if not os.path.exists(save_path):
            print(
                f"The path {save_path} does not exist. Creating directory 'results' in the current working directory."
            )
            save_path = os.path.join(os.getcwd(), "results")
            os.makedirs(save_path, exist_ok=True)
        else:
            print(
                f"The path {save_path} exists. Saving results in the specified directory."
            )

        # This part saves the average results in a .json file
        avg_results = {
            "average_accuracy": self.average_accuracy,
            "average_f1_score": self.average_f1_score,
            "average_weighted_f1_score": self.average_weighted_f1_score,
            "average_precision": self.average_precision,
            "average_recall": self.average_recall,
        }

        print("Saving average results...")
        if self.model_name == "random_forest":
            json_name = "average_rfc_results.json"
        elif self.model_name == "logistic_regression":
            json_name = "average_logreg_results.json"
        elif self.model_name == "xgboost":
            json_name = "average_xgboost_results.json"
        elif self.model_name == "most_frequent":
            json_name = "average_most_frequent_results.json"
        elif self.model_name == "stratified":
            json_name = "average_stratified_results.json"
        with open(os.path.join(save_path, json_name), "w") as f:
            json.dump(avg_results, f, indent=4)

        with open(os.path.join(save_path, "fold_times.txt"), "w") as f:
            for i, (elapsed_train, elapsed_pred) in enumerate(
                zip(self.train_times, self.pred_times)
            ):
                f.write(f"Fold {i+1} training_time: {elapsed_train:.2f}\n")
                f.write(f"Fold {i+1} prediction_time: {elapsed_pred:.2f}\n")

        labels = pd.read_csv(label_path)
        label_dict = dict(zip(labels["label"], labels["phenotype"]))

        # This part saves the confusion matrices
        def __create_labeled_cm(cm, label_dict):
            df = pd.DataFrame(
                cm, columns=label_dict.values(), index=label_dict.values()
            )
            df.index.name = "Actual"
            df.columns.name = "Predicted"
            return df

        print("Saving confusion matrices...")

        for fold, cm in enumerate(self.confusion_matrices, 1):
            labeled_cm = __create_labeled_cm(cm, label_dict)
            csv_filename = f"confusion_matrix_fold_{fold}.csv"
            labeled_cm.to_csv(os.path.join(save_path, csv_filename))

        # This part saves the classification reports
        def __translate_and_save_report(report_str, label_dict, fold):
            lines = report_str.strip().split("\n")

            header = lines[0].split()
            header.insert(0, "label")

            data_rows = []
            for line in lines[2:]:  # Skip the header and the empty line
                if (
                    line.strip()
                    and not line.startswith("accuracy")
                    and not line.startswith("macro avg")
                    and not line.startswith("weighted avg")
                ):
                    parts = line.split()
                    if len(parts) == 5:  # Ensure it's a data row
                        label_num = int(parts[0])
                        label_name = label_dict.get(label_num, str(label_num))
                        data_rows.append([label_name] + parts[1:])

            summary_rows = []
            for line in lines[-3:]:
                parts = line.split()
                if parts[0] == "accuracy":
                    summary_rows.append(
                        ["accuracy", "", "", parts[1], parts[2]]
                    )  # Add an empty field for alignment
                elif parts[0] == "macro" or parts[0] == "weighted":
                    summary_rows.append(
                        [f"{parts[0]}avg", parts[2], parts[3], parts[4], parts[5]]
                    )

            all_rows = data_rows + summary_rows

            output_file = os.path.join(
                save_path, f"classification_report_fold_{fold}.csv"
            )
            with open(output_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(all_rows)

        print("Saving classification reports...")
        for i, cr in enumerate(self.classification_reports, 1):
            __translate_and_save_report(cr, label_dict, i)

        print("Saving predictions on test csv...")
        for i, (key, value) in enumerate(self.predictions.items(), 1):
            test_data = pd.read_csv(os.path.join(data_path, f"fold_{i}_test.csv"))
            test_data["predicted_phenotype"] = value
            test_data["predicted_phenotype"] = test_data["predicted_phenotype"].map(
                label_dict
            )
            test_data["true_phenotype"] = test_data["encoded_phenotype"].map(label_dict)
            test_data.drop(columns="encoded_phenotype", inplace=True)
            test_data.to_csv(
                os.path.join(save_path, f"predictions_fold_{i}.csv"), index=False
            )

        # This part saves the model
        if save_model:
            models_path = os.path.join(save_path, "model")
            os.makedirs(models_path, exist_ok=True)
            print(f"Saving model in {models_path}...")
            # Save the best model for each fold
            for i, model in enumerate(self.best_models):
                model_filename = os.path.join(
                    models_path, f"{self.model_name}_fold_{i+1}_model_default.pkl"
                )
                with open(model_filename, "wb") as f:
                    pickle.dump(model, f)
            print(f"Best models for all folds saved successfully in {models_path}.")
        else:
            print(f"Results saved successfully in {save_path}. Models not saved.")
