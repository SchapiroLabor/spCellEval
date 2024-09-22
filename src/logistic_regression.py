import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self, random_state: int, data_handler) -> None:
        self.random_state = random_state
        self.data_handler = data_handler
        self.model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=self.data_handler.random_state)
        self.scaler = StandardScaler()

    def train_and_evaluate(self) -> None:
        