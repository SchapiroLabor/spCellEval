import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Union, List
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score
from data_handler import DataSetHandler
from logistic_regression import MultinomialLogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

path_to_data = 'data/quantified/processed/kfolds/'

AML = MultinomialLogisticRegression(random_state=42, max_iter=1000, c=1.0, penalty='elasticnet', n_jobs=2, l1_ratio=0.5)