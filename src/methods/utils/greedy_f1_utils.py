# greedy_f1_utils.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, adjusted_rand_score, normalized_mutual_info_score,
    matthews_corrcoef, cohen_kappa_score
)

def greedy_f1_score(df, true_label_col, predicted_cluster_col, tie_strategy="first"):
    contingency = pd.crosstab(df[true_label_col], df[predicted_cluster_col])
    cluster_to_label = {}

    for cluster in df[predicted_cluster_col].unique():
        if cluster not in contingency.columns:
            cluster_to_label[cluster] = None
            continue

        col_counts = contingency[cluster]
        max_count = col_counts.max()
        top_labels = col_counts[col_counts == max_count].index.tolist()

        if len(top_labels) == 1 or tie_strategy == "first":
            chosen_label = top_labels[0]
        elif tie_strategy == "random":
            chosen_label = np.random.choice(top_labels)
        elif tie_strategy == "raise":
            raise ValueError(f"Tie detected for cluster {cluster}: {top_labels}")
        else:
            raise ValueError(f"Unknown tie_strategy: {tie_strategy}")

        cluster_to_label[cluster] = chosen_label

    mapped_predictions = np.array([
        cluster_to_label.get(cluster, "unmapped") for cluster in df[predicted_cluster_col]
    ])

    valid_mask = mapped_predictions != "unmapped"
    y_true_valid = df[true_label_col].values[valid_mask]
    mapped_valid = mapped_predictions[valid_mask]

    f1_macro = f1_score(y_true_valid, mapped_valid, average='macro')
    f1_weighted = f1_score(y_true_valid, mapped_valid, average='weighted')
    ari = adjusted_rand_score(y_true_valid, mapped_valid)
    nmi = normalized_mutual_info_score(y_true_valid, mapped_valid)
    accuracy = (y_true_valid == mapped_valid).mean()
    mcc = matthews_corrcoef(y_true_valid, mapped_valid)
    kappa = cohen_kappa_score(y_true_valid, mapped_valid)

    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'ari': ari,
        'nmi': nmi,
        'accuracy': accuracy,
        'mcc': mcc,
        'kappa': kappa,
        'mapping': cluster_to_label,
        'mapped_predictions': mapped_predictions
    }