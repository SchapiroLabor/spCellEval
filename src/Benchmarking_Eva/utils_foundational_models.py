"""
utils_benchmark.py — Shared benchmark utilities for foundation model evaluation

Shared functions used by run_eva.py, run_kronos.py, run_virtues.py etc.
Each model script only implements model loading + feature extraction.
Everything else (supervised RF, Leiden, greedy F1, output saving) can be found here.

Functions
---------
load_label_map(data_dir)
load_folds(data_dir, n_folds)
get_label(cell_id, cell_labels, label_map)
load_embeddings(output_dir)
save_embeddings(output_dir, all_feats_arr, metadata_all)
run_supervised(args, folds, img_feature_store, all_feats_arr, metadata_all)
run_leiden(args, folds, img_feature_store, all_feats_arr, metadata_all)
greedy_f1_score(df, true_label_col, predicted_cluster_col, tie_strategy)
add_shared_args(parser)
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (f1_score, adjusted_rand_score,
                                 normalized_mutual_info_score,
                                 matthews_corrcoef, cohen_kappa_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Ground Truth 

def load_label_map(data_dir):
    """Load integer → phenotype name mapping from labels_cell_type.csv."""
    path = Path(data_dir) / 'CellTypes' / 'labels_cell_type.csv'
    df   = pd.read_csv(path)
    return dict(zip(df['label'], df['phenotype']))


def get_label(cell_id, cell_labels, label_map):
    """
    Validated ground truth lookup.
      cell_labels[cell_id - 1] → raw integer label (cell IDs are 1-indexed)
      -1  = Unknown (no annotation) → excluded from training and evaluation
      0-13 = valid phenotype class
    """
    idx = cell_id - 1
    if idx >= len(cell_labels):
        return 'Unknown'
    raw = cell_labels[idx]
    if raw == -1:
        return 'Unknown'
    return label_map.get(raw, 'Unknown')


def load_folds(data_dir, n_folds):
    """Load predefined image-level CV splits from folds.json."""
    folds_file = Path(data_dir) / 'CellTypes' / 'folds.json'
    with open(folds_file) as f:
        folds_data = json.load(f)
    folds = [{'train': folds_data[f'fold_{i}_train_set'],
               'test':  folds_data[f'fold_{i}_test_set']}
             for i in range(n_folds)]
    all_images = sorted(set(
        img for fold in folds for split in ['train', 'test'] for img in fold[split]
    ))
    print(f"Loaded {n_folds} folds — {len(all_images)} total images")
    for i, fold in enumerate(folds):
        print(f"  Fold {i}: {len(fold['train'])} train | {len(fold['test'])} test")
    return folds, all_images


#  Embedding I/O 

def save_embeddings(output_dir, all_feats_arr, metadata_all):
    """Save combined feature matrix and metadata to disk."""
    emb_dir = Path(output_dir) / 'embeddings'
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / 'all_cell_features.npy', all_feats_arr)
    metadata_all.to_csv(emb_dir / 'all_cell_metadata.csv', index=False)
    print(f"Saved embeddings: {all_feats_arr.shape} → {emb_dir}")


def load_embeddings(output_dir):
    """
    Load precomputed embeddings if available.
    Returns (all_feats_arr, metadata_all) or (None, None) if not found.
    """
    emb_dir   = Path(output_dir) / 'embeddings'
    feat_file = emb_dir / 'all_cell_features.npy'
    meta_file = emb_dir / 'all_cell_metadata.csv'
    if feat_file.exists() and meta_file.exists():
        all_feats_arr = np.load(feat_file)
        metadata_all  = pd.read_csv(meta_file)
        print(f"Loaded precomputed embeddings: {all_feats_arr.shape} from {emb_dir}")
        return all_feats_arr, metadata_all
    return None, None


def rebuild_img_feature_store(output_dir, all_images):
    """
    Rebuild per-image feature store from per-image cache files.
    Used when loading precomputed embeddings instead of re-extracting.
    """
    cache_dir = Path(output_dir) / 'embeddings' / 'cache'
    img_feature_store = {}
    for img_name in all_images:
        cf = cache_dir / f"{img_name}_feat.npy"
        cm = cache_dir / f"{img_name}_meta.csv"
        if cf.exists() and cm.exists():
            feats = np.load(cf)
            meta  = pd.read_csv(cm)
            img_feature_store[img_name] = (
                feats, meta['label'].tolist(), meta['cell_id'].tolist()
            )
    print(f"Rebuilt img_feature_store for {len(img_feature_store)} images")
    return img_feature_store


#  Greedy F1 (inline — fallback if spCellEval not available) 

def greedy_f1_score(df, true_label_col, predicted_cluster_col,
                    tie_strategy='first'):
    """
    Assign each Leiden cluster to its majority ground truth label,
    then compute F1 and clustering metrics.

    This is a local reimplementation of spCellEval's greedy_f1_utils.py.
    If the spCellEval repo is available, the original is imported instead
    (see load_greedy_f1()).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain true_label_col and predicted_cluster_col.
    true_label_col : str
    predicted_cluster_col : str
    tie_strategy : 'first' | 'random' | 'raise'

    Returns
    -------
    dict with keys: f1_macro, f1_weighted, accuracy, ari, nmi,
                    mcc, kappa, mapping, mapped_predictions
    """
    
    contingency     = pd.crosstab(df[true_label_col], df[predicted_cluster_col])
    cluster_to_label = {}

    for cluster in df[predicted_cluster_col].unique():
        if cluster not in contingency.columns:
            cluster_to_label[cluster] = None
            continue
        col_counts = contingency[cluster]
        max_count  = col_counts.max()
        top_labels = col_counts[col_counts == max_count].index.tolist()

        if len(top_labels) == 1 or tie_strategy == 'first':
            chosen = top_labels[0]
        elif tie_strategy == 'random':
            chosen = np.random.choice(top_labels)
        elif tie_strategy == 'raise':
            raise ValueError(f"Tie for cluster {cluster}: {top_labels}")
        else:
            raise ValueError(f"Unknown tie_strategy: {tie_strategy}")
        cluster_to_label[cluster] = chosen

    mapped_predictions = np.array([
        cluster_to_label.get(c, 'unmapped')
        for c in df[predicted_cluster_col]
    ])

    valid_mask   = mapped_predictions != 'unmapped'
    y_true_valid = df[true_label_col].values[valid_mask]
    y_pred_valid = mapped_predictions[valid_mask]

    return {
        'f1_macro':           f1_score(y_true_valid, y_pred_valid,
                                       average='macro',    zero_division=0),
        'f1_weighted':        f1_score(y_true_valid, y_pred_valid,
                                       average='weighted', zero_division=0),
        'accuracy':           (y_true_valid == y_pred_valid).mean(),
        'ari':                adjusted_rand_score(y_true_valid, y_pred_valid),
        'nmi':                normalized_mutual_info_score(y_true_valid, y_pred_valid),
        'mcc':                matthews_corrcoef(y_true_valid, y_pred_valid),
        'kappa':              cohen_kappa_score(y_true_valid, y_pred_valid),
        'mapping':            cluster_to_label,
        'mapped_predictions': mapped_predictions,
    }


def load_greedy_f1(spceleval_dir=None):
    """
    Try to import greedy_f1_score from spCellEval repo.
    Falls back to local reimplementation if not found.
    """
    if spceleval_dir is not None:
        greedy_path = Path(spceleval_dir) / 'src' / 'clustering_methods'
        if greedy_path.exists():
            sys.path.insert(0, str(greedy_path))
            try:
                from greedy_f1_utils import greedy_f1_score as _gf1
                print(f"Loaded greedy_f1_score from spCellEval: {greedy_path}")
                return _gf1
            except ImportError:
                pass
    print("greedy_f1_score: using local reimplementation (spCellEval not found)")
    return greedy_f1_score


#  Supervised: Random Forest 

def run_supervised(args, folds, img_feature_store, all_feats_arr, metadata_all,
                   method_name=None):
    """
    5-fold cross-validation with Random Forest classifier.

    Saves to spCellEval folder structure:
        {output_dir}/{method_name}/level3/predictions_{fold}.csv
        {output_dir}/{method_name}/level3/fold_times.txt

    CSV columns: image_id, cell_id, fold, true_phenotype, predicted_phenotype,
                 confidence, emb_0 ... emb_767

    Parameters
    ----------
    args        : argparse.Namespace — must have: n_estimators, n_jobs,
                  n_folds, output_dir, spceleval_dir
    folds       : list of {'train': [...], 'test': [...]}
    img_feature_store : dict img_name → (feats_arr, labels, cell_ids)
    all_feats_arr     : (N_cells, D) numpy array
    metadata_all      : pd.DataFrame with image_id, cell_id, label
    method_name       : override output folder name (default: EVA_supervised_{embedding_mode})
    """


    if method_name is None:
        method_name = f"EVA_supervised_{args.embedding_mode}"

    sds_base  = Path(getattr(args, 'spceleval_dir', None) or args.output_dir)
    out_dir   = sds_base / method_name / 'level3'
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_cols = [f"emb_{i}" for i in range(all_feats_arr.shape[1])]

    print(f"\nSupervised RF | n_estimators={args.n_estimators} | "
          f"n_jobs={args.n_jobs} | folds={args.n_folds}")
    print("=" * 60)

    fold_times   = []
    fold_reports = {}
    all_preds    = []

    for fold_idx in range(args.n_folds):
        fold_start   = time.time()
        train_images = folds[fold_idx]['train']
        test_images  = folds[fold_idx]['test']
        print(f"\n── Fold {fold_idx} ──────────────────────")

        # Collect train cells — exclude Unknown
        X_train, y_train = [], []
        for img_name in train_images:
            if img_name not in img_feature_store:
                continue
            feats, labels, _ = img_feature_store[img_name]
            mask_known = np.array(labels) != 'Unknown'
            X_train.extend(np.array(feats)[mask_known])
            y_train.extend(np.array(labels)[mask_known])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        print(f"  Train: {len(X_train):,} cells | {len(train_images)} images")

        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=2,
            n_jobs=args.n_jobs,          # user-controlled parallelism
            random_state=42,
            class_weight='balanced',     # corrects for Cancer class dominance -> tbd! 
        )
        clf.fit(X_train, y_train)

        # Predict test cells
        fold_preds = []
        for img_name in test_images:
            if img_name not in img_feature_store:
                continue
            feats, labels, cell_ids = img_feature_store[img_name]
            feats   = np.array(feats)
            y_pred  = clf.predict(feats)
            y_proba = clf.predict_proba(feats)
            for cell_id, true_label, pred_label, proba in zip(
                cell_ids, labels, y_pred, y_proba
            ):
                fold_preds.append({
                    'image_id':            img_name,
                    'cell_id':             cell_id,
                    'fold':                fold_idx,
                    'true_phenotype':      true_label,
                    'predicted_phenotype': pred_label,
                    'confidence':          proba.max(),
                })

        fold_df = pd.DataFrame(fold_preds)

        # Add embeddings via single pd.concat (avoids PerformanceWarning)
        test_mask    = metadata_all['image_id'].isin(test_images)
        test_indices = metadata_all[test_mask].index.values
        emb_df       = pd.DataFrame(all_feats_arr[test_indices], columns=feat_cols)
        meta_test    = metadata_all[test_mask].reset_index(drop=True)
        emb_lookup   = pd.concat(
            [meta_test[['image_id', 'cell_id']].reset_index(drop=True), emb_df],
            axis=1
        )
        fold_df = fold_df.merge(emb_lookup, on=['image_id', 'cell_id'], how='left')

        # Save predictions CSV
        fold_df.to_csv(out_dir / f"predictions_{fold_idx}.csv", index=False)

        fold_time = time.time() - fold_start
        fold_times.append(fold_time)

        # Evaluate on known labels
        known  = fold_df[fold_df['true_phenotype'] != 'Unknown']
        report = classification_report(
            known['true_phenotype'], known['predicted_phenotype'],
            output_dict=True, zero_division=0
        )
        fold_reports[fold_idx] = report
        print(f"  Test:     {len(fold_df):,} cells | {len(test_images)} images")
        print(f"  Accuracy: {report['accuracy']:.3f} | "
              f"Macro F1: {report['macro avg']['f1-score']:.3f}")
        print(f"  Time:     {fold_time:.1f}s")
        all_preds.append(fold_df)

    # Save fold_times.txt
    _save_fold_times(out_dir, fold_times, prefix='fold')

    # Print summary
    accs  = [fold_reports[i]['accuracy']                 for i in range(args.n_folds)]
    f1s   = [fold_reports[i]['macro avg']['f1-score']    for i in range(args.n_folds)]
    wf1s  = [fold_reports[i]['weighted avg']['f1-score'] for i in range(args.n_folds)]
    print(f"\n{method_name} Results:")
    print(f"  Accuracy:    {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Macro F1:    {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"  Weighted F1: {np.mean(wf1s):.3f} ± {np.std(wf1s):.3f}")
    print(f"  Output:      {out_dir}")

    return fold_reports


#  Leiden + Greedy F1 

def run_leiden(args, folds, img_feature_store, all_feats_arr, metadata_all,
               method_name=None):
    """
    Leiden clustering on all cell embeddings + greedy F1 majority assignment.

    Saves to spCellEval folder structure:
        {output_dir}/{method_name}/level3/predictions_{fold}.csv
        {output_dir}/{method_name}/level3/fold_times.txt

    CSV columns: image_id, cell_id, fold, true_phenotype, predicted_phenotype,
                 leiden_cluster, emb_0 ... emb_767

    Parameters
    ----------
    args        : argparse.Namespace — must have: leiden_resolution,
                  leiden_n_neighbors, leiden_subsample, n_jobs,
                  n_folds, output_dir, spceleval_dir
    folds       : list of {'train': [...], 'test': [...]}
    img_feature_store : dict img_name → (feats_arr, labels, cell_ids)
    all_feats_arr     : (N_cells, D) numpy array
    metadata_all      : pd.DataFrame with image_id, cell_id, label
    method_name       : override output folder name
    """
    

    if method_name is None:
        method_name = f"EVA_leiden_{args.embedding_mode}"

    gf1_fn    = load_greedy_f1(getattr(args, 'spceleval_dir', None))
    sds_base  = Path(getattr(args, 'spceleval_dir', None) or args.output_dir)
    out_dir   = sds_base / method_name / 'level3'
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_cols = [f"emb_{i}" for i in range(all_feats_arr.shape[1])]

    print(f"\nLeiden | res={args.leiden_resolution} | "
          f"n_neighbors={args.leiden_n_neighbors} | "
          f"subsample={args.leiden_subsample}")
    print("=" * 60)

    leiden_start = time.time()

    # Build AnnData on known-label cells
    known_mask = metadata_all['label'] != 'Unknown'
    X_known    = all_feats_arr[known_mask.values]
    y_known    = metadata_all.loc[known_mask, 'label'].values
    meta_known = metadata_all[known_mask].reset_index(drop=True)

    adata = sc.AnnData(X=X_known)
    adata.obs['true_label'] = pd.Categorical(y_known)
    adata.obs['image_id']   = meta_known['image_id'].values
    adata.obs['cell_id']    = meta_known['cell_id'].values.astype(str)

    if len(adata) > args.leiden_subsample:
        print(f"  Subsampling to {args.leiden_subsample:,} cells...")
        sc.pp.subsample(adata, n_obs=args.leiden_subsample, random_state=42)

    print(f"  Computing KNN graph (n_neighbors={args.leiden_n_neighbors})...")
    sc.pp.neighbors(adata, n_neighbors=args.leiden_n_neighbors, use_rep='X')
    print(f"  Running Leiden (res={args.leiden_resolution}, igraph backend)...")
    sc.tl.leiden(
        adata,
        resolution=args.leiden_resolution,
        random_state=42,
        flavor='igraph',     # fast backend — avoids FutureWarning and was recommended by scanpy team for large datasets
        n_iterations=2,      # required for igraph
        directed=False,      # required for igraph
    )
    n_clusters = adata.obs['leiden'].nunique()
    print(f"  Found {n_clusters} clusters")
    print("  Computing UMAP...")
    sc.tl.umap(adata, random_state=42)

    # Save Leiden results
    leiden_df = pd.DataFrame({
        'image_id':       adata.obs['image_id'].values,
        'cell_id':        adata.obs['cell_id'].values,
        'true_label':     adata.obs['true_label'].values,
        'leiden_cluster': adata.obs['leiden'].values,
        'umap_1':         adata.obsm['X_umap'][:, 0],
        'umap_2':         adata.obsm['X_umap'][:, 1],
    })
    emb_dir = Path(args.output_dir) / 'embeddings'
    emb_dir.mkdir(parents=True, exist_ok=True)
    leiden_df.to_csv(
        emb_dir / f'leiden_clusters_res{args.leiden_resolution}.csv',
        index=False
    )

    # Greedy F1 on subsampled clusters
    greedy = gf1_fn(leiden_df, 'true_label', 'leiden_cluster',
                    tie_strategy='first')
    print(f"\nGreedy F1 (res={args.leiden_resolution}, n={n_clusters} clusters):")
    print(f"  Accuracy:    {greedy['accuracy']:.3f}")
    print(f"  Macro F1:    {greedy['f1_macro']:.3f}")
    print(f"  Weighted F1: {greedy['f1_weighted']:.3f}")
    print(f"  ARI:         {greedy['ari']:.3f}")
    print(f"  NMI:         {greedy['nmi']:.3f}")

    # Save greedy metrics CSV
    pd.DataFrame([{
        'method':      method_name,
        'leiden_res':  args.leiden_resolution,
        'n_clusters':  n_clusters,
        'accuracy':    greedy['accuracy'],
        'f1_macro':    greedy['f1_macro'],
        'f1_weighted': greedy['f1_weighted'],
        'ari':         greedy['ari'],
        'nmi':         greedy['nmi'],
        'mcc':         greedy.get('mcc', float('nan')),
        'kappa':       greedy.get('kappa', float('nan')),
    }]).to_csv(emb_dir / f'greedy_f1_metrics_{method_name}.csv', index=False)

    # Assign Leiden clusters to ALL cells via KNN
    print("\n  Assigning clusters to all cells via KNN...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=args.n_jobs)
    knn.fit(adata.X, adata.obs['leiden'].values)
    all_leiden = knn.predict(all_feats_arr)

    metadata_all = metadata_all.copy()
    metadata_all['leiden_cluster']   = all_leiden
    metadata_all['leiden_predicted'] = [
        greedy['mapping'].get(str(c), greedy['mapping'].get(c, 'unmapped'))
        for c in all_leiden
    ]

    leiden_cluster_time = time.time() - leiden_start
    fold_times = []

    # Save per-fold predictions
    for fold_idx in range(args.n_folds):
        fold_start  = time.time()
        test_images = folds[fold_idx]['test']

        fold_mask    = (metadata_all['image_id'].isin(test_images) &
                        (metadata_all['label'] != 'Unknown'))
        fold_meta    = metadata_all[fold_mask].reset_index(drop=True)
        test_indices = metadata_all[fold_mask].index.values

        emb_df  = pd.DataFrame(all_feats_arr[test_indices], columns=feat_cols)
        base_df = fold_meta[['image_id', 'cell_id', 'label',
                             'leiden_cluster', 'leiden_predicted']].rename(columns={
            'label':            'true_phenotype',
            'leiden_predicted': 'predicted_phenotype',
        }).reset_index(drop=True)
        base_df['fold'] = fold_idx

        out_df = pd.concat([base_df, emb_df], axis=1)
        out_df.to_csv(out_dir / f"predictions_{fold_idx}.csv", index=False)

        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        print(f"  Fold {fold_idx}: {len(out_df):,} cells → "
              f"predictions_{fold_idx}.csv ({fold_time:.1f}s)")

    # Save fold_times.txt
    _save_fold_times(out_dir, fold_times,
                     prefix='fold_assignment',
                     extra={'leiden_clustering': leiden_cluster_time})

    print(f"\n{method_name} complete | output: {out_dir}")
    return greedy


#  Output Helpers 

def _save_fold_times(out_dir, fold_times, prefix='fold', extra=None):
    """Save fold timing to fold_times.txt in spCellEval format."""
    with open(Path(out_dir) / 'fold_times.txt', 'w') as f:
        if extra:
            for key, val in extra.items():
                f.write(f"{key}: {val:.2f}s\n")
        for i, t in enumerate(fold_times):
            f.write(f"{prefix}_{i}: {t:.2f}s\n")
        f.write(f"total: {sum(fold_times) + sum((extra or {}).values()):.2f}s\n")
        f.write(f"mean_fold: {np.mean(fold_times):.2f}s\n")


#  Shared CLI Arguments 

def add_shared_args(parser):
    """
    Add shared CLI arguments to any model's argument parser.
    Call this from run_eva.py, run_kronos.py, run_virtues.py etc.
    after adding model-specific arguments.
    """
    # Required paths
    parser.add_argument('--data-dir', required=True,
                        help='Root of IMMUcan dataset (contains CellTypes/, segmentation/)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for embeddings and predictions')
    parser.add_argument('--spceleval-dir', default=None,
                        help='spCellEval root dir — used for output structure '
                             'and greedy_f1_utils.py import')

    # CV
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of cross-validation folds')

    # Supervised RF
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='[supervised] Random Forest: number of trees')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 = all cores). '
                             'Used by Random Forest and KNN.')

    # Leiden
    parser.add_argument('--leiden-resolution', type=float, default=2.0,
                        help='[leiden] Leiden clustering resolution')
    parser.add_argument('--leiden-n-neighbors', type=int, default=15,
                        help='[leiden] KNN graph neighbours for Leiden')
    parser.add_argument('--leiden-subsample', type=int, default=50000,
                        help='[leiden] Max cells for Leiden (subsampled for speed)')

    return parser
