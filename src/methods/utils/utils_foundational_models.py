"""
utils_foundational_models.py — Shared benchmark utilities for foundation
model evaluation.

Shared functions used by run_eva.py, run_kronos.py, run_virtues.py etc.
Each model script only implements model loading + feature extraction.
Everything else (supervised LogReg, Leiden, greedy F1, output saving) can be
found here.

Functions
---------
save_embeddings(output_dir, all_feats_arr, metadata_all)
load_embeddings(output_dir)
check_cache_params(cache_dir, params)
check_fold_coverage(folds, available_keys, splits=('train', 'test'))
run_supervised(args, folds, all_feats_arr, metadata_all, method_name)
run_leiden(args, folds, all_feats_arr, metadata_all, method_name)
greedy_f1_score(df, true_label_col, predicted_cluster_col, tie_strategy)
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


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
        metadata_all  = pd.read_csv(meta_file, dtype={'image_id': str})
        print(f"Loaded precomputed embeddings: {all_feats_arr.shape} from {emb_dir}")
        return all_feats_arr, metadata_all
    return None, None


def check_cache_params(cache_dir, params):
    """
    Bind a feature cache to its extraction settings: params.json is written on
    first use, and a cache written with other settings (or without params.json)
    is refused.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / 'params.json'
    params = json.loads(json.dumps(params))
    if path.exists():
        cached = json.loads(path.read_text())
        if cached != params:
            changed = sorted(k for k in set(cached) | set(params) if cached.get(k) != params.get(k))
            raise ValueError(f"{cache_dir} was extracted with different settings ({changed}); "
                             f"use a new --output-dir")
        return
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise ValueError(f"{cache_dir} holds features without params.json; use a new --output-dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=1))

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
    from sklearn.metrics import (f1_score, adjusted_rand_score,
                                 normalized_mutual_info_score,
                                 matthews_corrcoef, cohen_kappa_score)

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
        greedy_path = Path(spceleval_dir) / 'src' / 'methods' / 'utils'
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

#  Fold coverage

def check_fold_coverage(folds, available_keys, splits=('train', 'test')):
    """
    Raise if any (image_id, cell_id) of a cell-level fold has no embedding.
    Without this check such cells would silently drop out of training/test.
    """
    available = set(available_keys)
    for fold_idx, fold in enumerate(folds):
        for split in splits:
            missing = fold[split] - available
            if missing:
                images = sorted({img for img, _ in missing})
                raise ValueError(
                    f"Fold {fold_idx}: {len(missing):,} {split} cells have no embedding "
                    f"(images: {images[:5]}{' ...' if len(images) > 5 else ''})"
                )

#  Supervised: Logistic Regression

def run_supervised(args, folds, all_feats_arr, metadata_all, method_name):
    """
    Cross-validation with a logistic-regression head on the frozen embeddings
    (StandardScaler + LogisticRegression), the protocol shared by all FM methods.

    Saves to spCellEval folder structure:
        {output_dir}/{method_name}/level3/predictions_{fold}.csv
        {output_dir}/{method_name}/level3/fold_times.txt

    CSV columns: image_id, cell_id, true_phenotype, fold, predicted_phenotype,
                 confidence, emb_0 ... emb_N

    Parameters
    ----------
    args          : argparse.Namespace with output_dir, spceleval_dir, logreg_c
    folds         : list of {'train': set, 'test': set} of (image_id, cell_id)
    all_feats_arr : (N_cells, D) numpy array
    metadata_all  : pd.DataFrame with image_id, cell_id, label (row-aligned)
    method_name   : output folder name
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report

    sds_base  = Path(getattr(args, 'spceleval_dir', None) or args.output_dir)
    out_dir   = sds_base / method_name / 'level3'
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_cols = [f"emb_{i}" for i in range(all_feats_arr.shape[1])]

    print(f"\nSupervised LogReg | C={args.logreg_c} | folds={len(folds)}")
    print("=" * 60)

    cell_keys = pd.Series(list(zip(metadata_all['image_id'], metadata_all['cell_id'])),
                          index=metadata_all.index)
    check_fold_coverage(folds, cell_keys)

    fold_times, train_times, predict_times, fold_reports = [], [], [], {}
    for fold_idx, fold in enumerate(folds):
        fold_start = time.time()
        print(f"\n── Fold {fold_idx} ──────────────────────")
        train_mask = cell_keys.isin(fold['train']).values
        test_mask  = cell_keys.isin(fold['test']).values
        print(f"  Train: {train_mask.sum():,} cells | "
              f"{metadata_all.loc[train_mask, 'image_id'].nunique()} images")

        train_start = time.time()
        scaler = StandardScaler()
        clf = LogisticRegression(C=args.logreg_c, max_iter=2000, random_state=42,
                                 class_weight='balanced', solver='lbfgs')
        clf.fit(scaler.fit_transform(all_feats_arr[train_mask]),
                metadata_all.loc[train_mask, 'label'].values)
        train_time = time.time() - train_start

        predict_start = time.time()
        X_test  = scaler.transform(all_feats_arr[test_mask])
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        predict_time = time.time() - predict_start

        fold_df = metadata_all.loc[test_mask, ['image_id', 'cell_id', 'label']].rename(
            columns={'label': 'true_phenotype'}).reset_index(drop=True)
        fold_df['fold']                = fold_idx
        fold_df['predicted_phenotype'] = y_pred
        fold_df['confidence']          = y_proba.max(axis=1)
        fold_df = pd.concat([fold_df, pd.DataFrame(all_feats_arr[test_mask], columns=feat_cols)], axis=1)
        fold_df.to_csv(out_dir / f"predictions_{fold_idx}.csv", index=False)

        fold_times.append(time.time() - fold_start)
        train_times.append(train_time)
        predict_times.append(predict_time)

        report = classification_report(fold_df['true_phenotype'], fold_df['predicted_phenotype'],
                                       output_dict=True, zero_division=0)
        fold_reports[fold_idx] = report
        print(f"  Test:     {len(fold_df):,} cells | {fold_df['image_id'].nunique()} images")
        print(f"  Accuracy: {report['accuracy']:.3f} | "
              f"Macro F1: {report['macro avg']['f1-score']:.3f}")
        print(f"  Time:     {fold_times[-1]:.1f}s (train={train_time:.1f}s, predict={predict_time:.1f}s)")

    _save_fold_times(out_dir, fold_times, prefix='fold',
                     train_times=train_times, predict_times=predict_times)

    accs = [r['accuracy']                 for r in fold_reports.values()]
    f1s  = [r['macro avg']['f1-score']    for r in fold_reports.values()]
    wf1s = [r['weighted avg']['f1-score'] for r in fold_reports.values()]
    print(f"\n{method_name} Results:")
    print(f"  Accuracy:    {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Macro F1:    {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"  Weighted F1: {np.mean(wf1s):.3f} ± {np.std(wf1s):.3f}")
    print(f"  Output:      {out_dir}")
    return fold_reports

#  Leiden + Greedy F1 

def run_leiden(args, folds, all_feats_arr, metadata_all, method_name):
    """
    Leiden clustering on all cell embeddings + greedy F1 majority assignment.

    Saves to spCellEval folder structure:
        {output_dir}/{method_name}/level3/predictions_{fold}.csv
        {output_dir}/{method_name}/level3/fold_times.txt

    CSV columns: image_id, cell_id, fold, true_phenotype, predicted_phenotype,
                 leiden_cluster, emb_0 ... emb_767

    Parameters
    ----------
    args          : argparse.Namespace with leiden_resolution, leiden_n_neighbors,
                    leiden_subsample, output_dir, spceleval_dir
    folds         : list of {'train': set, 'test': set} of (image_id, cell_id);
                    only selects the cells written to each predictions_{fold}.csv,
                    the clustering always runs over all cells
    all_feats_arr : (N_cells, D) numpy array
    metadata_all  : pd.DataFrame with image_id, cell_id, label (row-aligned)
    method_name   : output folder name
    """
    import scanpy as sc
    from sklearn.neighbors import KNeighborsClassifier

    check_fold_coverage(folds, zip(metadata_all['image_id'], metadata_all['cell_id']),
                        splits=('test',))

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

    adata = sc.AnnData(X=all_feats_arr)
    adata.obs['true_label'] = pd.Categorical(metadata_all['label'].values)
    adata.obs['image_id']   = metadata_all['image_id'].values
    adata.obs['cell_id']    = metadata_all['cell_id'].values.astype(str)

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
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=getattr(args, 'n_jobs', None))
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

    cell_keys = pd.Series(list(zip(metadata_all['image_id'], metadata_all['cell_id'])),
                          index=metadata_all.index)

    # Save per-fold predictions
    for fold_idx, fold in enumerate(folds):
        fold_start = time.time()
        fold_mask  = cell_keys.isin(fold['test']).values

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

def _save_fold_times(out_dir, fold_times, prefix='fold', extra=None,
                     train_times=None, predict_times=None):
    """
    Save fold timing to fold_times.txt in spCellEval format.
    If train_times/predict_times are given (same length as fold_times), each
    fold line also reports the train/predict split, plus totals for each.
    """
    with open(Path(out_dir) / 'fold_times.txt', 'w') as f:
        if extra:
            for key, val in extra.items():
                f.write(f"{key}: {val:.2f}s\n")
        for i, t in enumerate(fold_times):
            if train_times is not None and predict_times is not None:
                f.write(f"{prefix}_{i}: {t:.2f}s "
                       f"(train={train_times[i]:.2f}s, predict={predict_times[i]:.2f}s)\n")
            else:
                f.write(f"{prefix}_{i}: {t:.2f}s\n")
        f.write(f"total: {sum(fold_times) + sum((extra or {}).values()):.2f}s\n")
        if train_times is not None:
            f.write(f"total_train: {sum(train_times):.2f}s\n")
        if predict_times is not None:
            f.write(f"total_predict: {sum(predict_times):.2f}s\n")
        f.write(f"mean_fold: {np.mean(fold_times):.2f}s\n")
