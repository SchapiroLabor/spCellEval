"""
plot_kronos_patchsize.py — KRONOS Patch Size Comparison
========================================================
Author: Julia Oesterle
Date:   May 2026

Compares KRONOS LogReg+Optuna at patch_size = 32, 64, 128.
Skips missing patch sizes gracefully.

Usage
-----
source /home/juliaoesterle/eva/venv/bin/activate
python3 plot_kronos_patchsize.py \\
    --plot-dir /home/juliaoesterle/results/kronos_patchsize_plots
"""

import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

CONFIGS = {
    32:  Path('/home/juliaoesterle/results/kronos_patch32/KRONOS_original_supervised/level3'),
    64:  Path('/home/juliaoesterle/results/kronos_original/KRONOS_original_supervised/level3'),
    128: Path('/home/juliaoesterle/results/kronos_patch128/KRONOS_original_supervised/level3'),
}
COLORS = {32: '#C0392B', 64: '#E67E22', 128: '#F0B27A'}
N_FOLDS = 5


def load_results(pred_dir, n_folds):
    fold_reports, all_dfs = {}, []
    for fold in range(n_folds):
        path = pred_dir / f'predictions_{fold}.csv'
        if not path.exists():
            continue
        df    = pd.read_csv(path)
        known = df[df['true_phenotype'] != 'Unknown']
        r     = classification_report(known['true_phenotype'],
                                       known['predicted_phenotype'],
                                       output_dict=True, zero_division=0)
        fold_reports[fold] = r
        all_dfs.append(df)
    all_preds = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return fold_reports, all_preds


def summary(fold_reports):
    folds = sorted(fold_reports.keys())
    if not folds:
        return None
    accs  = [fold_reports[i]['accuracy']                 for i in folds]
    f1s   = [fold_reports[i]['macro avg']['f1-score']    for i in folds]
    wf1s  = [fold_reports[i]['weighted avg']['f1-score'] for i in folds]
    return dict(accuracy=np.mean(accs), accuracy_std=np.std(accs),
                macro_f1=np.mean(f1s),  macro_f1_std=np.std(f1s),
                weighted_f1=np.mean(wf1s), weighted_f1_std=np.std(wf1s),
                n_folds=len(folds))


def get_ct_f1(fold_reports):
    ct_f1 = defaultdict(list)
    for r in fold_reports.values():
        for ct, m in r.items():
            if ct in ['accuracy','macro avg','weighted avg']: continue
            ct_f1[ct].append(m['f1-score'])
    return {ct: (np.mean(v), np.std(v)) for ct, v in ct_f1.items()}


def savefig(fig, plot_dir, fname):
    fig.savefig(plot_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")


def plot_01_head_to_head(all_stats, plot_dir):
    valid  = sorted(all_stats.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, title in zip(axes,
        ['accuracy','macro_f1','weighted_f1'],
        ['Accuracy','Macro F1','Weighted F1']):
        values = [all_stats[c][metric]          for c in valid]
        stds   = [all_stats[c][f'{metric}_std'] for c in valid]
        colors = [COLORS[c]                     for c in valid]
        labels = [f'{c}×{c}px\n({all_stats[c]["n_folds"]} folds)' for c in valid]
        bars   = ax.bar(labels, values, color=colors, alpha=0.85,
                        yerr=stds, capsize=7, error_kw={'elinewidth':2})
        for bar, val, std in zip(bars, values, stds):
            ax.text(bar.get_x()+bar.get_width()/2, val+std+0.015,
                    f"{val:.3f}", ha='center', fontsize=12, fontweight='bold')
        best = int(np.argmax(values))
        bars[best].set_edgecolor('black'); bars[best].set_linewidth(2)
        ax.set_ylim(0, 1.0); ax.set_ylabel('Score')
        ax.set_title(title, fontsize=13)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    plt.suptitle("KRONOS — Patch Size Comparison (LogReg+Optuna)\n"
                 "IMMUcan: 179 Images, 5-Fold CV", fontsize=13)
    plt.tight_layout()
    savefig(fig, plot_dir, 'plot_01_head_to_head.png')


def plot_02_per_fold(all_reports, all_stats, plot_dir):
    valid = sorted(all_reports.keys())
    n     = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(5*n+1, 5))
    if n == 1: axes = [axes]
    for ax, ps in zip(axes, valid):
        reports = all_reports[ps]
        stats   = all_stats[ps]
        folds   = sorted(reports.keys())
        accs    = [reports[i]['accuracy']              for i in folds]
        f1s     = [reports[i]['macro avg']['f1-score'] for i in folds]
        x       = np.arange(len(folds))
        ax.bar(x-0.2, accs, 0.35, label='Accuracy',  color='steelblue', alpha=0.85)
        ax.bar(x+0.2, f1s,  0.35, label='Macro F1',  color=COLORS[ps],  alpha=0.85)
        for xi, (a, f) in enumerate(zip(accs, f1s)):
            ax.text(xi-0.2, a+0.01, f"{a:.3f}", ha='center', fontsize=7)
            ax.text(xi+0.2, f+0.01, f"{f:.3f}", ha='center', fontsize=7)
        ax.axhline(stats['accuracy'], color='steelblue', linestyle='--', alpha=0.3)
        ax.axhline(stats['macro_f1'], color=COLORS[ps],  linestyle='--', alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i}' for i in folds], fontsize=8)
        ax.set_ylim(0, 1.0); ax.set_ylabel('Score'); ax.legend(fontsize=8)
        ax.set_title(f"patch={ps}×{ps}px\n"
                     f"Acc={stats['accuracy']:.3f} | F1={stats['macro_f1']:.3f}")
    plt.suptitle("KRONOS — Per-Fold Performance by Patch Size\nIMMUcan: 179 Images", fontsize=13)
    plt.tight_layout()
    savefig(fig, plot_dir, 'plot_02_per_fold.png')


def plot_03_f1_per_celltype(all_ct_f1, plot_dir):
    valid   = sorted(all_ct_f1.keys())
    all_cts = sorted(set(ct for c in valid for ct in all_ct_f1[c]))
    ref     = all_ct_f1[valid[0]]
    order   = np.argsort([ref.get(ct,(0,0))[0] for ct in all_cts])
    all_cts = [all_cts[i] for i in order]
    offsets = np.linspace(-0.25, 0.25, len(valid))
    fig, ax = plt.subplots(figsize=(12, max(8, len(all_cts)*0.55)))
    y = np.arange(len(all_cts))
    for ps, offset in zip(valid, offsets):
        means = [all_ct_f1[ps].get(ct,(0,0))[0] for ct in all_cts]
        stds  = [all_ct_f1[ps].get(ct,(0,0))[1] for ct in all_cts]
        ax.barh(y+offset, means, 0.22, label=f'{ps}×{ps}px',
                color=COLORS[ps], alpha=0.85,
                xerr=stds, error_kw={'ecolor':'gray','capsize':2})
    ax.set_yticks(y); ax.set_yticklabels(all_cts, fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Mean F1 Score (± std, 5 folds)', fontsize=11)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    ax.set_title("KRONOS — F1 per Cell Type by Patch Size\nIMMUcan: 5-Fold CV", fontsize=12)
    plt.tight_layout()
    savefig(fig, plot_dir, 'plot_03_f1_per_celltype.png')


def plot_04_f1_heatmap(all_ct_f1, plot_dir):
    valid   = sorted(all_ct_f1.keys())
    all_cts = sorted(set(ct for c in valid for ct in all_ct_f1[c]))
    data    = pd.DataFrame(
        {f'{c}px': [all_ct_f1[c].get(ct,(0,0))[0] for ct in all_cts]
         for c in valid}, index=all_cts
    )
    data = data.loc[data.max(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(4+2*len(valid), len(all_cts)*0.55+2))
    sns.heatmap(data.T, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax, linewidths=0.5, annot_kws={'size':10})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    ax.set_title("KRONOS — F1 per Cell Type per Patch Size\nIMMUcan: 5-Fold CV", fontsize=12)
    plt.tight_layout()
    savefig(fig, plot_dir, 'plot_04_f1_heatmap.png')


def plot_05_summary_table(all_stats, plot_dir):
    valid = sorted(all_stats.keys())
    rows  = {f'{c}×{c}px ({all_stats[c]["n_folds"]}F)': {
        'Accuracy': all_stats[c]['accuracy'],
        'Acc ±':    all_stats[c]['accuracy_std'],
        'Macro F1': all_stats[c]['macro_f1'],
        'F1 ±':     all_stats[c]['macro_f1_std'],
        'Weighted F1': all_stats[c]['weighted_f1'],
        'WF1 ±':    all_stats[c]['weighted_f1_std'],
    } for c in valid}
    df      = pd.DataFrame(rows).T
    numeric = ['Accuracy','Macro F1','Weighted F1']
    fig, ax = plt.subplots(figsize=(10, max(3, len(rows)*0.9)))
    sns.heatmap(df[numeric], annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.5, vmax=0.85, ax=ax, linewidths=0.5,
                annot_kws={'size':13,'weight':'bold'})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_title("KRONOS Patch Size Summary — IMMUcan 5-Fold CV\n"
                 "LogReg+Optuna (authors' approach)", fontsize=13)
    for j, metric in enumerate(numeric):
        std_col = {'Accuracy':'Acc ±','Macro F1':'F1 ±','Weighted F1':'WF1 ±'}[metric]
        for i, row_name in enumerate(df.index):
            ax.text(j+0.5, i+0.85, f"±{df.loc[row_name,std_col]:.3f}",
                    ha='center', fontsize=8, color='gray')
    plt.tight_layout()
    savefig(fig, plot_dir, 'plot_05_summary_table.png')


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--plot-dir', required=True)
    p.add_argument('--n-folds',  type=int, default=5)
    args = p.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading KRONOS patch size results...")
    all_reports, all_stats, all_ct_f1 = {}, {}, {}

    for ps, pred_dir in CONFIGS.items():
        if not pred_dir.exists():
            print(f"  patch{ps}: not found — skipping")
            continue
        n_found = len(list(pred_dir.glob('predictions_*.csv')))
        if n_found == 0:
            print(f"  patch{ps}: no predictions — skipping")
            continue
        print(f"  Loading patch{ps} ({n_found} folds)...")
        reports, _ = load_results(pred_dir, args.n_folds)
        if not reports: continue
        all_reports[ps] = reports
        all_stats[ps]   = summary(reports)
        all_ct_f1[ps]   = get_ct_f1(reports)
        s = all_stats[ps]
        print(f"    Acc={s['accuracy']:.3f}±{s['accuracy_std']:.3f} | "
              f"F1={s['macro_f1']:.3f}±{s['macro_f1_std']:.3f}")

    if not all_stats:
        print("No results found!"); return

    plot_01_head_to_head(all_stats, plot_dir)
    plot_02_per_fold(all_reports, all_stats, plot_dir)
    plot_03_f1_per_celltype(all_ct_f1, plot_dir)
    plot_04_f1_heatmap(all_ct_f1, plot_dir)
    plot_05_summary_table(all_stats, plot_dir)

    best = max(all_stats, key=lambda c: all_stats[c]['macro_f1'])
    print(f"\n✓ 5 plots saved to {plot_dir}")
    print(f"Best patch size: {best}×{best}px "
          f"(F1={all_stats[best]['macro_f1']:.3f}, {all_stats[best]['n_folds']} folds)")


if __name__ == '__main__':
    main()