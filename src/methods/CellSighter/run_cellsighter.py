#!/usr/bin/env python3
"""
run_cellsighter.py — CellSighter (KerenLab/CellSighter) on the shared benchmark data and folds
=============================================================================================

Runs the authors' train.py and eval.py unchanged from a CellSighter checkout
(--cellsighter-dir), as in Amitay et al., Nat Commun 2023: an ensemble of
n_runs ResNet50s per fold, each trained for epoch_max epochs on raw images with
the authors' augmentations and class-balanced sampling; the class probabilities
of the ensemble are averaged and the maximum gives the prediction.

The Helix folds are grouped by image, so each fold maps onto CellSighter's image
lists: train_set = the fold's training (+ validation) images, evaluated on the
fold's test images. The validation cells of the Helix folds lie in the training
images, and CellSighter reads one label file per image, so all labelled cells of
the training images are trained on. Labels are the processed cell_type of every cell.

Modes
-----
prepare   Write the authors' data layout to <output-dir>/CellTypes (once per dataset).
train     Train ensemble member --run of fold --fold and evaluate it on the fold's test images
          (one GPU; run several in parallel).
predict   Average the members of each fold into CELLSIGHTER_supervised/level3/predictions_<fold>.csv.

Usage
-----
python src/methods/CellSighter/run_cellsighter.py {prepare,train,predict} \\
    --dataset-path    <data>/IMMUcan \\
    --config          src/methods/configs/cellsighter.json \\
    --cellsighter-dir <KerenLab/CellSighter checkout> \\
    --output-dir      <results>/IMMUcan/cellsighter \\
    [--fold 0 --run 0 --device cuda:0]
"""

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from benchmark_io import (  # noqa: E402
    fill_from_config, load_benchmark_data, load_config, read_image, read_markers, read_segmentation,
)

# config keys passed through to the authors' config.json
CELLSIGHTER_KEYS = ['crop_input_size', 'crop_size', 'epoch_max', 'lr', 'batch_size', 'num_workers',
                    'sample_batch', 'to_pad', 'hierarchy_match', 'aug']


def image_sets(folds, fold):
    """Image ids of a fold's training and test cells (the Helix folds are grouped by image)."""
    train, test = ({img for img, _ in folds[fold][split]} for split in ('train', 'test'))
    if train & test:
        raise ValueError(f"fold {fold}: images in both train and test: {sorted(train & test)}")
    return sorted(train), sorted(test)


def prepare(args, cfg, meta_df, img_paths, seg_paths):
    """Images (H, W, C), masks and per-cell label files in the layout CellSighter's loader reads."""
    root = Path(args.output_dir) / 'CellTypes'
    for sub in ('data/images', 'cells', 'cells2labels'):
        (root / sub).mkdir(parents=True, exist_ok=True)
    markers = read_markers(Path(args.dataset_path))
    keep = [i for i, m in enumerate(markers) if m not in set(cfg.get('exclude_markers', []))]
    (root / 'channels.txt').write_text('\n'.join(markers[i] for i in keep) + '\n')
    classes = sorted(meta_df['label'].unique())
    (root / 'label_map.json').write_text(json.dumps(classes, indent=1))
    label_id = {c: i for i, c in enumerate(classes)}

    for img_id, cells in meta_df.groupby('image_id'):
        seg = read_segmentation(seg_paths[img_id])
        labels = np.full(int(seg.max()) + 1, -1)  # indexed by cell id; -1 = not scored
        labels[cells['cell_id'].to_numpy()] = cells['label'].map(label_id).to_numpy()
        img = read_image(img_paths[img_id], keep).transpose(1, 2, 0)  # float32, raw intensities
        np.savez(root / 'data' / 'images' / f'{img_id}.npz', data=img)
        np.savez(root / 'cells' / f'{img_id}.npz', data=seg)
        (root / 'cells2labels' / f'{img_id}.txt').write_text('\n'.join(map(str, labels)) + '\n')
    print(f"Prepared {meta_df['image_id'].nunique()} images, {len(keep)} channels, "
          f"{len(classes)} classes -> {root}")


def run_authors_script(script, run_dir, cfg_json, args):
    """Write config.json and run the authors' train.py / eval.py on it (they use device 'cuda')."""
    (run_dir / 'config.json').write_text(json.dumps(cfg_json, indent=1))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device.removeprefix('cuda:'))
    subprocess.run([sys.executable, script, '--base_path', str(run_dir)],
                   cwd=args.cellsighter_dir, env=env, check=True)


def authors_config(args, cfg, train_set, val_set, weights=''):
    root = Path(args.output_dir)
    classes = json.loads((root / 'CellTypes' / 'label_map.json').read_text())
    return {**{k: cfg[k] for k in CELLSIGHTER_KEYS}, 'root_dir': str(root), 'train_set': train_set,
            'val_set': val_set, 'num_classes': len(classes), 'blacklist': [],
            'channels_path': str(root / 'CellTypes' / 'channels.txt'), 'weight_to_eval': weights}


def evaluate(args, cfg, folds, fold, run_dir):
    """The authors' eval.py with the run's last weights on the fold's test images -> val_results.csv."""
    train_set, test_set = image_sets(folds, fold)
    weights = run_dir / f"weights_{cfg['epoch_max'] - 1}_count.pth"
    run_authors_script('eval.py', run_dir, authors_config(args, cfg, train_set, test_set, str(weights)), args)


def train(args, cfg, folds):
    train_set, _ = image_sets(folds, args.fold)
    # The authors' train.py scores val_set every 10 epochs (monitoring only; it needs a non-empty set).
    # By default that is the training images themselves; --holdout-images N instead holds out N
    # whole training images (to choose epoch_max). The fold's test images are never loaded here.
    if args.holdout_images:
        rng = np.random.default_rng(args.fold)
        held_out = sorted(rng.choice(train_set, args.holdout_images, replace=False))
        train_set, monitor = sorted(set(train_set) - set(held_out)), held_out
    else:
        monitor = train_set
    stage = 'pilot' if args.holdout_images else 'runs'
    run_dir = Path(args.output_dir) / stage / f'fold{args.fold}' / f'run{args.run}'
    run_dir.mkdir(parents=True, exist_ok=True)
    run_authors_script('train.py', run_dir, authors_config(args, cfg, train_set, monitor), args)
    last = run_dir / f"weights_{cfg['epoch_max'] - 1}_count.pth"
    for w in run_dir.glob('weights_*_count.pth'):
        if w != last:
            w.unlink()  # ResNet50 checkpoints of every epoch; only the last one is evaluated
    if not args.holdout_images:
        evaluate(args, cfg, folds, args.fold, run_dir)
    print(f"fold {args.fold} run {args.run}: {len(train_set)} training images -> {run_dir}")


def predict(args, cfg, folds, meta_df):
    classes = json.loads((Path(args.output_dir) / 'CellTypes' / 'label_map.json').read_text())
    out_dir = Path(args.output_dir) / 'CELLSIGHTER_supervised' / 'level3'
    out_dir.mkdir(parents=True, exist_ok=True)
    last = f"weights_{cfg['epoch_max'] - 1}_count.pth"
    for fold in range(len(folds)) if args.fold is None else [args.fold]:
        runs = sorted((Path(args.output_dir) / 'runs' / f'fold{fold}').glob(f'run*/{last}'))
        if len(runs) != cfg['n_runs']:
            raise FileNotFoundError(f"fold {fold}: {len(runs)} of {cfg['n_runs']} trained runs found")
        test = meta_df[[(i, c) in folds[fold]['test'] for i, c in zip(meta_df['image_id'], meta_df['cell_id'])]]
        keys = pd.MultiIndex.from_frame(test[['image_id', 'cell_id']])
        p = 0
        for weights in runs:  # the authors' ensemble: mean of the members' class probabilities
            if not (weights.parent / 'val_results.csv').exists():
                evaluate(args, cfg, folds, fold, weights.parent)
            res = pd.read_csv(weights.parent / 'val_results.csv', dtype={'image_id': str})
            res = res.set_index(['image_id', 'cell_id'])['prob_list'].map(ast.literal_eval)
            p = p + np.stack(res.loc[keys].to_numpy()) / len(runs)  # KeyError if a test cell is missing
        test.rename(columns={'label': 'true_phenotype'}).assign(
            fold=fold, predicted_phenotype=np.array(classes)[p.argmax(1)], confidence=p.max(1),
        ).to_csv(out_dir / f'predictions_{fold}.csv', index=False)
        print(f"fold {fold}: {len(runs)} runs, {len(test):,} test cells -> {out_dir}/predictions_{fold}.csv")


def build_parser():
    parser = argparse.ArgumentParser(prog='run_cellsighter.py', description='CellSighter benchmark pipeline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=['prepare', 'train', 'predict'])
    parser.add_argument('--dataset-path', required=True,
                        help='Dataset directory (markers.txt, quantification/, raw_images/, segmentation/)')
    parser.add_argument('--config', required=True, help='Path to cellsighter.json')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cellsighter-dir', default=None, help='KerenLab/CellSighter checkout (train, predict)')
    parser.add_argument('--folds-dir', default=None,
                        help='k-fold directory, absolute or relative to --dataset-path')
    parser.add_argument('--include-validation', action=argparse.BooleanOptionalAction, default=None,
                        help="Train on the folds' validation images too")
    parser.add_argument('--fold', type=int, default=None, help='train: required; predict: default all folds')
    parser.add_argument('--run', type=int, default=None, help='train: ensemble member, 0 .. n_runs-1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epoch-max', type=int, default=None)
    parser.add_argument('--holdout-images', type=int, default=0,
                        help='train: hold out N training images and monitor them (choosing epoch_max)')
    return parser


def main():
    args = build_parser().parse_args()
    data_root = Path(args.dataset_path).resolve()
    args.output_dir = str(Path(args.output_dir).resolve())
    cfg = load_config(args.config, data_root.name)
    fill_from_config(args, cfg)
    cfg['epoch_max'] = args.epoch_max
    meta_df, folds, img_paths, seg_paths = load_benchmark_data(
        data_root, data_root / args.folds_dir, cfg.get('exclude_labels', []), args.include_validation)

    if args.mode == 'prepare':
        prepare(args, cfg, meta_df, img_paths, seg_paths)
    elif args.mode == 'train':
        if args.fold is None or args.run is None:
            raise ValueError("train needs --fold and --run")
        train(args, cfg, folds)
    else:
        predict(args, cfg, folds, meta_df)


if __name__ == '__main__':
    main()
