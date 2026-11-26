"""
fold_timer.py — Run times of all benchmark methods as <method>/level3/fold_times.csv (standard library only,
so every method environment can import it; the R methods use write_fold_times() in clustering_utils.R).

    fold,time_s
    shared,3912.40     <- everything outside the folds: data loading, one-off steps all folds reuse
    0,28.74               (e.g. embedding every cell), writing outputs
    ...
    total,4060.88      <- shared + the folds

Package imports are not timed: START = time.perf_counter() right after the script's imports.
"""

import csv
import time
from contextlib import contextmanager
from pathlib import Path


class FoldTimer:
    """Wall-clock run time of a method, written to <out_dir>/fold_times.csv (fold, time_s).

    start: time.perf_counter() taken right after the script's package imports (not timed).
    Each fold is timed with `with timer.fold(i):`, or, inside an existing loop, by setting
    timer.folds[i] = seconds. Everything else from start to write() is the 'shared' row: data loading,
    one-off steps all folds reuse (e.g. embedding every cell) and writing outputs. 'total' = shared + the
    folds. Steps skipped because of a cache are not timed, so timing runs start from an empty output directory.
    """

    def __init__(self, start):
        self.start, self.folds = start, {}

    @contextmanager
    def fold(self, i):
        t = time.perf_counter()
        yield
        self.folds[i] = time.perf_counter() - t

    def write(self, out_dir):
        total = time.perf_counter() - self.start
        rows = [('shared', total - sum(self.folds.values())), *sorted(self.folds.items()), ('total', total)]
        with open(Path(out_dir) / 'fold_times.csv', 'w', newline='') as f:
            csv.writer(f, lineterminator='\n').writerows([('fold', 'time_s'), *((k, f'{v:.2f}') for k, v in rows)])


def read_fold_times(path):
    """fold_times.csv as {'shared': s, <fold>: s, ..., 'total': s}, folds as int; None if the file is missing."""
    if not Path(path).exists():
        return None
    with open(path, newline='') as f:
        return {int(r['fold']) if r['fold'].isdigit() else r['fold']: float(r['time_s']) for r in csv.DictReader(f)}
