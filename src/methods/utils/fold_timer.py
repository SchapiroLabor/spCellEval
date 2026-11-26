"""
fold_timer.py — Run times of the benchmark methods, written to fold_times.csv (columns fold, time_s).

    START = time.perf_counter()     # right after the script's imports (imports are not timed)
    timer = FoldTimer(START)
    for i, fold in enumerate(folds):
        with timer.fold(i):         # or: t0 = time.perf_counter(); ...; timer.folds[i] = time.perf_counter() - t0
            ...
    timer.write(out_dir)

Rows: 'shared' = everything outside the folds (data loading, one-off steps all folds reuse such as
embedding every cell or a kNN graph, writing outputs), one row per fold (numbered like the script's
prediction files), 'total' = shared + the folds. Steps skipped because of a cache are not timed, so
timing runs start from an empty output directory. Only the standard library is used, so every method
environment can import it. R scripts: fold_timer.R.
"""

import csv
import time
from contextlib import contextmanager
from pathlib import Path


class FoldTimer:
    """Wall-clock times (time.perf_counter, seconds) of one method run, see the module docstring."""

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
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / 'fold_times.csv', 'w', newline='') as f:
            csv.writer(f).writerows([('fold', 'time_s'), *((fold, f'{t:.2f}') for fold, t in rows)])
