import os
import re
import sys
import shutil
import argparse
from collections import Counter

# One-off migration: renames the method-specific cluster column in existing clustering predictions to 'cluster_label'
# (CellLENS 'Cluster_Labels', starling 'st_label', Leiden 'leiden_res<r>', 'phenograph_<k>', 'FuseSOM_<n>',
# 'flowsom', 'flowsom_meta_clusters', incl. the Nimbus and subclustering runs).
# Only predictions inside clustering method folders are touched (see CLUSTERING_FOLDER), the column is found from the header.
# Only the header line is rewritten, the rest of the file is copied byte by byte (R quoting is preserved).

OLD_COLUMN = re.compile(r'Cluster_Labels|st_label|leiden_res[\d_]+|phenograph_\d+|FuseSOM_\d+|flowsom|flowsom_meta_clusters')
NEW_COLUMN = 'cluster_label'
# Method folders of the clustering methods, e.g. 'CellLENS_Lite_subclust2', 'celllens_full', 'FuseSOM_12_subclust1',
# 'flowsom_meta_clusters', 'leiden_res1_0', 'phenograph_30', 'starling', 'nimbus_FuseSOM', 'nimbus_leiden_0_8'
CLUSTERING_FOLDER = re.compile(r'(nimbus_)?(celllens_.+|fusesom.*|flowsom.*|leiden_.+|phenograph_.+|starling.*)', re.IGNORECASE)

parser = argparse.ArgumentParser(description="Rename the cluster column of all clustering predictions_*.csv to 'cluster_label'.")
parser.add_argument('results', help='Results folder (or a single dataset folder), searched recursively for predictions_*.csv in clustering method folders')
parser.add_argument('-n', '--dry_run', action='store_true', help='Only report what would be renamed')
args = parser.parse_args()


def header_fields(line):
    return [f.strip('"') for f in line.rstrip('\r\n').split(',')]


counts = Counter()
skipped_dirs = set()
for root, _, files in os.walk(args.results):
    for name in sorted(files):
        if not re.fullmatch(r'predictions_\d+\.csv', name):
            continue
        path = os.path.join(root, name)
        if not any(CLUSTERING_FOLDER.fullmatch(d) for d in os.path.relpath(root, args.results).split(os.sep)):
            continue
        with open(path, newline='') as f:
            header = f.readline()
        fields = header_fields(header)
        old = [c for c in fields if OLD_COLUMN.fullmatch(c)]
        if NEW_COLUMN in fields:
            counts['already harmonized'] += 1
            continue
        if len(old) != 1:
            if old:
                print(f'WARNING: several cluster columns {old}, skipped: {path}', file=sys.stderr)
            counts['no (unique) cluster column'] += 1
            skipped_dirs.add(os.path.dirname(root))
            continue
        counts[f'{old[0]} -> {NEW_COLUMN}'] += 1
        if args.dry_run:
            continue
        # Replace the field with and without R's quotes, keep everything else in the header as is
        new_header = re.sub(rf'(^|,)("?){re.escape(old[0])}\2(?=,|\r?\n|$)', rf'\1\2{NEW_COLUMN}\2', header)
        tmp = path + '.tmp'
        with open(path, 'rb') as src, open(tmp, 'wb') as dst:
            src.readline()
            dst.write(new_header.encode())
            shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
        os.replace(tmp, path)

for key, n in sorted(counts.items()):
    print(f'{n:6d}  {key}')
print('\nClustering method folders with files without a known cluster column (check these by hand):')
for d in sorted(skipped_dirs):
    print(f'  {d}')

#example usage (dry run first): python3 rename_cluster_label_column.py /path/to/results -n
