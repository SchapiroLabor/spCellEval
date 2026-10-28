import os
import re
import argparse

# Relabels train_time/training_time as prediction_time in <method>/level3/fold_times.txt of the clustering
# method folders in a dataset results folder. Without --apply it only prints what would change.

parser = argparse.ArgumentParser(description='Relabel train_time as prediction_time in fold_times.txt of clustering methods.')
parser.add_argument('-i', '--input', dest='input_path', required=True, help='Dataset results folder containing the method folders (e.g. results/cHL_2_MIBI)')
parser.add_argument('--apply', action='store_true', help='Write the changes (default: only print them)')
args = parser.parse_args()

# Method folder prefixes of the clustering methods; foundation model folders (eva_leiden, kronos_leiden, ...) do not match
CLUSTERING_PREFIXES = ('flowsom', 'FuseSOM_', 'leiden_res', 'nimbus_leiden_res', 'phenograph_')
TRAIN_LABEL = re.compile(r'\b(train|training)_time\b')

for method in sorted(os.listdir(args.input_path)):
    if not method.startswith(CLUSTERING_PREFIXES):
        continue
    path = os.path.join(args.input_path, method, 'level3', 'fold_times.txt')
    if not os.path.exists(path):
        print(f'{method}: no level3/fold_times.txt')
        continue
    with open(path) as f:
        text = f.read()
    new_text, n = TRAIN_LABEL.subn('prediction_time', text)
    print(f'{method}: {n} lines relabeled')
    if args.apply and n > 0:
        with open(path, 'w') as f:
            f.write(new_text)

#example usage: python3 relabel_clustering_fold_times.py -i ../../../results/cHL_2_MIBI          (dry run)
#               python3 relabel_clustering_fold_times.py -i ../../../results/cHL_2_MIBI --apply
