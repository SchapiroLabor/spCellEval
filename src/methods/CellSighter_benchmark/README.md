# Cell_Sighter Benchmarking


# Cell_Sighter: Cell Classification Pipeline (WSL2-Ready)

A reproducible pipeline for preprocessing, training, validating, and analyzing cell classification data, adapted from the original CellSighter framework https://github.com/KerenLab/CellSighter/ and https://github.com/mahmoodlab/MAPS/ .

---

## Environment Setup

### 1. Clone the Repo

```bash
git clone https://github.com/SchapiroLabor/Marcel_Fastner_2025_Cell_Sighter.git
cd path/to/your/folder/Cell_Sighter
```

Additionally, Cell_Sighter.sh needs to be converted into unix format, so it can be executed from WSL terminal.

```bash
sudo apt install dos2unix  # if not installed
dos2unix Cell_Sighter.sh
```

### 2. Create Anaconda Environment

Anaconda3 or Miniconda3 need to be installed.

Afterwards create and activate the environment manually, if you want to execute any of the scripts outside of the provided pipeline (Cell_Sighter.sh script).
```bash
conda env create -f requirements.yml
conda activate Cell_Sighter
```

This uses Python 3.9 and includes PyTorch 1.13.1 with CUDA 11.7.

---

## Usage

Edit the hardcoded parameters, such as path to the datasets in Cell_Sighter.sh and execute the pipeline from WSL2 terminal:

```bash
bash Cell_Sighter.sh --datasets cHL_2_MIBI IMMUcan --folds fold_0 fold_1 ... fold_4 --cell_type_col level_1_cell_type
```

### Cell_Sighter Flags

| Flag | Type | Description | Required | Default |
|------|------|-------------|----------|---------|
| `--datasets` | `list of str` | list of datasets | required | empty list |
| `--cell_type_col` | `str` | Column of cell type level | required | "" |
| `--folds` | `list of str` | List of folds for which training.py, validation.py and confusion.py will be run) | required | empty list |
| `--skip_preprocessing` | `flag` | Skip preprocessing step | optional | false |
| `--skip_training` | `flag` | Skip training step | optional | false |
| `--skip_validation` | `flag` | Skip validation step | optional | false |
| `--confusion_matrix` | `flag` | Include matrix generation | optional | false |

Example (skip preprocessing):

```bash
bash Cell_Sighter.sh --datasets cHL_2_MIBI --skip_preprocessing --folds fold_0 --cell_type_col level_1_cell_type
```

---

## Expected Directory Structure of the raw data (before preprocessing)

```bash
test_data/
└── cHL_2_MIBI/
|    ├── raw_images/multistack_tiffs/
|    |   ├── 1_stacked.ome.tif
|    |   ├── .
|    |   ├── .
|    |   ├── .
|    |   └── 6_stacked.ome.tif
|    ├── segmentation/
|    |   ├── 1/H3_memSUM_noCD163_deepcell060_AutoHist_mpp1.75/segmentationMap.tif
|    │   ├── .
|    |   ├── .
|    |   ├── .
|    |   └── 6/H3_memSUM_noCD163_deepcell060_AutoHist_mpp1.75/segmentationMap.tif
|    ├── quantification/processed/
|    |   ├── cHL_2_MIBI_quantification.csv
|    └── markers.txt
└── IMMUcan/
     ├── raw_images/multistack_tiffs/
     |   ├── xyz_stacked.tif
     |   ├── .
     |   ├── .
     |   ├── .
     |   └── ijk_stacked.tif
     ├── segmentation/
     |   ├── xyz_stacked.tif
     |   ├── .
     |   ├── .
     |   ├── .
     |   └── ijk_stacked.tif
     ├── quantification/processed/
     |   ├── IMMUcan_quantification.csv
     └── markers.txt
```
##  Core Scripts Overview (preprocessing, training, validation, confusion_matrix)

### `data_pre_processing.py`

The preprocessing can be done in isolation or with the pipeline (Cell_Sighter.sh) using skip flags as described below.
On one side, it takes tif, tiff or npz files as input for the raw images (multistack_tiffs) and masks, transposes the raw images if needed and saves both, raw images and masks as npz files.
On the other side, it uses a <dataset>_quantification.csv to map integer labels to each sample_id, cell_id and cell_type respectively. Missing cell_ids are filled in, mapped to -1, and excluded cell_types are mapped to -1.
In addition, the label_map is saved as .csv, and the markers.txt is copied as channels.txt. Furthermore, the samples are either split into folds of training/validation data and saved in a folds.json or a folds.json can be provided. Lastly, a config.json file is created and saved.

```bash
python data_pre_processing.py --dataset_name cHL_2_MIBI --data_root <root_dir> --working_dir <working_dir> --cell_type_col level_1_cell_type level_2_cell_type cell_type --transpose --split 0.7 --sample_batch --to_pad --aug_data
or
bash Cell_Sighter.sh --datasets cHL_2_MIBI --skip_training --skip_validation --folds fold_0 --cell_type_col level_1_cell_type
# If you only want to use the preprocessing from Cell_Sighter.sh you still need to provide atleast one fold and cell_type_col as flags, or the script will not work.
```
#### `data_pre_processing.py Flags`

| Flag | Type | Description | Required | default in .py script | hardcoded in Cell_Sighter.sh |
|------|------|-------------|----------|-----------------------|------------------------------|
| `--dataset_name` | `str` | Name of the dataset | required | --- | --- |
| `--data_root` | `str` | path/to/the/dir/of/test_data/<dataset> | required | --- | path/to/the/dir/of/test_data/<dataset> |
| `--working_dir` | `str` | path/to/the/folder/Cell_Sighter | required | --- | path/to/the/dir/of/test_data/<dataset> |
| `--cell_type_col` | `list of str` | Column with cell types in quant CSV (levels) | required | empty list | level_1_cell_type level_2_cell_type cell_type |
| `--transpose` | `flag` | If set, transposes input images | required, if image shape is CxXxY instead of XxYxC | false | true |
| `--exclude_celltypes` | `list of str` | List of cell types to exclude | optional | empty list | --- |
| `--num_folds` | `int` | Number of folds for KFold split | optional if folds.json is provided | 5 | --- |
| `--split` | `float` | Train/test split ratio | optional if folds.json is provided | None | 0.7 |
| `--crop_input_size` | `int` | Cell-centered input crop size | optional | 60 | --- |
| `--crop_size` | `int` | Final crop size | optional | 128 | --- |
| `--epoch_max` | `int` | Max training epochs | optional | 100 | --- |
| `--lr` | `float` | Learning rate | optional | 0.001 | --- |
| `--sample_batch` | `flag` | Whether to sample equally from the category in each batch during training | optional | false | true |
| `--to_pad` | `flag` | Whether to work on the border of the image or not | optional | false | true |
| `--aug_data` | `flag` | Enable data augmentation | optional | false | true |
| `--folds_json` | `str` | Path to folds.json | (optional if --split used) | "" | --- |
---

#### `data_pre_processing.py output`

After running the preprocessing, the following data structure is obtained:

```
Cell_Sighter/
└── datasets/
    └── <name_of_dataset>/
        ├── CellTypes/
        │   ├── cells/<npz_files_of_masks>
        │   ├── cells2labels/<column names of cell_type_levels>/<text_files_with_labels_list>
        │   └── data/images/<npz_files_of_raw_images>
        ├── config.json
        ├── folds.json
        ├── channels.txt
        └── <labels_files>.csv

```

---

##  Training

Once the data has been preprocessed as described, models can be trained.
This can be done as part of the pipeline or in isolation:

```bash
python training.py --dataset cHL_2_MIBI --fold_id fold_0 --time --cell_type_col level_1_cell_type
or
bash Cell_Sighter.sh --datasets cHL_2_MIBI --skip_preprocessing --skip_validation --folds fold_0 fold_1 ... fold_4 --cell_type_col level_1_cell_type
```
If you execute training.py, you can do it only for one fold_id at a time.

#### `training.py Flags`

| Flag | Type | Description | Required | default in .py script | hardcoded in Cell_Sighter.sh |
|------|------|-------------|----------|-----------------------|------------------------------|
| `--dataset` | `str` | Name of the dataset | required | --- | "$dataset" of a loop |
| `--fold_id` | `str` | Fold to train on (e.g., fold_0) | required | fold_0 | "$FOLD" of a loop |
| `--batch_size` | `int` | Batch size for training | optional | 256 | --- |
| `--num_workers` | `int` | Number of workers | optional | 8 |  |
| `--epoch_min` | `int` | Minimum epochs before early stop | optional | 50 | --- |
| `--n_no_change` | `int` | Patience for early stopping | optional | 20 | --- |
| `--cell_type_col` | `str` | Column of celltype level | required | cell_type (level3)| "level_1_cell_type" |
| `--time` | `flag` | Whether to create time.txt with timestamps for each fold | optional | false | true |

The trained models are saved as "weights.pth".
If --time is used, time.txt will be created under /Cell_Sighter/results/<dataset>/<cell_type_col>/ listing the duration for the training of each fold of the respective dataset and cell_type_col.

### `validation.py`

After training models, the validation can be executed as part of the pipeline or in isolation:

```bash
python validation.py --dataset cHL_2_MIBI --fold_id fold_0 --cell_type_col level_1_cell_type --append --data_root <path/to/the/dir/of/test_data/<dataset>>
or
bash Cell_Sighter.sh --datasets cHL_2_MIBI --skip_preprocessing --skip_training --folds fold_0 fold_1 ... fold_4 --cell_type_col level_1_cell_type
```
If you execute validation.py, you can do it only for one fold_id at a time.

#### `validation.py Flags`

| Flag | Type | Description | Required | default in .py script | hardcoded in Cell_Sighter.sh |
|------|------|-------------|----------|-----------------------|------------------------------|
| `--dataset` | `str` | Name of the dataset | required | --- | "$dataset" of a loop |
| `--fold_id` | `str` | Fold to train on (e.g., fold_0) | required | fold_0 | "$FOLD" of a loop |
| `--batch_size` | `int` | Batch size for training | optional | 256 | --- |
| `--num_workers` | `int` | Number of workers | optional | 8 | 8 |
| `--output_path` | `str` | Alternative output path for the predictions_{fold_id}.csv's (alternative results folder)| optional | "" | /path/to/your/results_folder/results |
| `--columns` | `list of str` | List of columns from the predictions_{fold_id}.csv's which will be extracted and saved additionally | optional | None | --- |
| `--cell_type_col` | `str` | Column of celltype level | required | cell_type (level3)| "level_1_cell_type" |
| `--append` | `flag` | Whether to use the respective quantification table as results file with replaced cell_type_col as "true_phenotype" and appended "predicted_phenotype" | optional | false | true |
| `--data_root` | `str` | path/to/the/dir/of/test_data/<dataset> | required if --append is used | --- | /path/to/your/folder/containing/the/${dataset} |

As output, the validation generates one predictions_{fold_id}.csv inlcuding columns with sample_id, cell_id, predicted_phenotypes and true_phenotypes per fold_id. In addition, predictions_{fold_id}_selected.csv will be created, which contains only the columns specified with the --columns flag. If --append is used, the initial quantification table is appended with the column "predicted_phenotypes", the column <cell_type_col> is replaced by "true_phenotypes" and the file is analogously saved under the respective output directory.

---

### `confusion_matrix.py`

Once the validation has been done, the predictions_{fold_id}.csv can be evaluated via generating a confusion matrix:

```bash
python analyze_results/confusion_matrix.py --dataset cHL_2_MIBI --fold_id fold_0 --cell_type_col level_1_cell_type
or
bash Cell_Sighter.sh --datasets cHL_2_MIBI --skip_preprocessing --skip_training --skip_validation --confusion_matrix --folds fold_0 fold_1 ... fold_4 --cell_type_col level_1_cell_type
```

| Flag | Type | Description | Required | default in .py script | hardcoded in Cell_Sighter.sh |
|------|------|-------------|----------|-----------------------|------------------------------|
| `--dataset` | `str` | Name of the dataset | required | --- | "$dataset" of a loop |
| `--fold_id` | `str` | Fold to train on (e.g., fold_0) | required | fold_0 | "$FOLD" of a loop |
| `--input_path` | `str` | Alternative inputput path for the predictions_{fold_id}.csv's, which needs to be identical to --output_path of the validation (alternative results folder) | depends on the validation output_path | "" | /path/to/your/results_folder/results |
| `--cell_type_col` | `str` | Column of celltype level | required | cell_type (level3)| "level_1_cell_type" |

## Output Directory Structure

After running the pipeline, the results-output (inside your working directory) is organized as follows (if default output paths were used):

```
Cell_Sighter/
├── results/
│   └── <dataset>/
|       ├── <cell_type_col>/
│       │   ├── fold_0/
│       │   │   ├── weights.pth
│       │   │   ├── results.csv
│       │   │   ├── results_selected.csv
│       │   │   └── confusion_matrix.png
```

If an alternative results folder is provided (validation.py --output_path "path/to/a/results/folder/results"), the results-output (of validation.py and confusion.py) is organized as follows:

```
results/
├── <dataset1>/
│   └── Cell_Sighter/
│       ├── level1/
│       │   ├── predictions_fold_0.csv
│       │   ├── predictions_fold_0_selected.csv
│       │   ├── confusion_matrix_fold_0.png
│       │   ├── predictions_fold_1.csv
│       │   ├── predictions_fold_1_selected.csv
│       │   └── confusion_matrix_fold_1.png
│       ├── level2/
│       └── level3/
└── <dataset2>/
```

## Additional options

After running the preprocessing, the config.json file can easily be modified manually. This yields the option of defining a "blacklist", as list of channels, which will we excluded from training and validation.
Furthermore, all relevant flags of each core script can be edited manually in the bash script Cell_Sighter.sh.
The dictionary to map cell_type_columns to their respective level can be changed inside validation.py and confusion.py:

```
    # get Cell_type_level from dic

    cell_type_levels = {
    "level_1_cell_type": "level1",
    "level_2_cell_type": "level2",
    "cell_type": "level3"
}
    level = cell_type_levels.get(args.cell_type_col)
```


##  Compatibility

-  **WSL2**: Fully compatible
-  **Python 3.9**: All scripts compatible
-  **CUDA 11.7**: Optional (CPU training also supported)
-  **No GUI dependencies**: No need for X11 in WSL2

---

## Acknowledgements

Originally developed by ([https://github.com/KerenLab/CellSighter/]) and https://github.com/mahmoodlab/MAPS/. This fork adds support for automation, WSL2, and multiple datasets.

---

## License

Same as the original CellSighter repository.
