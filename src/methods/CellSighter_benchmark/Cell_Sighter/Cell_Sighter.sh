#!/bin/bash

# Usage:
# bash Cell_Sighter.sh --datasets cHL_2_MIBI IMMUcan --skip_preprocessing

set -e

# Default skip flags
skip_preprocessing=false
skip_training=false
skip_validation=false
confusion_matrix=false
datasets=()
folds=()
cell_type_col=""

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --datasets)
            shift
            while [[ "$#" -gt 0 && "$1" != --* ]]; do
                datasets+=("$1")
                shift
            done
            ;;
        --folds)
            shift
            while [[ "$#" -gt 0 && "$1" != --* ]]; do
                folds+=("$1")
                shift
            done
            ;;
        --cell_type_col)
            shift
            cell_type_col="$1"
            shift
            ;;
        --skip_preprocessing) skip_preprocessing=true; shift ;;
        --skip_training) skip_training=true; shift ;;
        --skip_validation) skip_validation=true; shift ;;
        --confusion_matrix) confusion_matrix=true; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

if [[ ${#datasets[@]} -eq 0 || ${#folds[@]} -eq 0 || -z "$cell_type_col" ]]; then
    echo "Usage: $0 --datasets <dataset1> <dataset2> ... --folds <fold_0> <fold_1> ... --cell_type_col <column> [--skip_preprocessing] [--skip_training] [--skip_validation] [--confusion_matrix]"
    exit 1
fi

# Setup Conda
if ! conda info --envs | grep -q '^Cell_Sighter'; then
    echo "Creating Conda environment from requirements.yml..."
    conda env create --file requirements.yml
else
    echo "Conda environment 'Cell_Sighter' already exists."
fi

echo "Activating Conda environment 'Cell_Sighter'..."

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "Error: Neither Anaconda nor Miniconda found in expected locations."
    exit 1
fi

conda activate Cell_Sighter

# Main loop
for dataset in "${datasets[@]}"; do
    echo -e "\n========================================="
    echo " Processing dataset: $dataset"
    echo "========================================="

    if [ "$skip_preprocessing" = false ]; then
        echo "Running data preprocessing for $dataset..."
        python data_pre_processing.py \
            --dataset_name "$dataset" \
            --data_root <"/path/to/your/folder/containing/the/${dataset}"> \
            --working_dir <"path/to/your/Cell_Sighter/folder"> \
            --cell_type_col level_1_cell_type level_2_cell_type cell_type \
            --transpose \
            --split 0.7 \
            --sample_batch \
            --to_pad \
            --aug_data
            # --folds_json <"if/existent/path/to/your/folds.json"> \
            # --crop_input_size 60 \
            # --crop_size 128 \
            # --epoch_max 100 \
            # --lr 0.001 \
            # --exclude_celltypes \
            # --num_folds 5 \

    else
        echo " Skipping preprocessing for $dataset"
    fi

    for FOLD in "${folds[@]}"; do
        if [ "$skip_training" = false ]; then
            echo " Training $dataset $FOLD"
            start_time=$(date +%s)

            python training.py \
                --dataset "$dataset" \
                --fold_id "$FOLD" \
                --num_workers 8 \
                --cell_type_col "$cell_type_col" \
                --time
                # --batch_size 256 \
                # --epoch_min 50 \
                # --n_no_change 20 \

            end_time=$(date +%s)
            elapsed=$((end_time - start_time))
            printf "Finished training: %02dh:%02dm:%02ds\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
            echo ""
        else
            echo " Skipping training for $dataset $FOLD"
        fi

        if [ "$skip_validation" = false ]; then
            echo " Validating $dataset $FOLD"

            python validation.py \
                --dataset "$dataset" \
                --fold_id "$FOLD" \
                --num_workers 8 \
                --cell_type_col "$cell_type_col" \
                --output_path <"/path/to/your/results_folder/results"> \
                --append \
                --data_root <"/path/to/your/folder/containing/the/${dataset}">
                # --columns cell_id sample_id true_phenotypes predicted_phenotypes \
                # --batch_size 256 \

            echo " Validation complete for $dataset $FOLD"
        else
            echo " Skipping validation for $dataset $FOLD"
        fi
    done

    if [ "$confusion_matrix" = true ]; then
        echo " Generating confusion matrices for $dataset..."
        for FOLD in "${folds[@]}"; do
            python analyze_results/confusion_matrix.py \
                --dataset "$dataset" \
                --fold_id "$FOLD" \
                --input_path <"/path/to/your/results_folder/results"> \
                --cell_type_col "$cell_type_col"

        done
    else
        echo " Skipping confusion matrix generation for $dataset"
    fi

    echo " All selected steps done for $dataset"
done

echo " All datasets completed!"
