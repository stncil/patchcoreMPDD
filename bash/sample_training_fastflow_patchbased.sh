#!/bin/bash

# FastFlow Patch-Based Training Script for MPDD Dataset
# Usage: bash sample_training_fastflow_patchbased.sh

# Set Python path
export PYTHONPATH="${PYTHONPATH}:src"

# Training parameters
DATASET_PATH="/mnt/c/Users/akhil/All_my_codes/Portfolio/MPDD/anomaly_dataset_og"
RESULTS_PATH="./results/fastflow_patchbased"
GPU_ID=0
SEED=42
LOG_GROUP="fastflow_patchbased"

# Categories to train on
CATEGORIES=("bracket_black")

for CATEGORY in "${CATEGORIES[@]}"; do
    echo "Training patch-based FastFlow for category: $CATEGORY"
    
    python bin/run_fastflow_patchbased.py \
        --gpu $GPU_ID \
        --seed $SEED \
        --log_group "${LOG_GROUP}_${CATEGORY}" \
        --save_model \
        --save_segmentation_images \
        "$RESULTS_PATH" \
        dataset --category "$CATEGORY" \
        --batch_size 1 \
        --num_workers 0 \
        --resize 256 \
        --imagesize 224 \
        "$DATASET_PATH" \
        fastflow_patch \
        --backbone "resnet18" \
        --flow_steps 2 \
        --input_size 224 \
        --hidden_ratio 0.25 \
        --use_class_specific \
        --patchsize 3 \
        --patchstride 1
    
    echo "Completed training for $CATEGORY"
    echo "----------------------------------------"
done

echo "All patch-based FastFlow training completed!"
echo "Results saved in: $RESULTS_PATH" 