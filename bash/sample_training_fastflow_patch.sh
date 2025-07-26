#!/bin/bash

# FastFlow Training Script for MPDD Dataset with Patch-based Inference
# Usage: bash sample_training_fastflow_patch.sh

# Set Python path
export PYTHONPATH="${PYTHONPATH}:src"

# Training parameters
DATASET_PATH="/mnt/c/Users/akhil/All_my_codes/Portfolio/MPDD/anomaly_dataset_og" 
RESULTS_PATH="./results/fastflow_patch"
GPU_ID=0
SEED=42
LOG_GROUP="fastflow_patch"

# Categories to train on
CATEGORIES=("bracket_white")

for CATEGORY in "${CATEGORIES[@]}"; do
    echo "Training FastFlow for category: $CATEGORY"
    
    python bin/run_fastflow.py \
        --gpu $GPU_ID \
        --seed $SEED \
        --log_group "${LOG_GROUP}_${CATEGORY}" \
        --save_model \
        "$RESULTS_PATH" \
        dataset --category "$CATEGORY" \
        --batch_size 10 \
        --num_workers 0 \
        --resize 256 \
        --imagesize 224 \
        --preserve_high_res \
        "$DATASET_PATH" \
        fastflow \
        --backbone "resnet18" \
        --flow_steps 4 \
        --input_size 224 \
        --hidden_ratio 0.5 \
        --use_class_specific \
        --enable_patch_inference \
        --patch_size 160 \
        --patch_stride 80 \
        --low_memory
    
    echo "Completed training for $CATEGORY"
    echo "----------------------------------------"
done

echo "All training completed!" 