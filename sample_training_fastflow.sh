#!/bin/bash
export PYTHONPATH=src
# Sample training script for FastFlow (MPDD dataset)
# Update datapath and categories as needed

datapath=/mnt/c/Users/akhil/All_my_codes/Portfolio/MPDD/anomaly_dataset_og
categories=("metal_plate" "bracket_white" "bracket_brown" "bracket_black")

for category in "${categories[@]}"; do
    echo "Training FastFlow on MPDD category: $category"
    python bin/run_fastflow.py \
        --log_group FFLOW_${category}_R18 \
        --save_model \
        --save_segmentation_images \
        results/fastflow_results \
        dataset --category $category $datapath \
        fastflow --backbone resnet18 --flow_steps 8 --input_size 224
    echo "Finished training FastFlow for $category"
done

echo "All FastFlow trainings complete!"