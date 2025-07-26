#!/bin/bash
export PYTHONPATH=src

datapath=/mnt/c/Users/akhil/All_my_codes/Portfolio/MPDD/anomaly_dataset_og
categories=("bracket_black")

for category in "${categories[@]}"; do
    model_results_path="results/fastflow_improved/${category}"
    evaluation_results_path="${model_results_path}/evaluation"
    checkpoint_path="${model_results_path}/fastflow_FFLOW_IMPROVED_${category}_R18_final.pth"
    
    if [ ! -f "$checkpoint_path" ]; then
        echo "Skipping $category - checkpoint not found"
        continue
    fi
    
    mkdir -p "$evaluation_results_path"
    
    python bin/run_fastflow.py "$evaluation_results_path" \
        dataset --category $category $datapath \
        fastflow --backbone resnet18 --flow_steps 8 --input_size 224 \
        --log_group FFLOW_IMPROVED_${category}_EVAL \
        --eval_only --checkpoint "$checkpoint_path" \
        --save_segmentation_images
        
done 