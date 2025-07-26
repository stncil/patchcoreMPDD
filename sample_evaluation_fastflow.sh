#!/bin/bash
export PYTHONPATH=src
# Sample evaluation script for FastFlow (MPDD dataset)
# Update datapath, category, and checkpoint_path as needed

datapath=/mnt/c/Users/akhil/All_my_codes/Portfolio/MPDD/anomaly_dataset_og
category="metal_plate"
checkpoint_path=results/fastflow_improved/${category}/fastflow_FFLOW_IMPROVED_${category}_R18_final.pth

python bin/run_fastflow.py results/fastflow_eval_results \
    dataset $datapath --category $category \
    fastflow --backbone resnet18 --flow_steps 8 --input_size 224 \
    --log_group FFLOW_${category}_R18_EVAL \
    --eval_only --checkpoint $checkpoint_path

echo "FastFlow evaluation complete!" 