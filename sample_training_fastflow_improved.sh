#!/bin/bash
export PYTHONPATH=src

datapath=/mnt/c/Users/akhil/All_my_codes/Portfolio/MPDD/anomaly_dataset_og
categories=("bracket_black")

echo "=== Improved FastFlow Training with Class-Specific Parameters ==="
echo "Data path: $datapath"
echo "Categories: ${categories[@]}"
echo ""

for category in "${categories[@]}"; do
    echo "Training FastFlow on MPDD category: $category (with optimized parameters)"
    
    category_results="results/fastflow_improved/${category}"
    mkdir -p "$category_results"
    
    python bin/run_fastflow.py \
        --log_group FFLOW_IMPROVED_${category}_R18 \
        --save_model \
        --save_segmentation_images \
        "$category_results" \
        dataset --category $category $datapath \
        fastflow --backbone resnet18 --use_class_specific 
    
    echo "Finished training FastFlow for $category"
    echo "Results saved in: $category_results"
    echo "----------------------------------------"
done

echo ""
echo "=== All Improved FastFlow Trainings Complete! ==="
echo ""
