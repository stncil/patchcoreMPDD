#!/bin/bash
export PYTHONPATH=src
# Improved FastFlow training script with class-specific parameters
# This version uses optimized hyperparameters for each MPDD category

datapath=/mnt/c/Users/akhil/All_my_codes/Portfolio/MPDD/anomaly_dataset_og
categories=("bracket_white")

echo "=== Improved FastFlow Training with Class-Specific Parameters ==="
echo "Data path: $datapath"
echo "Categories: ${categories[@]}"
echo ""

for category in "${categories[@]}"; do
    echo "Training FastFlow on MPDD category: $category (with optimized parameters)"
    
    # Create category-specific results directory
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
echo "Training Summary:"
echo "- metal_plate: Standard parameters (lr=1e-3, epochs=50)"
echo "- bracket_white: Slower learning (lr=5e-4, epochs=70, more flow steps)"
echo "- bracket_brown: Much slower learning (lr=2e-4, epochs=100, complex flows)"
echo "- bracket_black: Much slower learning (lr=2e-4, epochs=100, complex flows)"
echo ""
echo "Check individual category results in: results/fastflow_improved/" 