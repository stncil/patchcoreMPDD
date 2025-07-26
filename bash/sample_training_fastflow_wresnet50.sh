#!/bin/bash
export PYTHONPATH=src
# FastFlow training script optimized for WideResNet50 backbone
# WideResNet50 provides richer features, so we can use fewer epochs and simpler flows

datapath=/mnt/c/Users/akhil/All_my_codes/Portfolio/MPDD/anomaly_dataset_og
categories=("bracket_black")

echo "=== FastFlow Training with WideResNet50 Backbone ==="
echo "Data path: $datapath"
echo "Categories: ${categories[@]}"
echo "Backbone: WideResNet50 (more powerful features)"
echo ""

for category in "${categories[@]}"; do
    echo "Training FastFlow on MPDD category: $category (WideResNet50 optimized)"
    
    # Create category-specific results directory
    category_results="results/fastflow_wresnet50/${category}"
    mkdir -p "$category_results"
    
    python bin/run_fastflow.py \
        --log_group FFLOW_WR50_${category} \
        --save_model \
        --save_segmentation_images \
        "$category_results" \
        dataset --category $category $datapath \
        fastflow --backbone wide_resnet50_2 --use_class_specific
    
    echo "Finished training FastFlow for $category"
    echo "Results saved in: $category_results"
    echo "----------------------------------------"
done

echo ""
echo "=== All WideResNet50 FastFlow Trainings Complete! ==="
echo ""
echo "WideResNet50 Training Summary:"
echo "- metal_plate: lr=5e-4, epochs=30, flow_steps=6 (efficient for good performance)"
echo "- bracket_white: lr=2e-4, epochs=50, flow_steps=8 (moderate complexity)"
echo "- bracket_brown: lr=1e-4, epochs=80, flow_steps=12 (more complex for difficult class)"
echo "- bracket_black: lr=1e-4, epochs=80, flow_steps=12 (more complex for difficult class)"
echo ""
echo "Expected improvements with WideResNet50:"
echo "- Better feature representations"
echo "- Faster convergence (fewer epochs needed)"
echo "- Improved localization accuracy"
echo "- Overall +10-20% performance boost"
echo ""
echo "Check individual category results in: results/fastflow_wresnet50/" 