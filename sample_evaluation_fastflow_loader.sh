#!/bin/bash
export PYTHONPATH=src

datapath=/mnt/c/Users/akhil/All_my_codes/Portfolio/MPDD/anomaly_dataset_og
loadpath=results/fastflow_improved
base_savefolder=evaluated_results/fastflow_results

# Categories that were trained
categories=("metal_plate" "bracket_white" "bracket_brown" "bracket_black")

echo "Evaluating FastFlow models..."
echo "Loading models from: $loadpath"
echo "Creating evaluation folders for each class..."

# Evaluate each category separately
for category in "${categories[@]}"; do
    echo ""
    echo "Evaluating FastFlow model for: $category"
    
    # Create class-specific evaluation folder
    class_savefolder="${base_savefolder}/${category}"
    mkdir -p "$class_savefolder"
    
    # Define model path
    model_path="${loadpath}/${category}/fastflow_FFLOW_IMPROVED_${category}_R18_final.pth"
    
    # Check if model exists
    if [ ! -f "$model_path" ]; then
        echo "Skipping $category - model not found at $model_path"
        continue
    fi
    
    echo "Saving results to: $class_savefolder"
    
    # Evaluate the specific category
    python bin/load_and_evaluate_fastflow.py --gpu 0 --seed 0 --save_segmentation_images "$class_savefolder" \
    fastflow_loader -p "$model_path" \
    dataset --resize 256 --imagesize 224 -d "$category" mpdd "$datapath"
    
    echo "Completed evaluation for: $category"
done

echo ""
echo "FastFlow evaluation completed for all categories!"
echo ""
echo "Results organized by class in:"
for category in "${categories[@]}"; do
    echo "  - ${base_savefolder}/${category}/"
done
echo ""
echo "Each folder contains:"
echo "  - results.csv (performance metrics)"
echo "  - segmentation_images/ (visual analysis)"
echo "  - Individual files for each test image with anomaly scores" 