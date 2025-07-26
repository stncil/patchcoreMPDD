#!/bin/bash
export PYTHONPATH=src

# Sample evaluation script for MPDD (Metal Parts Defect Detection Dataset)
# Make sure to update the datapath, loadpath, and datasets variables according to your setup

datapath=/mnt/c/Users/akhil/All_my_codes/Portfolio/MPDD/anomaly_dataset_og

loadpath=/mnt/c/Users/akhil/All_my_codes/Portfolio/patchcore-inspection/results/MPDD_Results
modelfolder=MPDD_IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0
savefolder=evaluated_results'/'$modelfolder

# Update these according to your actual MPDD categories
# These should match the categories you used during training
datasets=('metal_plate' 'bracket_white' 'bracket_brown' 'bracket_black')

# Create the dataset and model flags
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mpdd_'$dataset; done))

echo "Evaluating MPDD PatchCore models..."
echo "Loading models from: $loadpath/$modelfolder"
echo "Saving results to: $savefolder"

# Evaluate the trained PatchCore models
python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 --save_segmentation_images $savefolder \
patch_core_loader "${model_flags[@]}" \
dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mpdd $datapath

echo ""
echo "MPDD evaluation completed!"
echo "Results saved in: $savefolder"
echo ""
echo "Check the results.csv file for detailed performance metrics including:"
echo "  - Image-level AUROC (anomaly detection performance)"
echo "  - Pixel-level AUROC (segmentation performance)"
echo "  - PRO score (region-based evaluation)"

# Optional: For higher resolution evaluation (if you trained with IM320)
echo ""
echo "For higher resolution evaluation (320x320), use:"
echo "python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 \$savefolder \\"
echo "patch_core_loader \"\${model_flags[@]}\" --faiss_on_gpu \\"
echo "dataset --resize 366 --imagesize 320 \"\${dataset_flags[@]}\" mpdd \$datapath" 