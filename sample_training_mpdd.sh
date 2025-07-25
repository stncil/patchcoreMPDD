#!/bin/bash
export PYTHONPATH=src
# Sample training script for MPDD (Metal Parts Defect Detection Dataset)
# Make sure to update the datapath and datasets variables according to your MPDD setup

datapath=/mnt/c/Users/akhil/All_my_codes/Portfolio/MPDD/anomaly_dataset_og

# Update these according to your actual MPDD categories
# These are example metal part categories - replace with actual ones from your dataset
datasets=('metal_plate' 'bracket_white' 'bracket_brown' 'bracket_black')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

############# MPDD Detection Examples

### IM224 (224x224 images):
# Baseline: Backbone: WR50, Blocks: 2 & 3, Coreset Percentage: 10%, Embedding Dimensionalities: 1024 > 1024
echo "Training PatchCore on MPDD with WideResNet50 backbone..."
python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --save_segmentation_images \
--log_group MPDD_IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MPDD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 \
--pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mpdd $datapath

# Ensemble: Multiple backbones for better performance
echo "Training PatchCore ensemble on MPDD..."
python bin/run_patchcore.py --gpu 0 --seed 3 --save_patchcore_model --save_segmentation_images \
--log_group MPDD_IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1_S3 --log_project MPDD_Results results \
patch_core -b wideresnet101 -b resnext101 -b densenet201 \
-le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 \
--pretrain_embed_dimension 1024 --target_embed_dimension 384 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.01 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mpdd $datapath

### IM320 (320x320 images - for higher resolution):
# Baseline with higher resolution for detailed metal part inspection
echo "Training PatchCore on MPDD with higher resolution (320x320)..."
python bin/run_patchcore.py --gpu 0 --seed 22 --save_patchcore_model --save_segmentation_images \
--log_group MPDD_IM320_WR50_L2-3_P001_D1024-1024_PS-3_AN-1_S22 --log_project MPDD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 \
--pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mpdd $datapath

############# MPDD Segmentation (for pixel-level defect localization)
### IM320 with parameters optimized for segmentation
echo "Training PatchCore for segmentation on MPDD..."
python bin/run_patchcore.py --gpu 0 --seed 39 --save_patchcore_model --save_segmentation_images \
--log_group MPDD_IM320_WR50_L2-3_P001_D1024-1024_PS-5_AN-3_S39 --log_project MPDD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 \
--pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 3 --patchsize 5 \
sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mpdd $datapath

echo "MPDD training completed!"
echo "Models will be saved in the results directory."
echo ""
echo "To evaluate the trained models, use:"
echo "bash sample_evaluation_mpdd.sh" 