# MPDD Anomaly Detection: PatchCore vs FastFlow Comparison

## Summary

A comprehensive comparison of **PatchCore** and **FastFlow** methods for anomaly detection on the MPDD (Metal Parts Defect Detection) dataset, focusing on bracket and metal plate inspection.

## Dataset Overview

- **Metal Plate**: Industrial metal plates with various surface defects
- **Bracket White**: White-colored bracket components with hole and painting defects  
- **Bracket Brown**: Brown-colored bracket components with parts mismatch defects
- **Bracket Black**: Black-colored bracket components with hole defects

## Methodology and Strategy

### PatchCore Approach
- **Architecture**: Memory-based anomaly detection using pretrained CNN features
- **Training**: Few-shot learning with coreset sampling (1-10% of training data)
- **Inference**: Nearest neighbor search in feature space
- **Strengths**: 
  - Excellent pixel-level localization
  - Memory efficient with coreset sampling
  - Strong performance on complex geometric anomalies
  - Multiple variants for different use cases
- **Input Resolution**: 224×224 (IM224) or 320×320 (IM320) configurations

Detailed description in Detailed_README.md

### FastFlow Approach  
- **Architecture**: Normalizing flow-based generative model
- **Training**: Full generative modeling of normal data distribution
- **Inference**: Likelihood estimation for anomaly scoring
- **Strengths**:
  - Better image-level classification
  - Real-time inference capability
  - Probabilistic anomaly scores
- **Input Resolution**: 224×224

Detailed description in Detailed_README.md

### PatchCore Variants Analysis

#### Available PatchCore Configurations

I have implemented multiple PatchCore variants, each optimized for different scenarios:

##### 1. **Standard Single Backbone (IM224_WR50)**
- **Backbone**: WideResNet50
- **Feature Layers**: layer2 + layer3
- **Resolution**: 224×224 pixels
- **Coreset**: 10% sampling
- **Embedding**: 1024 → 1024 dimensions
- **Use Case**: General-purpose anomaly detection
- **Performance**: Balanced speed and accuracy

##### 2. **Ensemble Multi-Backbone (IM224_Ensemble)**
- **Backbones**: WideResNet101 + ResNext101 + DenseNet201
- **Feature Layers**: Multiple layer combinations
- **Resolution**: 224×224 pixels  
- **Coreset**: 1% sampling (more selective)
- **Embedding**: 1024 → 384 dimensions (compressed)
- **Use Case**: Maximum accuracy applications
- **Trade-off**: Higher computational cost, better performance

##### 3. **High-Resolution Detection (IM320_WR50)**
- **Backbone**: WideResNet50
- **Feature Layers**: layer2 + layer3
- **Resolution**: 320×320 pixels
- **Coreset**: 1% sampling
- **Embedding**: 1024 → 1024 dimensions
- **Use Case**: Detailed defect detection requiring fine-grained analysis
- **Advantage**: Better detection of small defects

##### 4. **Segmentation-Optimized (IM320_Segmentation)**
- **Backbone**: WideResNet50
- **Feature Layers**: layer2 + layer3
- **Resolution**: 320×320 pixels
- **Coreset**: 1% sampling
- **Embedding**: 1024 → 1024 dimensions
- **Patch Parameters**: patchsize=5, anomaly_scorer_num_nn=3
- **Use Case**: Precise pixel-level defect localization
- **Optimization**: Larger neighborhood for smoother segmentation

#### PatchCore Parameter Impact

| Parameter | Standard | Ensemble | High-Res | Segmentation |
|-----------|----------|----------|----------|--------------|
| **Memory Usage** | Medium | High | High | High |
| **Inference Speed** | Fast | Slow | Medium | Medium |
| **Defect Detection** | Good | Excellent | Very Good | Good |
| **Pixel Localization** | Good | Very Good | Very Good | Excellent |
| **Small Defect Sensitivity** | Medium | High | High | Very High |
| **Training Time** | Fast | Slow | Medium | Medium |

#### Backbone Comparison

**WideResNet50 (Standard)**
- **Fast inference** and training
- **Good generalization** across defect types  
- **Memory efficient**
- May miss very subtle defects

**Multi-Backbone Ensemble**
- **Highest accuracy** through complementary features
- **Robust to various defect types**
- **Best overall performance**
- **3x slower** inference and training
- **Higher memory** requirements

**Resolution Impact (224×224 vs 320×320)**
- **IM224**: Faster processing, good for real-time applications
- **IM320**: Better fine-detail detection, preferred for quality control

## Performance Comparison

I have used the following accuracy metrics:  
Instance AUROC gauges image-level defect detection accuracy  
Full Pixel AUROC reflects pixel-wise segmentation performance over all images  
Anomaly Pixel AUROC captures pixel-wise accuracy only within images that actually contain defects.

### 1. Instance-Level AUROC (Image Classification)

| Component | PatchCore (Single) | PatchCore (Ensemble) | FastFlow | Best Method |
|-----------|-------------------|---------------------|----------|-------------|
| **Metal Plate** | **1.000** | **1.000** | **1.000** | 🟡 Tie |
| **Bracket White** | 0.889 | 0.891 | **0.917** | ✅ FastFlow |
| **Bracket Brown** | **0.952** | 0.946 | **0.977** | ✅ FastFlow |
| **Bracket Black** | 0.878 | 0.856 | **0.889** | ✅ FastFlow |
| **Average** | 0.930 | 0.923 | **0.946** | ✅ FastFlow |

### 2. Full Pixel-Level AUROC (Segmentation - All Images)

| Component | PatchCore (Single) | PatchCore (Ensemble) | FastFlow | Best Method |
|-----------|-------------------|---------------------|----------|-------------|
| **Metal Plate** | **0.986** | **0.985** | **0.984** | ✅ PatchCore |
| **Bracket White** | **0.998** | **0.998** | 0.938 | ✅ PatchCore |
| **Bracket Brown** | **0.981** | **0.982** | 0.804 | ✅ PatchCore |
| **Bracket Black** | **0.982** | **0.976** | 0.825 | ✅ PatchCore |
| **Average** | **0.987** | **0.985** | 0.888 | ✅ PatchCore |

### 3. Anomaly Pixel-Level AUROC (Segmentation - Anomalous Images Only)

| Component | PatchCore (Single) | PatchCore (Ensemble) | FastFlow | Best Method |
|-----------|-------------------|---------------------|----------|-------------|
| **Metal Plate** | **0.981** | **0.979** | **0.981** | 🟡 Tie |
| **Bracket White** | **0.996** | **0.995** | 0.895 | ✅ PatchCore |
| **Bracket Brown** | **0.974** | **0.975** | 0.740 | ✅ PatchCore |
| **Bracket Black** | **0.973** | **0.966** | 0.781 | ✅ PatchCore |
| **Average** | **0.981** | **0.979** | 0.849 | ✅ PatchCore |


## Component-Specific Analysis

### Metal Plate
- **Classification**: All methods achieve perfect performance (1.0 AUROC)
- **Localization**: PatchCore slightly better (0.986 vs 0.984)
- **Conclusion**: Minimal difference - choose based on speed requirements

### Bracket Components
- **White Brackets**: 
  - FastFlow better at classification (0.917 vs 0.889)
  - PatchCore significantly better at localization (0.998 vs 0.938)
- **Brown Brackets**:
  - FastFlow better at classification (0.977 vs 0.952) 
  - PatchCore much better at localization (0.981 vs 0.804)
- **Black Brackets**:
  - Similar pattern: FastFlow for classification, PatchCore for localization


## Technical Configuration

### PatchCore Variants Setup

#### All-in-One Training Script
```bash
# Run all PatchCore variants
bash sample_training_mpdd.sh

# Evaluation - Load and evaluate trained models
bash sample_evaluation_mpdd.sh
```

#### Standard PatchCore (IM224_WR50)
```bash
# Fast, balanced performance variant
python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --save_segmentation_images --save_anomaly_scores \
--log_group MPDD_IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 results \
patch_core -b wideresnet50 -le layer2 -le layer3 \
--pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 -d metal_plate -d bracket_white -d bracket_brown -d bracket_black mpdd $datapath
```

#### Ensemble PatchCore (IM224_Ensemble)
```bash
# Maximum accuracy, multiple backbones
python bin/run_patchcore.py --gpu 0 --seed 3 --save_patchcore_model --save_segmentation_images --save_anomaly_scores \
--log_group MPDD_IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1_S3 results \
patch_core -b wideresnet101 -b resnext101 -b densenet201 \
-le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 \
--pretrain_embed_dimension 1024 --target_embed_dimension 384 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.01 approx_greedy_coreset dataset --resize 256 --imagesize 224 -d metal_plate -d bracket_white -d bracket_brown -d bracket_black mpdd $datapath
```

#### High-Resolution PatchCore (IM320_WR50)
```bash
# Better small defect detection
python bin/run_patchcore.py --gpu 0 --seed 22 --save_patchcore_model --save_segmentation_images --save_anomaly_scores \
--log_group MPDD_IM320_WR50_L2-3_P001_D1024-1024_PS-3_AN-1_S22 results \
patch_core -b wideresnet50 -le layer2 -le layer3 \
--pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 -d metal_plate -d bracket_white -d bracket_brown -d bracket_black mpdd $datapath
```

#### Segmentation-Optimized PatchCore (IM320_Segmentation)
```bash
# Precise pixel-level localization
python bin/run_patchcore.py --gpu 0 --seed 39 --save_patchcore_model --save_segmentation_images --save_anomaly_scores \
--log_group MPDD_IM320_WR50_L2-3_P001_D1024-1024_PS-5_AN-3_S39 results \
patch_core -b wideresnet50 -le layer2 -le layer3 \
--pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 3 --patchsize 5 \
sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 -d metal_plate -d bracket_white -d bracket_brown -d bracket_black mpdd $datapath
```

### FastFlow Setup
```bash
# Training - Run FastFlow improved training
bash sample_training_fastflow_improved.sh

# Alternative FastFlow training scripts available:
# bash sample_training_fastflow.sh
# bash sample_training_fastflow_patch.sh
# bash sample_training_fastflow_wresnet50.sh

# Evaluation - Load and evaluate trained FastFlow models
bash sample_evaluation_fastflow_loader.sh
```
### Credits:
The repository is adapted from https://github.com/amazon-science/patchcore-inspection

## Sample Heatmap Visualizations

Metal-plate localisation results from each model:

| PatchCore Good | PatchCore Scratches | FastFlow Good | FastFlow Scratches |
|:--------------:|:-------------------:|:-------------:|:------------------:|
| <img src="images/patchcore_metal_plate_good.png" width="200"/> | <img src="images/patchcore_metal_plate_scratches.png" width="200"/> | <img src="images/fastflow_metal_plate_good.png" width="200"/> | <img src="images/fastflow_metal_plate_scratches.png" width="200"/> |