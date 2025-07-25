# MPDD (Metal Parts Defect Detection) Dataset Setup Guide

This guide explains how to use the MPDD dataset with the PatchCore implementation for industrial anomaly detection.

## What's Been Added

- `src/patchcore/datasets/mpdd.py` - MPDD dataset implementation
- `sample_training_mpdd.sh` - Training script examples for MPDD
- `sample_evaluation_mpdd.sh` - Evaluation script for MPDD models
- Updated dataset registries in training and evaluation scripts

## Directory Structure Requirements

Your MPDD dataset should be organized in one of these structures:

### Structure 1 (Preferred - Similar to MVTec):
```
mpdd/
├── screw/
│   ├── train/
│   │   └── good/
│   │       ├── image001.jpg
│   │       ├── image002.jpg
│   │       └── ...
│   ├── test/
│   │   ├── good/
│   │   │   ├── image001.jpg
│   │   │   └── ...
│   │   ├── scratch/
│   │   │   ├── defect001.jpg
│   │   │   └── ...
│   │   └── dent/
│   │       ├── defect002.jpg
│   │       └── ...
│   └── ground_truth/
│       ├── scratch/
│       │   ├── defect001_mask.png
│       │   └── ...
│       └── dent/
│           ├── defect002_mask.png
│           └── ...
├── nut/
│   ├── train/
│   ├── test/
│   └── ground_truth/
└── ...
```

### Structure 2 (Alternative):
```
mpdd/
├── train/
│   ├── screw/
│   │   └── good/
│   └── nut/
│       └── good/
├── test/
│   ├── screw/
│   │   ├── good/
│   │   ├── scratch/
│   │   └── dent/
│   └── nut/
│       ├── good/
│       └── crack/
└── ground_truth/
    ├── screw/
    │   ├── scratch/
    │   └── dent/
    └── nut/
        └── crack/
```

## Supported Image Formats

The dataset loader supports the following image formats:
- `.png`
- `.jpg` / `.jpeg`
- `.bmp`
- `.tiff`

## Usage Instructions

### 1. Update Dataset Categories

Edit the `datasets` variable in `sample_training_mpdd.sh` and `sample_evaluation_mpdd.sh` to match your actual MPDD categories:

```bash
# Example categories - update these based on your dataset
datasets=('screw' 'nut' 'bolt' 'washer' 'bracket' 'gear' 'spring' 'bearing' 'pipe' 'plate')
```

You can also update the default categories in `src/patchcore/datasets/mpdd.py` by modifying the `_CLASSNAMES` list.

### 2. Set Data Path

Update the `datapath` variable in both scripts:

```bash
datapath=/path/to/your/mpdd/dataset
```

### 3. Training

Run the training script:

```bash
bash sample_training_mpdd.sh
```

This will train multiple PatchCore models:
- **Baseline WR50**: WideResNet50 backbone with 224x224 images
- **Ensemble**: Multiple backbones (WR101, ResNext101, DenseNet201)
- **High Resolution**: 320x320 images for detailed inspection
- **Segmentation Optimized**: Parameters tuned for pixel-level defect localization

### 4. Evaluation

After training, evaluate the models:

```bash
bash sample_evaluation_mpdd.sh
```

Update the `loadpath` and `modelfolder` variables to point to your trained models.

## Training Parameters Explanation

### Key Parameters:

- **`-b wideresnet50`**: Backbone network (also supports resnet50, resnet101, etc.)
- **`-le layer2 -le layer3`**: Feature extraction layers
- **`--pretrain_embed_dimension 1024`**: Dimension of features from backbone
- **`--target_embed_dimension 1024`**: Final PatchCore feature dimension
- **`--anomaly_scorer_num_nn 1`**: Number of nearest neighbors for anomaly scoring
- **`--patchsize 3`**: Neighborhood size for local feature aggregation
- **`-p 0.1`**: Coreset subsampling percentage (10%)
- **`--resize 256 --imagesize 224`**: Input preprocessing (resize to 256, center crop to 224)

### For Better Segmentation:
- Increase `--patchsize` to 5
- Increase `--anomaly_scorer_num_nn` to 3 or 5
- Use higher resolution (320x320)

### For Better Detection:
- Use ensemble of multiple backbones
- Lower coreset percentage (1% = `-p 0.01`) for more representative memory
- Higher resolution images

## Expected Performance

The trained models will be evaluated on:
- **Image-level AUROC**: Overall anomaly detection performance
- **Pixel-level AUROC**: Defect localization accuracy
- **PRO Score**: Region-based evaluation metric

Results are saved in `results.csv` with performance metrics for each metal part category.

## Troubleshooting

### Common Issues:

1. **"Class path does not exist"**: Check your directory structure and update category names
2. **"No images found"**: Ensure image files have supported extensions
3. **Memory errors**: Reduce batch size or use lower resolution images
4. **GPU errors**: Check GPU memory and reduce model size if needed

### Debug Mode:

For debugging, try training on a single category first:

```bash
datasets=('screw')  # Single category for testing
```

## Customization

### Adding New Categories:

1. Add category names to the `datasets` array in scripts
2. Ensure your data follows the directory structure
3. Optionally update `_CLASSNAMES` in `mpdd.py`

### Different Backbones:

Available backbones (see `src/patchcore/backbones.py`):
- `wideresnet50`, `wideresnet101`
- `resnet18`, `resnet34`, `resnet50`, `resnet101`
- `resnext50`, `resnext101`
- `densenet121`, `densenet161`, `densenet201`

### Custom Preprocessing:

Modify the transform pipelines in `MPDDDataset.__init__()` if your images require specific preprocessing.

## Results Interpretation

After evaluation, check `results.csv` for:
- **Instance AUROC**: How well the model detects anomalous parts (higher is better)
- **Pixelwise AUROC**: How accurately it localizes defects (higher is better)
- **PRO**: Precision-Recall-Optimized score for region-based evaluation

Good performance typically shows:
- Instance AUROC > 90%
- Pixelwise AUROC > 85%
- PRO > 80%

Performance may vary depending on:
- Dataset quality and size
- Defect complexity
- Image resolution
- Training parameters 