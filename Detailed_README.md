# PatchCore Anomaly Detection Methods

I have implemented multiple anomaly detection methods for industrial defect detection, and tested on the MPDD (Metal Parts Defect Detection) dataset.

## PatchCore Method

PatchCore is a memory-based anomaly detection approach that leverages pretrained CNN features for few-shot anomaly detection. The method extracts multi-scale feature representations from pretrained networks (WideResNet, ResNet, DenseNet) and stores them in a memory bank using coreset sampling to reduce computational overhead. During inference, it performs nearest neighbor search in the feature space to compute anomaly scores, making it highly effective for pixel-level defect localization without requiring extensive training data.

### PatchCore Variants

#### Standard PatchCore (IM224_WR50)
Uses WideResNet50 backbone with 224×224 input resolution and 10% coreset sampling. This variant provides balanced performance between speed and accuracy, making it suitable for general-purpose anomaly detection with fast inference times and moderate memory requirements.

#### High-Resolution PatchCore (IM320_WR50)
Extends the standard approach to 320×320 input resolution with 1% coreset sampling for more selective feature retention. This variant excels at detecting small defects and fine-grained anomalies that might be missed at lower resolutions, though it requires more computational resources.

#### Ensemble PatchCore (IM224_Ensemble)
Combines multiple backbone networks (WideResNet101, ResNext101, DenseNet201) with compressed 384-dimensional embeddings and 1% coreset sampling. This variant achieves maximum accuracy by leveraging complementary features from different architectures, making it ideal for research applications and high-stakes quality control.

#### Segmentation-Optimized PatchCore (IM320_Segmentation)
Specifically tuned for precise pixel-level localization using larger patch sizes (5×5) and multiple nearest neighbors (3) for smoother segmentation maps. This variant produces pixel-perfect defect boundaries and is optimized for applications requiring exact defect area measurements and detailed quality reporting.

## FastFlow Method

FastFlow is a normalizing flow-based generative model that learns the distribution of normal data to detect anomalies through likelihood estimation. The method uses coupling layers and invertible transformations to model the normal data distribution, enabling real-time inference with probabilistic anomaly scores. It excels at image-level classification tasks.

### FastFlow Implementation

The FastFlow implementation uses ResNet18 as the feature extractor backbone with configurable flow steps and hidden ratios. Class-specific hyperparameters are employed for different MPDD categories, with optimized learning rates, training epochs, and architectural parameters tailored to each defect type. The model supports both standard and improved variants(by changing the feature extractor, not experiemented here though) with enhanced localization capabilities.

## Usage

Both methods support comprehensive evaluation with anomaly score output, segmentation visualization, and CSV result exports. The implementation includes training scripts, evaluation utilities, and comparison tools for benchmarking performance across different industrial inspection scenarios.

Training scripts are available for each method variant:
- `sample_training_mpdd.sh` - All PatchCore variants
- `sample_training_fastflow_improved.sh` - FastFlow with class-specific parameters

Evaluation scripts generate performance metrics and visual results:
- `sample_evaluation_mpdd.sh` - PatchCore model evaluation
- `sample_evaluation_fastflow_loader.sh` - FastFlow model evaluation with class-specific folders 