import os
import sys
import logging
import click
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from PIL import Image
import inspect

# Import post-processing functions
from patchcore.fastflow.improve_localization import improve_anomaly_maps, adaptive_threshold_anomaly_maps, multi_scale_enhancement

# Add sPRO computation function
import numpy as np
def compute_spro(anomaly_maps, gt_masks):
    """
    Compute soft PRO (sPRO) score: mean soft IoU between anomaly map and mask for all images with anomalies.
    Args:
        anomaly_maps: torch.Tensor or np.ndarray, shape (N, H, W)
        gt_masks: torch.Tensor or np.ndarray, shape (N, H, W), binary
    Returns:
        spro: float
    """
    if isinstance(anomaly_maps, torch.Tensor):
        anomaly_maps = anomaly_maps.cpu().numpy()
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.cpu().numpy()
    spros = []
    for pred, gt in zip(anomaly_maps, gt_masks):
        if np.sum(gt) == 0:
            continue
        # Normalize pred to [0, 1]
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        # Soft intersection and union
        intersection = np.sum(pred * gt)
        union = np.sum(np.maximum(pred, gt))
        spros.append(intersection / (union + 1e-8))
    if spros:
        return float(np.mean(spros))
    else:
        return 0.0

from patchcore.fastflow.model import FastFlow
from patchcore.datasets.mpdd import MPDDDataset, DatasetSplit
from patchcore.fastflow.constants import (
    BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY,
    LOG_INTERVAL, EVAL_INTERVAL, CHECKPOINT_INTERVAL,
    SUPPORTED_BACKBONES, CLASS_SPECIFIC_PARAMS, CLASS_SPECIFIC_PARAMS_WRESNET50
)

LOGGER = logging.getLogger(__name__)

def save_segmentation_visualizations(results_path, dataset, anomaly_maps, image_scores, max_images=200):
    """Save segmentation visualization images similar to PatchCore."""
    segmentation_path = os.path.join(results_path, "segmentation_images")
    os.makedirs(segmentation_path, exist_ok=True)
    
    # Get dataset information
    data_to_iterate = dataset.data_to_iterate
    transform_mean = np.array(dataset.transform_mean).reshape(-1, 1, 1)
    transform_std = np.array(dataset.transform_std).reshape(-1, 1, 1)
    
    # Save up to max_images examples
    num_images = min(len(data_to_iterate), max_images, len(anomaly_maps))
    
    for idx in range(num_images):
        try:
            classname, anomaly_type, image_path, mask_path = data_to_iterate[idx]
            
            # Load and process original image
            original_image = Image.open(image_path).convert('RGB')
            original_image = dataset.transform_img(original_image)
            
            # Denormalize for visualization
            denorm_image = original_image.numpy() * transform_std + transform_mean
            denorm_image = np.clip(denorm_image * 255, 0, 255).astype(np.uint8)
            denorm_image = np.transpose(denorm_image, (1, 2, 0))
            
            # Get anomaly map
            anomaly_map = anomaly_maps[idx].squeeze().numpy()
            
            # Load ground truth mask if available
            if mask_path and os.path.exists(mask_path):
                gt_mask = Image.open(mask_path).convert('L')
                gt_mask = dataset.transform_mask(gt_mask).numpy().squeeze()
            else:
                gt_mask = np.zeros_like(anomaly_map)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(denorm_image)
            axes[0].set_title(f'Original\n{classname}_{anomaly_type}')
            axes[0].axis('off')
            
            # Ground truth mask
            axes[1].imshow(gt_mask, cmap='gray')
            axes[1].set_title('Ground Truth Mask')
            axes[1].axis('off')
            
            # Anomaly heatmap
            im = axes[2].imshow(denorm_image)
            heatmap = axes[2].imshow(anomaly_map, cmap='jet', alpha=0.6)
            axes[2].set_title(f'Anomaly Heatmap\nScore: {image_scores[idx]:.3f}')
            axes[2].axis('off')
            
            # Add colorbar for heatmap
            plt.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
            
            # Save the figure
            save_name = f"{idx:03d}_{classname}_{anomaly_type}_score{image_scores[idx]:.3f}.png"
            save_path = os.path.join(segmentation_path, save_name)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            LOGGER.warning(f"Failed to save segmentation image {idx}: {e}")
            continue
    
    LOGGER.info(f"Saved {num_images} segmentation images to: {segmentation_path}")

@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=0, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_model", is_flag=True)
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--eval_only", is_flag=True, help="Run evaluation only (no training)")
@click.option("--checkpoint", type=str, default=None, help="Path to model checkpoint for evaluation")
def main(**kwargs):
    pass

@main.command("dataset")
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--category", type=str, required=True)
@click.option("--batch_size", default=None, type=int, show_default=True)
@click.option("--num_workers", default=4, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
def dataset(data_path, category, batch_size, num_workers, resize, imagesize):
    def get_dataloaders(seed):
        bs = batch_size if batch_size is not None else BATCH_SIZE
        train_dataset = MPDDDataset(
            source=data_path,
            classname=category,
            resize=resize,
            imagesize=imagesize,
            split=DatasetSplit.TRAIN,
            use_augmentation=True,
        )
        test_dataset = MPDDDataset(
            source=data_path,
            classname=category,
            resize=resize,
            imagesize=imagesize,
            split=DatasetSplit.TEST,
            use_augmentation=False,
        )
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)
        return {"training": train_loader, "testing": test_loader}
    return ("get_dataloaders", get_dataloaders)

@main.command("fastflow")
@click.option("--backbone", type=str, default="resnet18")
@click.option("--flow_steps", type=int, default=8)
@click.option("--input_size", type=int, default=224)
@click.option("--conv3x3_only", is_flag=True)
@click.option("--hidden_ratio", type=float, default=1.0)
@click.option("--use_class_specific", is_flag=True, help="Use class-specific hyperparameters")
def fastflow(backbone, flow_steps, input_size, conv3x3_only, hidden_ratio, use_class_specific):
    def get_fastflow():
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone {backbone} not supported. Choose from: {SUPPORTED_BACKBONES}")
        model = FastFlow(
            backbone_name=backbone,
            flow_steps=flow_steps,
            input_size=input_size,
            conv3x3_only=conv3x3_only,
            hidden_ratio=hidden_ratio,
        )
        return model
    return ("get_fastflow", get_fastflow)

@main.result_callback()
def run(methods, results_path, gpu, seed, log_group, log_project, save_model, save_segmentation_images, eval_only, checkpoint):
    methods = {key: item for (key, item) in methods}
    os.makedirs(results_path, exist_ok=True)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataloaders = methods["get_dataloaders"](seed)
    
    # Get category for class-specific parameters
    category = dataloaders["training"].dataset.classnames_to_use[0]
    
    # Get backbone name from get_fastflow closure
    backbone_name = "resnet18"  # default
    if "get_fastflow" in methods:
        closure_vars = inspect.getclosurevars(methods["get_fastflow"])
        if "backbone" in closure_vars.nonlocals:
            backbone_name = closure_vars.nonlocals["backbone"]
        elif "backbone_name" in closure_vars.nonlocals:
            backbone_name = closure_vars.nonlocals["backbone_name"]
    LOGGER.info(f"Detected backbone: {backbone_name}")

    # Choose parameter set based on backbone
    if "wide_resnet50" in backbone_name.lower():
        param_set = CLASS_SPECIFIC_PARAMS_WRESNET50
        LOGGER.info(f"Using WideResNet50-optimized parameters")
    else:
        param_set = CLASS_SPECIFIC_PARAMS
        LOGGER.info(f"Using ResNet18-optimized parameters")
    
    # Check if class-specific parameters should be used and create model accordingly
    use_class_params = any('use_class_specific' in str(result) for result in methods.items())
    if category in param_set:
        class_params = param_set[category]
        learning_rate = class_params["lr"]
        num_epochs = class_params["num_epochs"] 
        weight_decay = class_params["weight_decay"]
        flow_steps = class_params.get("flow_steps", 8)
        hidden_ratio = class_params.get("hidden_ratio", 1.0)
        LOGGER.info(f"Using class-specific params for {category}: lr={learning_rate}, epochs={num_epochs}, flow_steps={flow_steps}, hidden_ratio={hidden_ratio}")
        
        # Create model with class-specific parameters
        model = FastFlow(
            backbone_name=backbone_name,
            flow_steps=flow_steps,
            input_size=224,
            conv3x3_only=False,
            hidden_ratio=hidden_ratio,
        ).to(device)
    else:
        learning_rate = LR
        num_epochs = NUM_EPOCHS
        weight_decay = WEIGHT_DECAY
        model = methods["get_fastflow"]().to(device)
        LOGGER.info(f"Using default params: lr={learning_rate}, epochs={num_epochs}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Add cosine annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Load checkpoint if provided
    if checkpoint is not None:
        LOGGER.info(f"Loading checkpoint from {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    if not eval_only:
        # Training loop
        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss = 0
            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            LOGGER.info(f"Epoch {epoch} - Learning Rate: {current_lr:.6f}")
            for step, batch in enumerate(dataloaders["training"]):
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = out["loss"]
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if (step + 1) % LOG_INTERVAL == 0 or (step + 1) == len(dataloaders["training"]):
                    LOGGER.info(f"Epoch {epoch} - Step {step+1}: loss = {loss.item():.3f} (avg: {train_loss/(step+1):.3f})")
            # Step the scheduler at the end of each epoch
            scheduler.step()
            if epoch % EVAL_INTERVAL == 0:
                # Evaluation
                model.eval()
                all_anomaly_maps = []
                all_masks = []
                with torch.no_grad():
                    for batch in dataloaders["testing"]:
                        for k in batch:
                            if isinstance(batch[k], torch.Tensor):
                                batch[k] = batch[k].to(device)
                        anomaly_map = model.predict(batch)
                        all_anomaly_maps.append(anomaly_map.cpu())
                        all_masks.append(batch["mask"].cpu())
                all_anomaly_maps = torch.cat(all_anomaly_maps, dim=0)
                all_masks = torch.cat(all_masks, dim=0)
                # Convert masks to binary (0 or 1) by thresholding
                binary_masks = (all_masks > 0.5).float()
                auroc = roc_auc_score(binary_masks.flatten().numpy(), all_anomaly_maps.flatten().numpy())
                LOGGER.info(f"[Epoch {epoch}] Pixel-level AUROC: {auroc}")
            if epoch % CHECKPOINT_INTERVAL == 0 and save_model:
                torch.save(model.state_dict(), os.path.join(results_path, f"fastflow_{log_group}_epoch{epoch}.pth"))
        # Save final model
        if save_model:
            torch.save(model.state_dict(), os.path.join(results_path, f"fastflow_{log_group}_final.pth"))

    # Final Evaluation with comprehensive metrics
    model.eval()
    all_anomaly_maps = []
    all_masks = []
    all_image_scores = []
    all_anomaly_labels = []
    
    with torch.no_grad():
        for batch in dataloaders["testing"]:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            anomaly_map = model.predict(batch)
            all_anomaly_maps.append(anomaly_map.cpu())
            all_masks.append(batch["mask"].cpu())
            
            # Compute image-level scores (max of anomaly map)
            image_scores = torch.mean(anomaly_map.view(anomaly_map.shape[0], -1), dim=1)
            all_image_scores.append(image_scores.cpu())
            
            # Get anomaly labels (1 if any pixel is anomalous, 0 otherwise)
            anomaly_labels = (torch.max(batch["mask"].view(batch["mask"].shape[0], -1), dim=1)[0] > 0.5).float()
            all_anomaly_labels.append(anomaly_labels.cpu())
    
    all_anomaly_maps = torch.cat(all_anomaly_maps, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_image_scores = torch.cat(all_image_scores, dim=0)
    all_anomaly_labels = torch.cat(all_anomaly_labels, dim=0)
    
    # === Post-processing for improved localization ===
    # all_anomaly_maps = improve_anomaly_maps(all_anomaly_maps, method="all")
    # all_anomaly_maps = adaptive_threshold_anomaly_maps(all_anomaly_maps)
    # all_anomaly_maps = multi_scale_enhancement(all_anomaly_maps)

    # Recompute image_scores from post-processed maps
    all_image_scores = torch.mean(all_anomaly_maps.view(all_anomaly_maps.shape[0], -1), dim=1)
    # ===============================================
    
    # Convert masks to binary (0 or 1) by thresholding
    binary_masks = (all_masks > 0.5).float()
    
    # Compute comprehensive metrics
    # 1. Instance-level AUROC (image-level anomaly detection)
    instance_auroc = roc_auc_score(all_anomaly_labels.numpy(), all_image_scores.numpy())
    
    # 2. Full pixel-level AUROC (all images)
    full_pixel_auroc = roc_auc_score(binary_masks.flatten().numpy(), all_anomaly_maps.flatten().numpy())
    
    # 3. Anomaly pixel-level AUROC (only anomalous images)
    anomalous_indices = torch.where(all_anomaly_labels > 0.5)[0]
    if len(anomalous_indices) > 0:
        anomaly_masks = binary_masks[anomalous_indices]
        anomaly_maps = all_anomaly_maps[anomalous_indices]
        anomaly_pixel_auroc = roc_auc_score(anomaly_masks.flatten().numpy(), anomaly_maps.flatten().numpy())
    else:
        anomaly_pixel_auroc = 0.0  # No anomalous images
    
    # Compute sPRO (soft PRO)
    spro = compute_spro(all_anomaly_maps, binary_masks)

    # Log results
    LOGGER.info("=== FastFlow Evaluation Results (with post-processing) ===")
    LOGGER.info(f"Instance AUROC: {instance_auroc:.4f}")
    LOGGER.info(f"Full Pixel AUROC: {full_pixel_auroc:.4f}")
    LOGGER.info(f"Anomaly Pixel AUROC: {anomaly_pixel_auroc:.4f}")
    LOGGER.info(f"sPRO: {spro:.4f}")
    
    # Save results to CSV (similar to PatchCore format)
    dataset_name = f"fastflow_{dataloaders['testing'].dataset.classnames_to_use[0]}"
    results = {
        "dataset_name": dataset_name,
        "instance_auroc": instance_auroc,
        "full_pixel_auroc": full_pixel_auroc,
        "anomaly_pixel_auroc": anomaly_pixel_auroc,
        "spro": spro,
    }
    
    # Save to CSV file
    csv_path = os.path.join(results_path, "fastflow_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['dataset_name', 'instance_auroc', 'full_pixel_auroc', 'anomaly_pixel_auroc', 'spro']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results)
    
    LOGGER.info(f"Results saved to: {csv_path}")
    
    if save_segmentation_images:
        save_segmentation_visualizations(results_path, dataloaders["testing"].dataset, all_anomaly_maps, all_image_scores)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
