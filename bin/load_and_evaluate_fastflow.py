import contextlib
import gc
import logging
import os
import sys

import click
import numpy as np
import torch

import patchcore.utils
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"],
             "mpdd": ["patchcore.datasets.mpdd", "MPDDDataset"]}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--save_segmentation_images", is_flag=True)
def main(**kwargs):
    pass


@main.result_callback()
def run(methods, results_path, gpu, seed, save_segmentation_images):
    methods = {key: item for (key, item) in methods}

    os.makedirs(results_path, exist_ok=True)

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    dataloader_iter, n_dataloaders = methods["get_dataloaders_iter"]
    dataloader_iter = dataloader_iter(seed)
    fastflow_iter, n_fastflows = methods["get_fastflow_iter"]
    fastflow_iter = fastflow_iter(device)
    
    if not (n_dataloaders == n_fastflows or n_fastflows == 1):
        raise ValueError(
            "Please ensure that #FastFlows == #Datasets or #FastFlows == 1!"
        )

    for dataloader_count, dataloaders in enumerate(dataloader_iter):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["testing"].name, dataloader_count + 1, n_dataloaders
            )
        )

        patchcore.utils.fix_seeds(seed, device)
        dataset_name = dataloaders["testing"].name

        with device_context:
            torch.cuda.empty_cache()
            if dataloader_count < n_fastflows:
                fastflow_model = next(fastflow_iter)

            # Evaluate FastFlow model
            fastflow_model.eval()
            all_image_scores = []
            all_pixel_scores = []
            all_labels = []
            all_masks = []

            for batch_idx, batch in enumerate(dataloaders["testing"]):
                if isinstance(batch, dict):
                    images = batch["image"]
                    labels = batch["is_anomaly"]
                    masks = batch["mask"] if "mask" in batch else None
                else:
                    images, labels, masks = batch[0], batch[1], batch[2] if len(batch) > 2 else None

                images = images.to(device)
                
                with torch.no_grad():
                    # Get FastFlow outputs
                    outputs = fastflow_model(images)
                    
                    if "anomaly_map" in outputs:
                        pixel_scores = outputs["anomaly_map"]
                        # Compute image-level scores from pixel scores
                        image_scores = torch.mean(pixel_scores.reshape(pixel_scores.shape[0], -1), dim=1)
                    else:
                        # Fallback: compute scores from loss
                        image_scores = outputs["loss"] if "loss" in outputs else torch.zeros(images.shape[0])
                        pixel_scores = torch.zeros((images.shape[0], images.shape[2], images.shape[3]))

                all_image_scores.extend(image_scores.cpu().numpy())
                all_pixel_scores.extend(pixel_scores.cpu().numpy())
                all_labels.extend(labels.numpy())
                if masks is not None:
                    all_masks.extend(masks.numpy())

            # Convert to numpy arrays
            image_scores = np.array(all_image_scores)
            pixel_scores = np.array(all_pixel_scores)
            labels = np.array(all_labels)
            masks = np.array(all_masks) if all_masks else None

            # Convert masks to binary (0 or 1) by thresholding
            if masks is not None:
                binary_masks = (masks > 0.5).astype(int)
            else:
                binary_masks = None

            # Compute metrics
            if len(np.unique(labels)) > 1:
                instance_auroc = roc_auc_score(labels, image_scores)
            else:
                instance_auroc = 0.5

            if binary_masks is not None and len(np.unique(binary_masks.flatten())) > 1:
                full_pixel_auroc = roc_auc_score(binary_masks.flatten(), pixel_scores.flatten())
                
                # Anomaly pixel AUROC (only anomalous images)
                anomalous_indices = np.where(labels > 0.5)[0]
                if len(anomalous_indices) > 0:
                    anomaly_binary_masks = binary_masks[anomalous_indices]
                    anomaly_pixels = pixel_scores[anomalous_indices]
                    if len(np.unique(anomaly_binary_masks.flatten())) > 1:
                        anomaly_pixel_auroc = roc_auc_score(anomaly_binary_masks.flatten(), anomaly_pixels.flatten())
                    else:
                        anomaly_pixel_auroc = 0.5
                else:
                    anomaly_pixel_auroc = 0.5
            else:
                full_pixel_auroc = 0.5
                anomaly_pixel_auroc = 0.5

            # Save segmentation images
            if save_segmentation_images:
                image_paths = [x[2] for x in dataloaders["testing"].dataset.data_to_iterate]
                mask_paths = [x[3] for x in dataloaders["testing"].dataset.data_to_iterate] if hasattr(dataloaders["testing"].dataset, 'data_to_iterate') else None

                def image_transform(image):
                    if hasattr(dataloaders["testing"].dataset, 'transform_std'):
                        in_std = np.array(dataloaders["testing"].dataset.transform_std).reshape(-1, 1, 1)
                        in_mean = np.array(dataloaders["testing"].dataset.transform_mean).reshape(-1, 1, 1)
                        image = dataloaders["testing"].dataset.transform_img(image)
                        return np.clip((image.numpy() * in_std + in_mean) * 255, 0, 255).astype(np.uint8)
                    return image

                def mask_transform(mask):
                    if hasattr(dataloaders["testing"].dataset, 'transform_mask'):
                        return dataloaders["testing"].dataset.transform_mask(mask).numpy()
                    return mask

                # Remove batch dimension from pixel scores for visualization
                pixel_scores_2d = pixel_scores.squeeze(1) if pixel_scores.ndim == 4 else pixel_scores
                
                patchcore.utils.plot_segmentation_images(
                    results_path,
                    image_paths,
                    pixel_scores_2d,
                    image_scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            result_collect.append({
                "dataset_name": dataset_name,
                "instance_auroc": instance_auroc,
                "full_pixel_auroc": full_pixel_auroc,
                "anomaly_pixel_auroc": anomaly_pixel_auroc,
            })

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            del fastflow_model
            gc.collect()

        LOGGER.info("\n\n-----\n")

    # Store results
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        results_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("fastflow_loader")
@click.option("--fastflow_paths", "-p", type=str, multiple=True, default=[])
@click.option("--backbone", type=str, default="resnet18")
@click.option("--flow_steps", type=int, default=8)
@click.option("--input_size", type=int, default=224)
def fastflow_loader(fastflow_paths, backbone, flow_steps, input_size):
    def get_fastflow_iter(device):
        from patchcore.fastflow.model import FastFlow
        from patchcore.fastflow.constants import CLASS_SPECIFIC_PARAMS
        
        for fastflow_path in fastflow_paths:
            gc.collect()
            
            # Extract category from path to get correct parameters
            category = None
            for cat in ["metal_plate", "bracket_white", "bracket_brown", "bracket_black"]:
                if cat in fastflow_path:
                    category = cat
                    break
            
            # Use class-specific parameters if available
            if category and category in CLASS_SPECIFIC_PARAMS:
                class_params = CLASS_SPECIFIC_PARAMS[category]
                model_flow_steps = class_params.get("flow_steps", flow_steps)
                model_hidden_ratio = class_params.get("hidden_ratio", 1.0)
                LOGGER.info(f"Loading {category} model with flow_steps={model_flow_steps}, hidden_ratio={model_hidden_ratio}")
            else:
                model_flow_steps = flow_steps
                model_hidden_ratio = 1.0
                LOGGER.info(f"Loading model with default parameters: flow_steps={model_flow_steps}")
            
            # Load FastFlow model with correct parameters
            model = FastFlow(
                backbone_name=backbone,
                flow_steps=model_flow_steps,
                input_size=input_size,
                conv3x3_only=False,
                hidden_ratio=model_hidden_ratio,
            )
            model.to(device)
            
            # Load checkpoint
            checkpoint = torch.load(fastflow_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            yield model

    return ("get_fastflow_iter", [get_fastflow_iter, len(fastflow_paths)])


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=1, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(name, data_path, subdatasets, batch_size, resize, imagesize, num_workers, augment):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders_iter(seed):
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader.name = name
            if subdataset is not None:
                test_dataloader.name += "_" + subdataset

            dataloader_dict = {"testing": test_dataloader}
            yield dataloader_dict

    return ("get_dataloaders_iter", [get_dataloaders_iter, len(subdatasets)])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main() 