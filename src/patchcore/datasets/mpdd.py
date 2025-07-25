import os
from enum import Enum

import PIL
import torch
from torchvision import transforms
import random
import numpy as np
import cv2

# Common metal part categories that might be in MPDD
# This can be updated based on the actual MPDD dataset structure
_CLASSNAMES = [
    "bracket_black",
    "bracket_brown",
    "bracket_white",
    "metal_plate",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class MPDDDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MPDD (Metal Parts Defect Detection Dataset).
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        use_augmentation=False,
        preserve_high_res=False,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MPDD data folder.
            classname: [str or None]. Name of MPDD class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mpdd.DatasetSplit.TRAIN. Note that
                   mpdd.DatasetSplit.TEST will also load mask data.
            use_augmentation: [bool]. If True, apply data augmentations (for FastFlow).
            preserve_high_res: [bool]. If True and split=TEST, preserve original resolution for patch-based inference.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        
        self.transform_img = [
                # transforms.Resize(resize),
                # transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            # transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)
        
        # Add these for visualization support
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            # MPDD might have different directory structure
            # Try both common structures
            classpath = os.path.join(self.source, classname, self.split.value)
            
            # Alternative structure if direct split folders don't exist
            if not os.path.exists(classpath):
                # Try structure where train/test are at root level
                classpath = os.path.join(self.source, self.split.value, classname)
            
            # For ground truth masks
            maskpath = os.path.join(self.source, classname, "ground_truth")
            if not os.path.exists(maskpath):
                # Alternative mask structure
                maskpath = os.path.join(self.source, "ground_truth", classname)

            if not os.path.exists(classpath):
                print(f"Warning: Class path {classpath} does not exist. Skipping class {classname}")
                continue

            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                if not os.path.isdir(anomaly_path):
                    continue
                    
                anomaly_files = sorted([f for f in os.listdir(anomaly_path) 
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                # Handle mask paths for defective samples
                if self.split == DatasetSplit.TEST and anomaly != "good" and os.path.exists(maskpath):
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    if os.path.exists(anomaly_mask_path):
                        anomaly_mask_files = sorted([f for f in os.listdir(anomaly_mask_path)
                                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
                        maskpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                        ]
                    else:
                        maskpaths_per_class[classname][anomaly] = [None] * len(imgpaths_per_class[classname][anomaly])
                else:
                    maskpaths_per_class[classname][anomaly] = [None] * len(imgpaths_per_class[classname][anomaly])

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        mask_path = maskpaths_per_class[classname][anomaly][i] if i < len(maskpaths_per_class[classname][anomaly]) else None
                        data_tuple.append(mask_path)
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate 

class RandomCLAHE:
    def __init__(self, p=0.3, clip_limit=2.0, tile_grid_size=(8,8)):
        self.p = p
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = np.array(img)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            img_np = clahe.apply(img_np)
        return PIL.Image.fromarray(img_np) 