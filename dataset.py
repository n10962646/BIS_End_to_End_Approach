import os
from copy import deepcopy

import cv2
import numpy as np

import torch
from torchvision import transforms
from albumentations.augmentations import Blur, HorizontalFlip, ElasticTransform, RandomScale, Resize, Rotate, RandomContrast
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2  
from skimage import filters


class BreastMRIFusionDataset(torch.utils.data.Dataset):

    """
     A custom PyTorch dataset for breast MRI fusion data.

        Args:
            meta_data (numpy.ndarray): Meta-data features for each sample.
            x_cropped_data (numpy.ndarray): Cropped MRI images.
            y_data (numpy.ndarray): Labels for each sample.
            augment (bool): Whether to apply data augmentation.
            n (int): Number of samples to use (subset of the data).
            target_size (tuple): Target size for image resizing.
        """
    def __init__(self, meta_data, x_cropped_data, y_data, augment=False, n=None):
        self.meta_data = meta_data
        self.x_cropped_data = x_cropped_data
        self.y_data = y_data
        self.augment = augment
        self.n = n

        if self.augment:
            self.transform = Compose([
                OneOf([Blur(blur_limit=4, p=0.5), RandomContrast(p=0.5)], p=1),
                Compose([RandomScale(scale_limit=0.2, p=1), Resize(224, 224, p=1)], p=0.5),
                ToTensorV2()
            ])
        else:
            self.transform = Compose([
                ToTensorV2()
            ])
        
        self.meta_features = self.meta_data.shape[1]

    def __len__(self):
        if self.n is not None:
            return self.n
        else:
            return len(self.x_cropped_data)

    def __getitem__(self, idx):
        x = self.x_cropped_data[idx]
        meta = self.meta_data[idx]
        y = self.y_data[idx]

        if self.augment:
            x = filters.sobel(x)
            augmented = self.transform(image=x)  
            x = augmented["image"].float() 
        else:
            augmented = self.transform(image=x)


        meta = torch.Tensor(meta).float()
        y = torch.Tensor(y).float()

        meta = torch.Tensor(meta).float()
        y = torch.Tensor(y).float()

        return {"image": x, "metadata": meta, "label": y}

    
class FeatureImpDataset(torch.utils.data.Dataset):
    def __init__(self, test_meta, test_x_cropped, test_y):

        """
        A custom PyTorch dataset for feature importance analysis.

        Args:
            test_meta (numpy.ndarray): Meta-data features for each test sample.
            test_x_cropped (numpy.ndarray): Cropped MRI images for testing.
            test_y (numpy.ndarray): Labels for each test sample.
        """
        self.x_test = test_x_cropped        
        self.meta_test = test_meta
        self.y_test = test_y
        self.n = len(test_x_cropped)  
        self.transform = ToTensorV2()

        self.orig_meta_test = test_meta  
        self.meta_test = deepcopy(self.orig_meta_test)  
        self.y_test = test_y  

        self.meta_features = self.orig_meta_test.shape[1]
        
    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Number of test samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
         Get a test sample from the dataset.

        Args:
            idx (int): Index of the test sample to retrieve.

        Returns:
            dict: A dictionary containing the image, metadata, and label.
        """
        x = self.x_test[idx]
        meta = self.meta_test[idx]
        y = self.y_test[idx]

        x = self.transform(image=x)["image"].float() 
        meta = torch.Tensor(meta).float()
        y = torch.Tensor(y).float()

        return {"image": x, "metadata": meta, "label": y}

