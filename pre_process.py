import os
from shutil import rmtree
import sys

import argparse
import tqdm
import matplotlib as mpl
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import DataLoader

from utils import *
from models import *
from dataset import *


def load_data(folder):

    meta_files = []
    x_cropped_files = []
    y_files = []

    # List all files in the folder
    files = os.listdir(folder)

    # Filter and sort files
    for file in files:
        if file.endswith("_meta.npy"):
            meta_files.append(file)
        elif file.endswith("_x_cropped.npy"):
            x_cropped_files.append(file)
        elif file.endswith("_y.npy"):
            y_files.append(file)

    meta_files.sort()
    x_cropped_files.sort()
    y_files.sort()

    # Initialize empty arrays to store the data
    meta_data = []
    x_cropped_data = []
    y_data = []

    # Load the data from the files and perform data type conversion
    for meta_file, x_cropped_file, y_file in zip(meta_files, x_cropped_files, y_files):
        meta_data.append(np.load(os.path.join(folder, meta_file)))
        x_cropped_data.append(np.load(os.path.join(folder, x_cropped_file)))
        y_data.append(np.load(os.path.join(folder, y_file)))

    def normalize_image(im):
        return (im - np.min(im)) / (np.max(im) - np.min(im))
    
    resized_x_cropped_data = [cv2.resize(image, (224, 224))[:, :, np.newaxis] for image in x_cropped_data]

    normalized_x_cropped_data = [normalize_image(image.astype(np.float32)) for image in resized_x_cropped_data]

    return np.array(meta_data), np.array(normalized_x_cropped_data), np.array(y_data)

