import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models import RoadNet
from dataset import train_data, test_data, load_dataset_from_dir

data_dir = os.environ['ROADNET_DATADIR']
