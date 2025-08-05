import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

class EchoDataset(Dataset):
    def __init__(self, cfg):
        pass