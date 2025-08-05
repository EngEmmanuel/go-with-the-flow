import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

class TestDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.files = list(Path(cfg.data_path).glob("*.png"))

        transform_list = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((cfg.image_size, cfg.image_size)),  # default interpolation (bilinear)
            v2.Grayscale(num_output_channels=cfg.channels)
        ]

        self.transform = v2.Compose(transform_list)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        return self.transform(image)