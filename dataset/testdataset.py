import torch
import random
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
    
class FlowTestDataset(Dataset):
    def __init__(self, B=2, T=32, C=4, H=28, W=28, device='cpu', cross_attention_dim=32):

        self.B = B
        self.T = T
        self.C = C
        self.H = H
        self.W = W
        self.device = device
        self.cross_attention_dim = cross_attention_dim
        super().__init__()


    def mask_random_frames(self, video: torch.Tensor, t:int):
        '''
        Masks 'n_missing_frames' in the range [0, t) of the input video tensor.
        '''
        T, C, H, W = video.shape
        assert t <= T, "t must be less than or equal to T"

        n_missing_frames = torch.randint(low=1, high=t, size=(1,)).item() #[1,t)i.e., Upper bound not inclusive therefore t-1 is max
        mask_indices = random.sample(range(t), n_missing_frames)

        mask = torch.ones_like(video)
        mask[mask_indices, ...] = 0

        masked_video = video * mask
        return masked_video

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        # Mock data
        video = torch.randn(self.T, self.C, self.H, self.W).to(self.device) # [T, C, H, W]
        masked_video = self.mask_random_frames(video, t=28).to(self.device)

        video = video.permute(1, 0, 2, 3).contiguous() # [C, T, H, W]
        masked_video = masked_video.permute(1, 0, 2, 3).contiguous() # [C, T, H, W]
        ef = torch.rand(1).to(self.device)
        # EF is a float per sample — treat it as a 1-dim token by expanding to match expected encoder_hidden_states
        # Here we create encoder_hidden_states of shape [B, 1, 1] (seq_len=1, dim=1)
        encoder_hidden_states = ef.view(1, 1).expand(1, self.cross_attention_dim)

        assert video.shape[1] == masked_video.shape[1], "Temporal dimensions must match"
        return {"x": video, "cond_image": masked_video, "encoder_hidden_states": encoder_hidden_states}