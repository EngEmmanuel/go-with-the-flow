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
        video = torch.randn(self.B, self.T, self.C, self.H, self.W).to(self.device) # [B, T, C, H, W]
        masked_video = torch.ones_like(video).to(self.device)
        for i in range(video.shape[0]):
            masked_video[i] = self.mask_random_frames(video[i], t=28)


        video = video.permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
        masked_video = masked_video.permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
        ef = torch.rand(self.B).to(self.device)
        # EF is a float per sample — treat it as a 1-dim token by expanding to match expected encoder_hidden_states
        # Here we create encoder_hidden_states of shape [B, 1, 1] (seq_len=1, dim=1)
        encoder_hidden_states = ef.view(self.B, 1, 1).expand(self.B, 1, self.cross_attention_dim)

        assert video.shape[2] == masked_video.shape[2], "Temporal dimensions must match"
        return video, masked_video, encoder_hidden_states