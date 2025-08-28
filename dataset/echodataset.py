import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from utils.util import select_device
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

device = select_device()

class EchoDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.device = device
        self.data_path = Path(cfg.dataset.path)
        self.df = pd.read_csv(self.data_path / 'metadata.csv')
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.shape = self.cfg.dataset.get('shape', None)
        if self.shape is None:
            stats = torch.load(self.data_path / f"{self.df.iloc[0]['video_name']}.pt", map_location='cpu')
            self.shape = stats['mu'].shape
        print("Data Shape:", self.shape) # (T, C, H, W)


    def __len__(self):
        return len(self.df)

    def resample_sequence(frames: np.ndarray | torch.Tensor, target_length: int = 32) -> np.ndarray | torch.Tensor:
        """
        Resample a sequence to a target length.
        
        Args:
            frames: Input frames of shape (T, H, W) or (T, H, W, C)
            target_length: Target number of frames
            
        Returns:
            Resampled frames of shape (target_length, H, W) or (target_length, H, W, C)
            
        Notes:
            - If T < target_length, append zero-padded frames
            - If T >= target_length, evenly sample frames including endpoints
        """

        lib = torch if isinstance(frames, torch.Tensor) else np

        T = frames.shape[0]
        if T < target_length:
            # Pad with zeros
            pad_shape = (target_length - T,) + frames.shape[1:]
            pad_block = lib.zeros(pad_shape, dtype=frames.dtype)
            return lib.concatenate([frames, pad_block], axis=0)
        else:
            # Evenly sample frames including endpoints
            indices = lib.linspace(0, T - 1, target_length)
            indices = lib.round(indices).astype(int)
            return frames[indices]
        

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

        observed_mask = mask[:,0:1,...]
        return masked_video, observed_mask



    def transform(self, z: torch.Tensor):
        # randomly mask frames
        z_masked, observed_mask = self.mask_random_frames(z, t=self.cfg.dataset.max_frames)
        #TODO: Look at adding masking to communicate missing/padded regions
        #z_masked = torch.cat([z_masked, observed_mask], dim=1) # concatenate along channel dimension

        # downsample or pad up to T frames
        z = self.resample_sequence(z, target_length=self.cfg.dataset.max_frames)
        z_masked = self.resample_sequence(z_masked, target_length=self.cfg.dataset.max_frames)

        return z, z_masked

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        stats = torch.load(self.data_path / f"{row['video_name']}.pt", map_location=self.device)
        mu, var = stats['mu'], stats['var'] #(T, C, H, W)

        eps = torch.randn_like(mu, device=self.device)
        z = mu + var.sqrt() * eps
        z, z_masked = self.transform(z)

        ef = row['ef']
        ehs_dim = self.cfg.model.kwargs.get('cross_attention_dim', self.cfg.model.kwargs.caption_channels)
        encoder_hidden_states = ef.view(1, 1).expand(1, ehs_dim)

        return {"x": z, "cond_image": z_masked, "encoder_hidden_states": encoder_hidden_states}