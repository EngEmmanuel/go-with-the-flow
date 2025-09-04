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
    def __init__(self, cfg, split='train', cache=True, **kwargs):
        self.cfg = cfg
        self.data_path = Path(cfg.dataset.path)
        self.df = pd.read_csv(self.data_path / 'metadata.csv')
        self.split = split.lower()
        self.cache = cache
        self._cache = {}  # dict for in-memory caching

        # Filter dataframe based on split
        if split == 'sample':
            self.df = self.df[self.df['split'].str.lower() == 'val'].reset_index(drop=True)
        else:
            self.df = self.df[self.df['split'].str.lower() == split].reset_index(drop=True)

        # Sample subset for monitoring
        if kwargs.get('n_sample_videos', None) is not None:
            n_sample_videos = kwargs['n_sample_videos']
            self.df = self.df.sample(n=min(n_sample_videos, len(self.df))).reset_index(drop=True)

        # Get shape
        self.shape = self.cfg.dataset.get('shape', None)
        if self.shape is None:
            stats = torch.load(self.data_path / f"{self.df.iloc[0]['video_name']}.pt", map_location='cpu')
            self.shape = (self.cfg.dataset.max_frames,) + stats['mu'].shape[1:]

        print(f"{split} Data Shape: {self.shape}")  # (T, C, H, W)
        print(f"{split} Dataset size: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def resample_sequence(self, frames: torch.Tensor, target_length: int = 32):
        """
        Resample to fixed length. Returns resampled frames and a padding mask.
        """
        T, *data_shape = frames.shape
        if T < target_length:
            pad_shape = (target_length - T,) + frames.shape[1:]
            pad_block = torch.zeros(pad_shape, dtype=frames.dtype)
            resampled_frames = torch.cat([frames, pad_block], dim=0)
            pad_mask = torch.cat(
                [torch.zeros((T, 1, *data_shape[1:]), dtype=frames.dtype),
                 torch.ones((target_length - T, 1, *data_shape[1:]), dtype=frames.dtype)],
                dim=0
            )
        else:
            indices = torch.linspace(0, T - 1, target_length).round().long()
            resampled_frames = frames[indices]
            pad_mask = torch.zeros((target_length, 1, *data_shape[1:]), dtype=frames.dtype)

        return resampled_frames, pad_mask

    def mask_random_frames(self, video: torch.Tensor, dist='uniform'):
        '''
        Masks 'n_missing_frames' in the range [0, T) of the input video tensor.
        '''
        T = video.shape[0]

        n_missing_frames = self.missing_frame_distribution(T, dist=dist)
        mask_indices = random.sample(range(T), n_missing_frames)

        mask = torch.ones_like(video)
        mask[mask_indices, ...] = 0
        masked_video = video * mask
        observed_mask = mask[:, 0:1, ...]

        return masked_video, observed_mask

    def missing_frame_distribution(self, T, dist):
        '''
        Returns the number of frames to remove based on the specified distribution.
        '''
        min_frames_removed = 1
        max_frames_removed = T - 1
        match dist:
            case 'uniform':
                val = np.random.randint(1, T)
            case 'geometric':
                p = 8/T
                val = T - np.random.geometric(p)

        return np.clip(val, min_frames_removed, max_frames_removed, dtype=int)

    def transform(self, z: torch.Tensor):
        # Random masking
        z_masked, observed_mask = self.mask_random_frames(z)

        # Resample to fixed length
        resampled, pad_mask = self.resample_sequence(
            torch.cat([z, z_masked], dim=1),
            target_length=self.cfg.dataset.max_frames
        )
        z, cond = resampled[:, :self.shape[1], ...], resampled[:, self.shape[1]:, ...]
        return z, cond

    def process_ef(self, ef, dtype=None):
        ef = torch.tensor(ef, dtype=dtype)  # stay on CPU
        ehs_dim = self.cfg.model.kwargs.get('cross_attention_dim',
                                            self.cfg.model.kwargs.get('caption_channels'))
        encoder_hidden_states = ef.view(1, 1).expand(1, ehs_dim)  # shape [1, dim]
        return encoder_hidden_states

    def _load_video(self, video_name: str):
        if self.cache and video_name in self._cache:
            return self._cache[video_name]

        stats = torch.load(self.data_path / f"{video_name}.pt", map_location='cpu')
        if self.cache:
            self._cache[video_name] = stats
        return stats

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        stats = self._load_video(row['video_name'])
        mu, var = stats['mu'], stats['var']  # (T, C, H, W)

        eps = torch.randn_like(mu) if self.split != 'val' else torch.zeros_like(mu)
        z = (mu + var.sqrt() * eps) * self.cfg.vae.scaling_factor
        z, cond = self.transform(z)

        # Rearrange to (C, T, H, W), stay on CPU, ensure contiguous
        z = z.permute(1, 0, 2, 3).contiguous()
        cond = cond.permute(1, 0, 2, 3).contiguous()

        ef = row['EF_Area'] / 100.0
        encoder_hidden_states = self.process_ef(ef, dtype=z.dtype)

        inputs = {
            "x": z,
            "cond_image": cond,
            "encoder_hidden_states": encoder_hidden_states
        }
        if self.split in ['sample', 'test']:
            inputs['video_name'] = row['video_name']

        return inputs
