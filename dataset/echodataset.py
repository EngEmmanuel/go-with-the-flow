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
    def __init__(self, cfg, split='train', **kwargs):
        self.cfg = cfg
        self.device = device
        self.data_path = Path(cfg.dataset.path)
        self.df = pd.read_csv(self.data_path / 'metadata.csv')
        self.split = split.lower()

        # Filter dataframe based on split
        if split == 'sample':
            self.df = self.df[self.df['split'].str.lower() == 'val'].reset_index(drop=True)
        else:
            self.df = self.df[self.df['split'].str.lower() == split].reset_index(drop=True)

        # Number of sample videos to use for train time monitoring
        if kwargs.get('n_sample_videos', None) is not None:
            n_sample_videos = kwargs['n_sample_videos']
            self.df = self.df.sample(n=min(n_sample_videos, len(self.df))).reset_index(drop=True)


        self.shape = self.cfg.dataset.get('shape', None)
        if self.shape is None:
            stats = torch.load(self.data_path / f"{self.df.iloc[0]['video_name']}.pt", map_location='cpu')
            self.shape = (self.cfg.dataset.max_frames,) + stats['mu'].shape[1:]


        print(f"{split} Data Shape:", self.shape) # (T, C, H, W)
        print(f"{split} Dataset size: {len(self.df)}")


    def __len__(self):
        return len(self.df)

    def resample_sequence(self, frames: torch.Tensor, target_length: int = 32) -> np.ndarray | torch.Tensor:
        """
        Resample a sequence to a target length.
        Args:
            frames: Input frames of shape (T, C, H, W)
            target_length: Target number of frames
        Returns:
            Resampled frames of shape (target_length, C, H, W)
            and a padding mask of shape (target_length, 1, H, W)
        Notes:
            - If T < target_length, append zero-padded frames
            - If T >= target_length, evenly sample frames including endpoints
        """
        T, *data_shape = frames.shape
        if T < target_length:
            # Pad with zeros
            pad_shape = (target_length - T,) + frames.shape[1:]
            pad_block = torch.zeros(pad_shape, dtype=frames.dtype)
            resampled_frames = torch.cat([frames, pad_block], dim=0)
            pad_mask = torch.cat(
                [torch.zeros((T, 1, *data_shape[1:]), dtype=frames.dtype),
                 torch.ones((target_length - T, 1, *data_shape[1:]), dtype=frames.dtype)
                 ],
                dim=0)
        else:
            # Evenly sample frames including endpoints
            indices = torch.linspace(0, T - 1, target_length)
            indices = torch.round(indices).to(torch.int64)
            resampled_frames = frames[indices]
            pad_mask = torch.zeros((target_length, 1, *data_shape[1:]), dtype=frames.dtype)

        return resampled_frames, pad_mask

    def mask_random_frames(self, video: torch.Tensor):
        '''
        Masks 'n_missing_frames' in the range [0, t) of the input video tensor.
        '''
        T = video.shape[0]

        n_missing_frames = torch.randint(low=1, high=T, size=(1,)).item() #[1,t)i.e., Upper bound not inclusive therefore t-1 is max
        mask_indices = random.sample(range(T), n_missing_frames)

        mask = torch.ones_like(video)
        mask[mask_indices, ...] = 0

        masked_video = video * mask

        observed_mask = mask[:,0:1,...]
        return masked_video, observed_mask



    def transform(self, z: torch.Tensor):
        # randomly mask frames
        z_masked, observed_mask = self.mask_random_frames(z)

        #z.shape == z_masked.shape = (t, C, H, W)
        # downsample or pad up to T frames
        resampled, pad_mask = self.resample_sequence(
            torch.cat([z, z_masked], dim=1),
            target_length=self.cfg.dataset.max_frames
        )

        z, cond = resampled[:,:self.shape[1],...], resampled[:,self.shape[1]:,...]

        return z, cond

    def process_ef(self, ef, dtype=None):
        ef = torch.tensor(ef, device=self.device, dtype=dtype)
        ehs_dim = self.cfg.model.kwargs.get('cross_attention_dim', self.cfg.model.kwargs.get('caption_channels'))
        encoder_hidden_states = ef.view(1, 1).expand(1, ehs_dim).to(self.device)
        return encoder_hidden_states

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        stats = torch.load(self.data_path / f"{row['video_name']}.pt")
        mu, var = stats['mu'], stats['var'] #(T, C, H, W)

        eps = torch.randn_like(mu)
        z = mu + var.sqrt() * eps
        z, cond = self.transform(z)

        z = z.permute(1, 0, 2, 3).to(self.device)
        cond = cond.permute(1, 0, 2, 3).to(self.device)

        ef = row['EF_Area'] / 100.0  # Normalize EF to [0, 1]
        encoder_hidden_states = self.process_ef(ef, dtype=z.dtype)

        inputs = {"x": z, "cond_image": cond, "encoder_hidden_states": encoder_hidden_states}
        if self.split == 'sample':
            inputs['video_name'] = row['video_name']
        return inputs