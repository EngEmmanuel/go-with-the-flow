import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import zoom

from utils.video_utils import load_video

class CachedVideoLoader():
    def __init__(self, path, deactivate=False):
        self.path = path
        self.video = {}
        self.deactivate = deactivate

    def __call__(self, fname):
        if self.deactivate:
            return loadvideo(os.path.join(self.path, fname)).astype(np.uint8) # type: ignore
        if not fname in self.video:
            self.video[fname] = loadvideo(os.path.join(self.path, fname)).astype(np.uint8) # type: ignore
        return self.video[fname]


class EchoVideoEF(Dataset):
    def __init__(self, config, splits=["TRAIN", "VAL", "TEST"], fix_samples=False, limit=-1):

        self.config = config
        self.splits = splits if isinstance(splits, list) else [splits]
        self.data_path = config.dataset.data_path
        self.sample_duration = config.dataset.get("seconds", 1)
        self.target_fps = config.dataset.get("target_fps", 12)
        self.videos_length = self.target_fps * self.sample_duration
        self.fix_samples = fix_samples
        self.limit = limit

        self.filelist = pd.read_csv(os.path.join(config.dataset.data_path, "FileList.csv"))

        new_fl = self.filelist[self.filelist['Split'] == ""]
        for split in self.splits:
            new_fl = pd.concat((new_fl, self.filelist[self.filelist['Split'] == split.upper()])) # type: ignore
        self.filelist = new_fl

        self.video_folder_path = os.path.join(config.dataset.data_path, "Videos")

        # filter out videos that are not in the video folder
        self.fnames = [f+".avi" if not f.endswith(".avi") else f for f in self.filelist["FileName"].tolist()]
        self.fnames = [f for f in self.fnames if os.path.exists(os.path.join(self.video_folder_path, f))]

        if self.limit > 0 and self.limit < len(self.fnames):
            self.fnames = np.random.choice(self.fnames, self.limit, replace=False)

        fps_dict = {f: v for f, v in zip(self.filelist["FileName"].tolist(), self.filelist["FPS"].tolist())}
        self.fps = [fps_dict[name[:-4]] for name in self.fnames]
        lvef_dict = {f: float(v) for f, v in zip(self.filelist["FileName"].tolist(), self.filelist["EF"].tolist())}
        self.lvef = [lvef_dict[name[:-4]] for name in self.fnames]

        if "VAL" in splits:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: torch.Tensor(x/255.0)),
                transforms.Resize((config.dataset.image_size,)*2),
            ])
        elif "TRAIN" in splits:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: torch.Tensor(x/255.0)),
                transforms.Pad(12),
                transforms.RandomCrop(112),
                transforms.Resize((int(config.dataset.image_size))),
            ])
        
        self.lazy_vloader = CachedVideoLoader(self.video_folder_path, deactivate=config.dataset.deactivate_cache)

        print(f"Loaded {len(self.fnames)} videos for {self.splits} split")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i, frame_idx=None):

        fname = self.fnames[i]
        video = self.lazy_vloader(fname)
        total_frames = video.shape[1]

        if frame_idx is None and self.fix_samples:
            frame_idx = 0

        if self.config.dataset.get("resample_to_num_frames_fps", False) == True:
            fps = self.fps[i]
            video = zoom(video, (1, self.target_fps/fps, 1, 1), order=1)

            if video.shape[1] < self.videos_length: # if the video is too short, pad it with zeros
                video = np.concatenate([video, np.zeros((3, self.videos_length-video.shape[1], 112, 112))], axis=1)
        
        if video.shape[1] == self.videos_length:
            frame_idx = 0
        else:
            frame_idx = np.random.randint(0, video.shape[1]-self.videos_length) if frame_idx==None else frame_idx 

        end_idx = frame_idx + self.videos_length
        frames = video[:, frame_idx:end_idx, :, :] # (3, T, 112, 112) -> (3, videos_length, 112, 112)
        
        frames = self.transform(frames) # Rescale and resize 

        lvef = self.lvef[i]

        return frames, lvef

# No need to normalise frame rate like they do

import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings

from torchvision import tv_tensors


# Import video loading utilities
from utils.video_utils import load_video, resample_sequence

class CAMUSVideoEF(Dataset):
    def __init__(self, cfg, splits=["TRAIN", "VAL", "TEST"], transform=None):
        """
        Simplified CAMUS Video Dataset for cardiac ultrasound analysis.
        Only loads video sequences and EF values (no masks).
        
        Args:
            metadata_csv: Path to CSV file containing dataset metadata
            data_root: Root directory containing video data
            channels: Number of output channels (1 for grayscale, 3 for RGB)
            target_frames: Number of frames to sample/pad to
            image_size: Target image size as (height, width) tuple
            transform: Optional custom transforms to apply
            preload_data: If True, preload all data into memory for faster training
        """
        self.root = Path(cfg.data_path)
        self.df = pd.read_csv(self.root / cfg.metadata_csv)
        self.df = self.df[self.df['split'].isin(splits)]  # Filter by splits
        self.channels = cfg.channels
        self.target_frames = cfg.target_frames
        self.image_size = cfg.image_size
        self.preload_data = cfg.preload_data

        # Cache for preloaded data if enabled
        self._data_cache = {} if self.preload_data else None

        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            train_transforms = [
                v2.Lambda(lambda x: x/255.0),  # Scale to [0, 1]
                # Mild geometric augmentation — random shift & scale
                v2.Resize((self.image_size, self.image_size), interpolation=v2.InterpolationMode.BILINEAR),
                v2.Pad(16),  # or 'constant'
                v2.RandomCrop((self.image_size, self.image_size)),
                v2.Normalize(mean=[cfg.data_mean] * self.channels,
                            std=[cfg.data_std] * self.channels) # assumes grayscale so cfg.data_x will be one element
            ]
            val_transforms = [
                v2.Lambda(lambda x: x / 255.0),  # Scale to [0, 1]
                v2.Resize(
                    [self.image_size] * 2,
                    interpolation=v2.InterpolationMode.BILINEAR
                ),
                v2.Normalize(
                    mean=[cfg.data_mean] * self.channels,
                    std=[cfg.data_std] * self.channels
                )
            ]
            self.transform = v2.Compose(train_transforms if "TRAIN" in splits else val_transforms)

        # Preload data if requested
        if self.preload_data:
            self._preload_all_data()

    def _preload_all_data(self):
        """Preload all video data into memory for faster training."""
        print(f"Preloading {len(self.df)} video samples into memory...")
        for i in range(len(self.df)):
            if i % 50 == 0:
                print(f"Preloaded {i}/{len(self.df)} samples...")
            
            row = self.df.iloc[i]
            vid_name = row.video_name
            seq_path = self.root / vid_name / f"{vid_name}.mp4"
            
            try:
                # Load and process video data only
                seq = load_video(seq_path, channels=self.channels)
                
                # Resample to target frames
                seq_F = resample_sequence(seq, target_length=self.target_frames)
                
                # Store in cache
                self._data_cache[i] = {
                    'seq': seq_F
                }
            except Exception as e:
                warnings.warn(f"Failed to preload sample {i} ({vid_name}): {str(e)}")
                continue
        
        print(f"Preloading complete. Loaded {len(self._data_cache)}/{len(self.df)} samples.")

    def _convert_to_tensor(self, seq_F):
        """
        Convert numpy array to tensor with proper channel dimensions.
        
        Args:
            seq_F: Video sequence array of shape (T, H, W) or (T, H, W, C)
            
        Returns:
            seq_tensor: Video tensor with proper channel dimensions
        """
        # Handle video tensor conversion
        if len(seq_F.shape) == 3:  # (T, H, W) -> add channel dimension
            if self.channels == 1:
                # Grayscale: (T, H, W) -> (T, 1, H, W)
                seq_tensor = torch.from_numpy(seq_F).float().unsqueeze(1)
            else:
                # Convert grayscale to multi-channel by replication
                seq_tensor = torch.from_numpy(seq_F).float().unsqueeze(1)
                seq_tensor = seq_tensor.repeat(1, self.channels, 1, 1)
        elif len(seq_F.shape) == 4:  # (T, H, W, C) -> (T, C, H, W)
            seq_tensor = torch.from_numpy(seq_F).float().permute(0, 3, 1, 2)
        else:
            # Already in correct format or unexpected shape
            seq_tensor = torch.from_numpy(seq_F).float()
        
        return seq_tensor

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        # Get row data (always needed for EF and video_name)
        row = self.df.iloc[i]
        
        # Use cached data if available
        if self.preload_data and i in self._data_cache:
            cached_data = self._data_cache[i]
            seq_F = cached_data['seq'].copy()  # Copy to avoid modifying cached data
        else:
            # Load data on-demand
            vid_name = row.video_name
            seq_path = self.root / vid_name / f"{vid_name}.mp4"

            # Load video only
            seq = load_video(seq_path, channels=self.channels)

            # Resample to target frames
            seq_F = resample_sequence(seq, target_length=self.target_frames)

        # Convert EF
        ef = row.EF / 100.0

        # Convert numpy array to tensor with proper channel dimensions
        seq_tensor = self._convert_to_tensor(seq_F)
        seq_tensor = tv_tensors.Video(seq_tensor)

        # Apply transforms if specified
        if self.transform is not None:
            seq_tensor = self.transform(seq_tensor)

        seq_tensor = seq_tensor.permute(1, 0, 2, 3)  # Change to (C, T, H, W) format
        ef_tensor = torch.tensor(ef, dtype=torch.float32)

        return seq_tensor, ef_tensor

