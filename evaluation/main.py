import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from dataset.echodataset import EchoDataset
from evaluation.functions import load_model_from_run
from dataset.util import make_sampling_collate
from utils import select_device

device = select_device()

run_dir = Path('outputs/hydra_outputs/2025-09-03/15-24-24')
model, cfg, ckpt_path = load_model_from_run(run_dir, ckpt_name=None)

test_n_missing_frames = ['max', 0.75, 0.5, 0.25]  # Proportion of frames to mask
test_ef_gen_range = (0.1, 0.8)  # Range of ef_gen values to sample from uniformly
n_ef_samples_in_range = 4  # Number of ef_gen samples per video in ef_gen_range

test_ds_list = [
    EchoDataset(cfg, split="test", cache=False, n_missing_frames=nmf)
    for nmf in test_n_missing_frames
]
test_dl_list = [
    DataLoader(test_ds, batch_size=1,
               collate_fn=make_sampling_collate(n_ef_samples_in_range, ef_gen_range=test_ef_gen_range))
    for test_ds in test_ds_list
]

tstamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = run_dir / 'evaluation' / 'latents' / tstamp
output_dir.mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def evaluate_to_latents():
    metadata_df_rows = []
    for dl in tqdm(test_dl_list, total=len(test_dl_list), desc="DataLoaders"):
        nmf = dl.dataset.kwargs['n_missing_frames']
        nmf = f"{str(int(100*nmf))}p" if not isinstance(nmf, str) else nmf # 0.75 -> '75p', 'max' -> 'max'

        for batch in tqdm(dl, desc="Batches"):
            reference_batch, repeated_batch = batch
            batch_size, *data_shape = repeated_batch['cond_image'].shape

            repeated_batch = {k: v.to(device) for k, v in repeated_batch.items()}

            sample_videos = model.sample(
                **repeated_batch,
                batch_size=batch_size,
                data_shape=data_shape
            ) # (n_ef_samples_in_range + 1, C, T, H, W)
            sample_videos = sample_videos.detach().cpu()
            
            for i, (ef, video) in enumerate(zip(reference_batch['ef_values'], sample_videos)):
                ef  = round(int(100*ef.item()),2)
                video_name = f"{reference_batch['video_name']}_ef{ef}_nmf{nmf}"
                metadata_df_rows.append({
                    'video_name': video_name,
                    'n_missing_frames': nmf,
                    'EF': ef,
                    'rec_or_gen': 'rec' if i == 0 else 'gen',
                    'original_real_video_name': reference_batch['video_name']
                })
                torch.save(
                    obj = {'video': video},
                    f = output_dir / f"{video_name}.pt"
                )
        
            break
        break
    df = pd.DataFrame(metadata_df_rows)
    df.to_csv(output_dir / 'metadata.csv', index=False)

evaluate_to_latents()





