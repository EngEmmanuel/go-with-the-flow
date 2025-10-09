import torch
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from typing import Dict, List, Callable, Any, Optional

from dataset.echodataset import EchoDataset
from dataset.util import make_sampling_collate, default_eval_collate
from evaluation.functions import stitch_video

DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 2

# --- scheme functions (top-level) ---
# Each function reads its kwargs from eval_cfg.inference_schemes[<name>]

def ef_histogram_matching(run_cfg, eval_cfg) -> List[DataLoader]:
    """
    Build dataloaders that match the EF distribution specified in the 
    histogram matching plan CSV.
    """
    scheme_args = eval_cfg.inference_schemes.get("ef_histogram_matching", {})

    plan_path = scheme_args["plan_path"] 
    batch_size = scheme_args.get("batch_size", eval_cfg.get("batch_size", DEFAULT_BATCH_SIZE))
    num_workers = scheme_args.get("num_workers", eval_cfg.get("num_workers", DEFAULT_NUM_WORKERS))

    plan_df = pd.read_csv(plan_path)
    plan_df = (
        plan_df
        .drop_duplicates(subset=["video_name", "target_ef"])
        .loc[:, ["video_name", "target_ef", "target_ef_bin"]]
    )


    # Add new target_ef
    ds_list = []
    for nmf in eval_cfg.test_n_missing_frames:
        ds = EchoDataset(cfg=run_cfg, split="test", ef_column="target_ef", n_missing_frames=nmf, **eval_cfg.dataset_kwargs)
        # Bring target_ef and target_ef_bin into ds.df
        ds.df = ds.df.merge(
            plan_df, on="video_name", how="inner", validate="one_to_many"
        )

        ds_list.append(ds)

    dl_list = [DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=default_eval_collate) for ds in ds_list]
    return dl_list


def ef_samples_in_range(run_cfg, eval_cfg) -> List[DataLoader]:
    """
    Build dataloaders that use `make_sampling_collate(...)`.
    Expects eval_cfg.inference_schemes['ef_samples_in_range'] may contain:
      {"batch_size": optional, "num_workers": optional,
       "n_ef_samples": optional, "ef_gen_range": optional}
    """
    scheme_args = eval_cfg.inference_schemes.get("ef_samples_in_range", {})

    num_workers = scheme_args.get("num_workers", DEFAULT_NUM_WORKERS)

    n_ef_samples = scheme_args.get("n_ef_samples")
    ef_gen_range = scheme_args.get("ef_gen_range")

    ds_list = [
        EchoDataset(cfg=run_cfg, split="test", cache=False, n_missing_frames=nmf, **eval_cfg.dataset_kwargs)
        for nmf in eval_cfg.test_n_missing_frames
    ]

    collate_fn = make_sampling_collate(n_ef_samples, ef_gen_range=ef_gen_range)

    dl_list = [
        DataLoader(ds, batch_size=1, num_workers=num_workers, collate_fn=collate_fn)
        for ds in ds_list
    ]
    return dl_list



@torch.no_grad()
def inference_ef_samples_in_range(model, dl_list, run_cfg, eval_cfg, device, latents_dir):
    """Evaluate model on list of DataLoaders and save latent videos and metadata.csv."""
    batch_size = eval_cfg.inference_schemes.ef_samples_in_range.n_ef_samples

    metadata_df_rows = []
    for dl in tqdm(dl_list, total=len(dl_list), desc=f"DataLoaders: {inference_ef_samples_in_range.__name__}"):
        data_shape = tuple(dl.dataset[0]['x'].shape)
        nmf = dl.dataset.kwargs['n_missing_frames']
        nmf = f"{str(int(100*nmf))}p" if not isinstance(nmf, str) else nmf # 0.75 -> '75p', 'max' -> 'max'

        for batch in tqdm(dl, desc="Batches"):
            reference_batch, repeated_batch = batch
            repeated_batch = {k: v.to(device) for k, v in repeated_batch.items()}

            sample_videos = model.sample(
                **repeated_batch,
                batch_size=batch_size,
                data_shape=data_shape
            ) # (n_ef_samples_in_range + 1, C, T, H, W)
            sample_videos = sample_videos.detach().cpu()
            sample_videos /= run_cfg.vae.scaling_factor 
            
            for i, (ef, video) in enumerate(zip(reference_batch['ef_values'], sample_videos)):
                ef  = round(int(100*ef.item()),2)
                video_name = f"{reference_batch['video_name']}_ef{ef}_nmf{nmf}"
                metadata_df_rows.append({
                    'video_name': video_name,
                    'n_missing_frames': nmf,
                    'EF': ef,
                    'rec_or_gen': 'rec' if i == 0 else 'gen',
                    'original_real_video_name': reference_batch['video_name'],
                    'observed_mask': reference_batch['observed_mask'].tolist(),
                    'not_pad_mask': reference_batch['not_pad_mask'].tolist()
                })
                stitched_video = stitch_video(
                    input_video=reference_batch['cond_image'].cpu()[:4]/run_cfg.vae.scaling_factor,
                    output_video=video,
                    observed_mask=reference_batch['observed_mask'].tolist(),
                    not_pad_mask=reference_batch['not_pad_mask'].tolist()
                )
                torch.save(
                    obj = {
                        'video': video,
                        'stitched_video': stitched_video, # for compatibility with EchoDataset latents
                    },
                    f = latents_dir / f"{video_name}.pt"
                )
            
        
    df = pd.DataFrame(metadata_df_rows)
    df.to_csv(latents_dir / 'metadata.csv', index=False)
    return latents_dir


@torch.no_grad()
def inference_ef_histogram_matching(model, dl_list, run_cfg, device, latents_dir):
    """Evaluate model on list of DataLoaders and save latent videos and metadata.csv."""

    metadata_df_rows = []
    for dl in tqdm(dl_list, total=len(dl_list), desc=f"DataLoaders: {inference_ef_histogram_matching.__name__}"):
        batch_size = dl.batch_size
        data_shape = tuple(dl.dataset[0]['x'].shape)
        nmf = dl.dataset.kwargs['n_missing_frames']
        nmf = f"{str(int(100*nmf))}p" if not isinstance(nmf, str) else nmf # 0.75 -> '75p', 'max' -> 'max'

        for batch in tqdm(dl, desc="Batches"):
            reference_batch, input_batch = batch
            input_batch = {k: v.to(device) for k, v in input_batch.items()}

            sample_videos = model.sample(
                **input_batch,
                batch_size=batch_size,
                data_shape=data_shape
            ) # (B, C, T, H, W)
            sample_videos = sample_videos.detach().cpu()
            sample_videos /= run_cfg.vae.scaling_factor 

            for i in range(batch_size):
                video = sample_videos[i]
                ef = round(int(100*reference_batch['ef_values'][i].item()),2)
                video_name = f"{reference_batch['video_name'][i]}_ef{ef}_nmf{nmf}"
                metadata_df_rows.append({
                    'video_name': video_name,
                    'n_missing_frames': nmf,
                    'EF': ef,
                    'rec_or_gen': 'gen',
                    'original_real_video_name': reference_batch['video_name'][i],
                    'target_ef_bin': reference_batch['target_ef_bin'][i],
                    'observed_mask': reference_batch['observed_mask'][i].tolist(),
                    'not_pad_mask': reference_batch['not_pad_mask'][i].tolist(),
                })
                stitched_video = stitch_video(
                    input_video=reference_batch['cond_image'][i].cpu()[:4]/run_cfg.vae.scaling_factor,
                    output_video=video,
                    observed_mask=reference_batch['observed_mask'][i].tolist(),
                    not_pad_mask=reference_batch['not_pad_mask'][i].tolist()
                )
                torch.save(
                    obj = {
                        'video': video,
                        'stitched_video': stitched_video, # for compatibility with EchoDataset latents
                    },
                    f = latents_dir / f"{video_name}.pt"
                )


    df = pd.DataFrame(metadata_df_rows)
    df.to_csv(latents_dir / 'metadata.csv', index=False)
    return latents_dir


# -------- FUNCTIONS TO IMPORT AND RUN -------------------------

# --- registry mapping config keys to functions ---
SCHEME_REG: Dict[str, Callable[[Any, Any], List[DataLoader]]] = {
    "ef_histogram_matching": ef_histogram_matching,
    "ef_samples_in_range": ef_samples_in_range,
}


def generate_dls_for_evaluation_scheme(run_cfg, eval_cfg) -> Dict[str, List[DataLoader]]:
    """
    Build dataloaders for every scheme listed in eval_cfg.inference_schemes (or inference_scheme).
    Returns a dict mapping scheme_name -> list[DataLoader].
    """
    schemes = eval_cfg.inference_schemes
    if schemes is None:
        raise ValueError("eval_cfg must contain 'inference_schemes'")

    all_dataloaders = {}
    for scheme_name in schemes.keys():
        if scheme_name not in SCHEME_REG:
            raise ValueError(f"Unknown inference scheme: {scheme_name}. Known: {list(SCHEME_REG.keys())}")
        
        fn = SCHEME_REG[scheme_name]
        all_dataloaders[scheme_name] = fn(run_cfg, eval_cfg)

    return all_dataloaders

def run_inference(eval_cfg, run_cfg, model, dataloaders, device):
    """
    Run inference on model using dataloaders specified in eval_cfg.
    Returns a dict mapping scheme_name -> latents_dir (Path).
    """
    # Setup output directory with timestamp
    date, time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S").split("_") 
    latents_dir = Path(eval_cfg.output_dir) / 'latents' / date / time
    latents_dir.mkdir(parents=True, exist_ok=True)

    all_latents_dirs = {}
    for scheme_name, dl_list in dataloaders.items():
        kwargs = {
            'model': model,
            'dl_list': dl_list,
            'run_cfg': run_cfg,
            'device': device,
        }

        if scheme_name == 'ef_histogram_matching':
            print(f"[info] Running inference scheme: {scheme_name}")
            (latents_dir / scheme_name).mkdir(parents=True, exist_ok=True)
            inference_ef_histogram_matching(
                latents_dir=latents_dir / scheme_name,
                **kwargs
            )
        elif scheme_name == 'ef_samples_in_range':
            print(f"[info] Running inference scheme: {scheme_name}")
            (latents_dir / scheme_name).mkdir(parents=True, exist_ok=True)
            inference_ef_samples_in_range(
                latents_dir=latents_dir / scheme_name,
                eval_cfg=eval_cfg,
                **kwargs
            )


        all_latents_dirs[scheme_name] = latents_dir / scheme_name
        
    return all_latents_dirs