import pandas as pd

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import torch
from omegaconf import OmegaConf

from utils.util import select_device
from trainer import load_model as build_model_from_cfg, load_flow


def _infer_data_shape(cfg):
	"""Infer (T, C, H, W) from cfg.dataset.shape or by reading the first latent .pt.

	Expects cfg.dataset.path to contain a metadata.csv with column 'video_name' and
	corresponding <video_name>.pt files with key 'mu' shaped (T, C, H, W).
	"""
	# If provided in config, use it
	shape = cfg.dataset.get("shape", None)
	if shape is not None:
		return shape

	data_path = Path(cfg.dataset.path)
	meta_path = data_path / "metadata.csv"

	df = pd.read_csv(meta_path)
	pt_path = data_path / f"{df.iloc[0]['video_name']}.pt"
	
	stats = torch.load(pt_path, map_location="cpu")
	mu = stats["mu"]
	return mu.shape 


def _find_checkpoint(ckpt_dir: Path, ckpt_name: Optional[str] = None) -> Path:
	if ckpt_name:
		p = ckpt_dir / ckpt_name
		if not p.exists():
			raise FileNotFoundError(f"Checkpoint not found: {p}")
		return p

	# Prefer last.ckpt if present
	last = ckpt_dir / "last.ckpt"
	if last.exists():
		return last

	# Else pick the most recently modified .ckpt
	candidates = list(ckpt_dir.glob("*.ckpt"))
	if not candidates:
		raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
	return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model_from_run(run_dir: str | Path, ckpt_name: Optional[str] = None):
	"""
	Load config and checkpoint weights for a trained run.

	Args:
		run_dir: Path to a Hydra run directory containing '.hydra/config.yaml' and 'checkpoints/*.ckpt'.
		device: Optional explicit device (e.g., 'cuda', 'mps', 'cpu'). If None, auto-selects.
		ckpt_name: Optional checkpoint filename to load; defaults to last.ckpt or the newest .ckpt in 'checkpoints'.

	Returns:
		model: The flow-wrapped model with weights loaded and moved to 'device'.
		cfg: The OmegaConf configuration loaded from the run.
		ckpt_path: The resolved checkpoint path used.
	"""
	run_dir = Path(run_dir)
	hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
	ckpt_dir = run_dir / "checkpoints"

	if not hydra_cfg_path.exists():
		raise FileNotFoundError(f"Hydra config not found at {hydra_cfg_path}")
	if not ckpt_dir.exists():
		raise FileNotFoundError(f"Checkpoints directory not found at {ckpt_dir}")

	cfg = OmegaConf.load(hydra_cfg_path)
	device = select_device()

	data_shape = _infer_data_shape(cfg)
	base_model = build_model_from_cfg(cfg, data_shape, device)
	model = load_flow(cfg, base_model).to(device)

	ckpt_path = _find_checkpoint(ckpt_dir, ckpt_name)
	ckpt = torch.load(ckpt_path, map_location=device)
	state_dict = ckpt.get("state_dict", ckpt)  # support raw SD too

	# FlowVideoGenerator saved as LightningModule with attribute 'model'
	# Strip leading 'model.' to match our flow wrapper keys
	cleaned = {k.split("model.", 1)[1] if k.startswith("model.") else k: v for k, v in state_dict.items()}

	# Load only matching keys
	missing, unexpected = model.load_state_dict(cleaned, strict=False)
	if missing:
		print(f"[load_model_from_run] Missing keys: {len(missing)} (showing up to 5): {missing[:5]}")
	if unexpected:
		print(f"[load_model_from_run] Unexpected keys: {len(unexpected)} (showing up to 5): {unexpected[:5]}")

	model.eval()
	return model, cfg, ckpt_path

# Sort out evaluation from models on cluster that have trained more. 


def stitch_video(input_video: torch.Tensor,
                 output_video: torch.Tensor,
                 observed_mask: list[float],
                 not_pad_mask: list[float]) -> torch.Tensor:
    """
    Combine observed frames from input_video with generated frames from output_video,
    removing padded frames.
    Args:
        input_video:  [C, T, H, W] tensor
        output_video: [C, T, H, W] tensor (same shape as input_video)
        observed_mask: list of length T with 1.0 (observed) or 0.0 (masked)
        not_pad_mask: list of length T with 1.0 (real) or 0.0 (padding)

    Returns:
        stitched_video: [C, T_real, H, W] tensor (padding removed)
    """
    C, T, H, W = input_video.shape

    # Convert masks to torch tensors and broadcast to [1, T, 1, 1]
    observed_mask_t = torch.tensor(
        observed_mask, dtype=torch.float32, device=input_video.device
	).view(1, T, 1, 1)

    not_pad_mask_t  = torch.tensor(
        not_pad_mask, dtype=torch.float32, device=input_video.device
	).view(1, T, 1, 1)

    # Compute stitched video
    stitched = (
        input_video * (observed_mask_t * not_pad_mask_t) +
        output_video * ((1.0 - observed_mask_t) * not_pad_mask_t)
    )

    # Drop padding frames
    keep_indices = torch.nonzero(not_pad_mask_t.view(-1), as_tuple=False).squeeze(-1)
    stitched = stitched[:, keep_indices, :, :]

    return stitched



@torch.no_grad()
def evaluate_to_latents(model, test_dl_list, run_cfg, eval_cfg, device):
	"""Evaluate model on list of DataLoaders and save latent videos and metadata.csv."""
	# Setup output directory with timestamp
	date, time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S").split("_") 
	latents_dir = Path(eval_cfg.output_dir) / 'latents' / date / time
	latents_dir.mkdir(parents=True, exist_ok=True)

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
					input_video=reference_batch['cond_image'].cpu()/run_cfg.vae.scaling_factor,
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