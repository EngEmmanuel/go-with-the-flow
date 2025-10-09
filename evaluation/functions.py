import pandas as pd

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import torch
from omegaconf import OmegaConf

from utils.util import select_device
from trainer import load_model, load_flow

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

def _get_run_config(run_dir: Path) -> OmegaConf:
	hydra_cfg_path = run_dir / ".hydra" / "config.yaml"

	if not hydra_cfg_path.exists():
		raise FileNotFoundError(f"Hydra config not found at {hydra_cfg_path}")
	
	cfg = OmegaConf.load(hydra_cfg_path)
	return cfg
	
def load_model_from_run(run_dir: str | Path, dummy_data: dict, ckpt_name: Optional[str] = None):
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
	cfg = _get_run_config(Path(run_dir))
	ckpt_dir = run_dir / "checkpoints"
	if not ckpt_dir.exists():
		raise FileNotFoundError(f"Checkpoints directory not found at {ckpt_dir}")


	device = select_device()

	base_model = load_model(cfg, dummy_data, device)
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
	return model, ckpt_path

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

