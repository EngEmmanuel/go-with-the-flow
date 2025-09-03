import pandas as pd

from pathlib import Path
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
