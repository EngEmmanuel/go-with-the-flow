import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import subprocess
from typing import Optional, Dict, List

import torch
import pandas as pd
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from datetime import datetime
from tqdm import tqdm

from utils import select_device
from vae.util import load_vae_and_processor
from vae.decode_latent_to_image import (
    _to_TCHW,
    _decode_frames_with_vae,
)

def query_to_filename(query: str) -> str:
    mapping = {
        '==': 'eq',
        '!=': 'neq',
        '>=': 'ge',
        '<=': 'le',
        '>': 'gt',
        '<': 'lt',
        '"': '',
        "'": '',
        ' ': '_',
    }
    for k, v in mapping.items():
        query = query.replace(k, v)
    return query


def _load_cfg(run_dir: Path):
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hydra config not found at {cfg_path}")
    return OmegaConf.load(cfg_path)


def _build_fps_map(metadata_csv_path: Optional[Path]) -> Optional[Dict[str, float]]:
    if metadata_csv_path is None:
        return None
    if not metadata_csv_path.exists():
        print(f"[warn] FPS metadata CSV not found: {metadata_csv_path}")
        return None
    df = pd.read_csv(metadata_csv_path)
    fps_col_candidates = [c for c in df.columns if c.lower() in ("framerate", "frame_rate", "fps")]
    if "video_name" not in df.columns or not fps_col_candidates:
        print(f"[warn] Could not find 'video_name' and FPS column in {metadata_csv_path}")
        return None
    fps_col = fps_col_candidates[0]
    return {str(row["video_name"]): float(row[fps_col]) for _, row in df.iterrows()}


def _get_fps(video_name: str, default_fps: int, fps_map: Optional[Dict[str, float]]) -> float:
    if fps_map is not None and video_name in fps_map:
        return float(fps_map[video_name])
    return float(default_fps)


def _save_frames(decoded_T3HW: torch.Tensor, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    T = decoded_T3HW.shape[0]
    for t in range(T):
        frame = decoded_T3HW[t]  # (3, H, W) in [0,1]
        arr = (frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"frame_{t}.png")


def _make_video_with_ffmpeg(frames_dir: Path, out_path: Path, fps: float) -> bool:
    """Create a video from PNG frames using ffmpeg's glob pattern.

    Returns True on success, False otherwise.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure there are frames to encode
    if len(list(frames_dir.glob("*.png"))) == 0:
        print(f"[warn] No PNG frames found in {frames_dir}, skipping video creation")
        return False
    # Use an absolute glob pattern so we don't rely on cwd
    input_pattern = str(frames_dir / "*.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except FileNotFoundError:
        print("[warn] ffmpeg not found on PATH; skipping video creation")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[warn] ffmpeg failed for {frames_dir} -> {out_path}: {e}")
        return False


def convert_latents_directory(
    latents_dir: Path,
    run_dir: Path,
    repo_id: str,
    output_dir: Optional[Path] = None,
    types: Optional[List[str]] = None,
    query: Optional[str] = None,
    fps: int = 16,
    fps_metadata_csv: Optional[Path] = None,
    decode_batch_size: int = 16,
    device: Optional[torch.device] = None,
):
    """Convert latent videos saved by evaluate_to_latents to videos/frames.

    - latent_dir: directory containing *.pt files and metadata.csv
    - run_dir: Hydra run directory containing .hydra/config.yaml (to derive VAE subfolder)
    - repo_id: HF repo id for the VAE (e.g., 'HReynaud/EchoFlow')
    - output_dir: where to write outputs; defaults to latent_dir.parent / 'decoded_videos'
    - types: list of 'framewise' and/or 'videowise'
    - query: pandas query string to filter metadata.csv rows
        - fps_metadata_csv: optional CSV with video_name and fps columns to set per-video FPS.
            You can include the placeholder '{vae_res}', which will be replaced with cfg.vae.resolution.
    - decode_batch_size: number of frames decoded per VAE batch
    - device: torch device; auto-selected if None
    """
    types = types or ["framewise", "videowise"]
    latents_dir = Path(latents_dir)
    run_dir = Path(run_dir)
    default_output_dir = run_dir / 'evaluation'/ 'decoded_videos' / Path('/'.join(latents_dir.parts[-2:])) / (query_to_filename(query) if query else 'all')
    output_dir = Path(output_dir) if output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cfg then VAE (subfolder derived from cfg)
    cfg = _load_cfg(run_dir)
    subfolder = f"vae/avae-{cfg.vae.resolution}"
    device = device or select_device()
    vae, _ = load_vae_and_processor(vae_locator=repo_id, subfolder=subfolder, device=device)

    # Load metadata and filter
    meta_path = latents_dir / "metadata.csv"
    df = pd.read_csv(meta_path)
    if query:
        try:
            df = df.query(query)
        except Exception as e:
            raise ValueError(f"Invalid query '{query}': {e}")
    if df.empty:
        print("[info] No rows to process after filtering.")
        return output_dir


    # Build FPS map, optionally formatting a path template containing '{vae_res}'
    fps_csv_path = None
    if fps_metadata_csv:
        fps_csv_str = str(fps_metadata_csv)
        if "{vae_res}" in fps_csv_str:
            fps_csv_str = fps_csv_str.replace("{vae_res}", str(cfg.vae.resolution))
        fps_csv_path = Path(fps_csv_str)
    fps_map = _build_fps_map(fps_csv_path)

    # Process each row
    for _, row in tqdm(df.iterrows()):
        video_name = str(row["video_name"]) if "video_name" in row else None
        if not video_name:
            continue
        pt_path = latents_dir / f"{video_name}.pt"
        if not pt_path.exists():
            print(f"[warn] Missing latent file: {pt_path}")
            continue

        # load latent video tensor
        data = torch.load(pt_path, map_location="cpu")
        latent = data["video"]

        # to (T, C, H, W)
        frames_TCHW = _to_TCHW(latent)
        # decode to (T, 3, H', W') on CPU
        decoded_T3HW = _decode_frames_with_vae(vae, frames_TCHW, batch_size=decode_batch_size)

        # Always write framewise outputs (needed for ffmpeg video composition)
        out_frames_dir = output_dir / "framewise" / video_name
        _save_frames(decoded_T3HW, out_frames_dir)

        # Optionally compose a videowise mp4 using ffmpeg over the saved frames
        if "videowise" in types:
            out_video_path = (output_dir / "videowise" / video_name).with_suffix(".mp4")
            video_fps = _get_fps(row['original_real_video_name'], fps, fps_map)
            _make_video_with_ffmpeg(out_frames_dir, out_video_path, fps=video_fps)

    df.to_csv(output_dir / 'metadata.csv', index=False)
    print(f"[done] Outputs saved under: {output_dir}")
    return output_dir


'''
Convert latent outputs to videos/frames"
--latent_dir", type=str, help="Directory with latent .pt files and metadata.csv")
--run_dir", type=str, help="Hydra run dir with .hydra/config.yaml (used to derive VAE subfolder)")
--repo_id", type=str, help="HF repo id for the VAE, e.g., HReynaud/EchoFlow")
--output-dir", type=str, default=None, help="Output directory (default: <latent_dir>/../decoded_eval)")
--types", type=str, nargs="*", choices=["framewise", "videowise"], default=["framewise", "videowise"], help="Output types")
--query", type=str, default=None, help="pandas query to filter metadata rows")
--fps", type=int, default=16, help="Default FPS when metadata CSV is not provided")
--fps-metadata-csv", type=str, default=None, help="CSV with video_name and FPS column to override default FPS")
--decode-batch-size", type=int, default=16, help="Frames per decode batch")
'''


if __name__ == "__main__":
    #args = parse_args()
    date, time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S").split("_")
    run_dir = Path('outputs/hydra_outputs/2025-09-03/15-24-24')
    latent_dir = run_dir / 'evaluation' / 'latents' / '2025-09-09' /'16-31-27'
    output_dir = run_dir / 'evaluation' / 'decoded_videos' / date/ time

    device = select_device()
    query = 'rec_or_gen == "gen"'
    convert_latents_directory(
        latents_dir=latent_dir,
        run_dir=run_dir,
        repo_id="HReynaud/EchoFlow",
        output_dir=output_dir,
        types=["videowise"],
        query=query,
        fps_metadata_csv="data/CAMUS_Latents_{vae_res}/metadata.csv",
        device=device,
    )
