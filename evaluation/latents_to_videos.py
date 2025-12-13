import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import subprocess
from typing import Optional, Dict, List, Tuple
import json
import ast

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


def _save_frames(decoded_T3HW: torch.Tensor, out_dir: Path, grayscale: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    T = decoded_T3HW.shape[0]
    for t in range(T):
        frame = decoded_T3HW[t]  # (3, H, W) in [0,1]
        if grayscale:
            gray = (0.2989 * frame[0] + 0.5870 * frame[1] + 0.1140 * frame[2]).numpy()  # H, W
            arr = (gray*255.0).astype(np.uint8)
            Image.fromarray(arr, mode='L').save(out_dir / f"frame_{t}.png")
        else:
            arr = (frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            Image.fromarray(arr).save(out_dir / f"frame_{t}.png")


def _load_real_frames_T3HW(real_dir: Path, resize_to: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Load PNG frames from real_dir into a (T, 3, H, W) tensor in [0,1].

    If resize_to=(H,W) is provided, frames are resized to that size.
    """
    frame_paths = sorted([p for p in real_dir.glob('*.png')])
    frames: List[torch.Tensor] = []
    for p in frame_paths:
        img = Image.open(p).convert('RGB')
        if resize_to is not None:
            img = img.resize((int(resize_to[1]), int(resize_to[0])), Image.BICUBIC)  # PIL uses (W,H)
        arr = np.asarray(img).astype(np.float32) / 255.0  # H,W,3
        t = torch.from_numpy(arr).permute(2, 0, 1)  # 3,H,W
        frames.append(t)
    if len(frames) == 0:
        return torch.zeros((0, 3, *(resize_to if resize_to is not None else (0, 0))), dtype=torch.float32)
    return torch.stack(frames, dim=0)  # T,3,H,W


def _resample_T3HW(frames_T3HW: torch.Tensor, target_length: int) -> torch.Tensor:
    """Resample or pad frames to target_length using EchoDataset.resample_sequence policy.

    - If T < target_length: pad with zeros at the end.
    - Else: pick indices = round(linspace(0, T-1, target_length)).
    """
    T = frames_T3HW.shape[0]
    if T == target_length:
        return frames_T3HW
    if T < target_length:
        pad_T = target_length - T
        pad_block = torch.zeros((pad_T,) + tuple(frames_T3HW.shape[1:]), dtype=frames_T3HW.dtype)
        return torch.cat([frames_T3HW, pad_block], dim=0)
    idx = torch.linspace(0, T - 1, target_length).round().long()
    return frames_T3HW[idx]


def _make_video_with_ffmpeg(frames_dir: Path, out_path: Path, fps: float, grayscale: bool = False) -> bool:
    """Create a video from PNG frames using ffmpeg's glob pattern.

    Build a video from frames named frame_0.png, frame_1.png, ... (contiguous from 0).
    Returns True on success, False otherwise.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure there are frames to encode
    if len(list(frames_dir.glob("*.png"))) == 0:
        print(f"[warn] No PNG frames found in {frames_dir}, skipping video creation")
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", str((frames_dir / "frame_%d.png").as_posix())
    ]
    if grayscale:
        cmd += ["-vf", "format=gray"]

    cmd += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vsync", "cfr",
        "-r", str(fps),
        str(out_path),
    ]
    try:
        output = subprocess.run(cmd, capture_output=True, check=True)
        return True
    except FileNotFoundError:
        print('\n', subprocess.CalledProcessError)
        print("[warn] ffmpeg not found on PATH; skipping video creation")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[warn] ffmpeg failed for {frames_dir} -> {out_path}: {e}")
        return False


def _coerce_mask_list(mask_val, T_expected: Optional[int] = None) -> List[float]:
    """Coerce a DataFrame cell containing a mask (list-like or JSON string) into a Python list of floats.

    If T_expected is given and lengths differ, the shorter is padded with zeros or the longer is truncated.
    """
    lst = None
    if isinstance(mask_val, list):
        lst = mask_val
    elif isinstance(mask_val, (np.ndarray, tuple)):
        lst = list(mask_val)
    elif isinstance(mask_val, (str, bytes)):
        s = mask_val.decode() if isinstance(mask_val, bytes) else mask_val
        s = s.strip()
        # Try JSON first, then literal_eval
        try:
            lst = json.loads(s)
        except Exception:
            try:
                lst = ast.literal_eval(s)
            except Exception:
                raise ValueError(f"Cannot parse mask list from string: {s[:80]}...")
    else:
        raise ValueError(f"Unsupported mask value type: {type(mask_val)}")
    #TODO linspace
    lst = [float(x) for x in lst]
    if T_expected is not None and len(lst) != T_expected:
        if len(lst) < T_expected:
            lst = lst + [0.0] * (T_expected - len(lst))
        else:
            lst = lst[:T_expected]
    return lst


def _apply_mask_select_TCHW(frames_TCHW: torch.Tensor, mask_list: List[float], invert: bool = False) -> torch.Tensor:
    """Apply a per-frame binary mask to (T, C, H, W) and return only selected frames.

    If invert=True, use (1 - mask) to select frames.
    """
    T = frames_TCHW.shape[0]
    mask = torch.tensor(mask_list, dtype=frames_TCHW.dtype)
    if invert:
        mask = 1.0 - mask
    # Keep indices where mask > 0.5
    keep = mask > 0.5
    if keep.sum().item() == 0:
        return frames_TCHW[:0]  # empty tensor with correct dims
    return frames_TCHW[keep]

#TODO Test it works as expected after the addition of grayscale option
def convert_latents_directory(
    real_data_path: Path,
    latents_dir: Path,
    run_dir: Path,
    repo_id: str,
    query: dict,
    output_dir: Optional[Path] = None,
    types: Optional[List[str]] = None,
    fps: int = 16,
    fps_metadata_csv: Optional[Path] = None,
    decode_batch_size: int = 32,
    device: Optional[torch.device] = None,
    debugging: bool = False,
    grayscale: bool = False,
    test_n: Optional[int] = None
):
    """Convert latent videos saved by evaluate_to_latents to videos/frames.

    - latent_dir: directory containing *.pt files and metadata.csv
    - run_dir: Hydra run directory containing .hydra/config.yaml (to derive VAE subfolder)
    - repo_id: HF repo id for the VAE
    - output_dir: where to write outputs; defaults to latent_dir.parent / 'decoded_videos'
    - types: list of 'framewise' and/or 'videowise'
    - query: pandas query string to filter metadata.csv rows
        - fps_metadata_csv: optional CSV with video_name and fps columns to set per-video FPS.
            You can include the placeholder '{vae_res}', which will be replaced with cfg.vae.resolution.
    - decode_batch_size: number of frames decoded per VAE batch
    - device: torch device; auto-selected if None
    """
    latents_dir = Path(latents_dir)
    run_dir = Path(run_dir)
    default_output_dir = run_dir / 'evaluation'/ 'decoded_videos' / Path('/'.join(latents_dir.parts[-3:])) / '/'.join(query['name'].split('_'))
    output_dir = Path(output_dir) if output_dir else default_output_dir
    output_dir = output_dir / 'debugging' if debugging else output_dir
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
            df = df.query(query['pattern'])
        except Exception as e:
            raise ValueError(f"Invalid query '{query}': {e}")
        
    if test_n is not None:
        df = df.head(min(test_n, len(df)))

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

    # Build map of original_real_video_name to path to real video
    real_video_map = {}
    real_names = df['original_real_video_name'].unique()
    # Determine a data root to search for real frames
    for name in real_names:
        matches = [x for x in real_data_path.rglob(name) if x.is_dir()]
        assert len(matches) == 1, f"Expected one match for {name}, found {len(matches)}. \n{matches}"
        real_video_map[name] = matches[0]


    # Process each row
    for _, row in tqdm(df.iterrows(), total=len(df), unit=" latents", desc="Converting latents to videos"):
        video_name = str(row["video_name"]) if "video_name" in row else None
        if not video_name:
            continue
        video_fps = _get_fps(row['original_real_video_name'], fps, fps_map)
        real_video_name = row['original_real_video_name']
        real_video_path = real_video_map[real_video_name]

        pt_path = latents_dir / f"{video_name}.pt"
        if not pt_path.exists():
            print(f"[warn] Missing latent file: {pt_path}")
            continue

        # load latent video tensor
        data = torch.load(pt_path, map_location="cpu")
        latent = data["video"]

        # to (T, C, H, W)
        frames_TCHW = _to_TCHW(latent, latent_channels=vae.config.latent_channels)

        # Decode base video (fake)
        decoded_base = _decode_frames_with_vae(vae, frames_TCHW, batch_size=decode_batch_size)

        # Prepare REAL frames loaded from disk, resized to match fake HxW, and resampled to T
        # decoded_base is (T, 3, H, W)
        target_H, target_W = int(decoded_base.shape[2]), int(decoded_base.shape[3])
        real_T3HW_raw = _load_real_frames_T3HW(real_video_path, resize_to=(target_H, target_W))
        real_T3HW_resampled = _resample_T3HW(real_T3HW_raw, target_length=frames_TCHW.shape[0])

        # Frame masks
        not_pad_mask_list = _coerce_mask_list(row.get('not_pad_mask'), T_expected=frames_TCHW.shape[0])
        observed_mask_list = _coerce_mask_list(row.get('observed_mask'), T_expected=frames_TCHW.shape[0])

        if "framewise" in types:
            out_frames_dir_base_fake = output_dir / "framewise" / "fake" / video_name
            out_frames_dir_base_real = output_dir / "framewise" / "real" / real_video_name
            _save_frames(decoded_base, out_frames_dir_base_fake, grayscale=grayscale)
            _save_frames(real_T3HW_resampled, out_frames_dir_base_real, grayscale=grayscale)

            if "videowise" in types:
                # base
                out_video_path = (output_dir / "videowise" / video_name).with_suffix(".mp4")
                _make_video_with_ffmpeg(out_frames_dir_base_fake, out_video_path, fps=video_fps, grayscale=grayscale)

        # Stitched video: use stored stitched latents if available
        stitched_T3HW = None
        if ('framewise_stitched' in types) and "stitched_video" in data:
            stitched_latent = data["stitched_video"]
            stitched_TCHW = _to_TCHW(stitched_latent, latent_channels=vae.config.latent_channels)
            stitched_T3HW = _decode_frames_with_vae(vae, stitched_TCHW, batch_size=decode_batch_size)

            out_frames_dir_stitched_fake = output_dir / "framewise_stitched" / "fake" / video_name
            _save_frames(stitched_T3HW, out_frames_dir_stitched_fake, grayscale=grayscale)

            # Real counterpart for stitched: real frames with padding removed (not_pad_mask > 0.5)
            keep = torch.tensor(not_pad_mask_list, dtype=torch.float32) > 0.5
            real_stitched = real_T3HW_resampled[keep]
            out_frames_dir_stitched_real = output_dir / "framewise_stitched" / "real" / real_video_name
            _save_frames(real_stitched, out_frames_dir_stitched_real, grayscale=grayscale)

            if "videowise_stitched" in types:
                out_video_path_stitched = (output_dir / "videowise_stitched" / video_name).with_suffix(".mp4")
                _make_video_with_ffmpeg(out_frames_dir_stitched_fake, out_video_path_stitched, fps=video_fps, grayscale=grayscale)

        if "framewise_no_pad" in types:
            # No-pad video: mask keep of not_pad_mask
            decoded_no_pad = _apply_mask_select_TCHW(decoded_base, not_pad_mask_list, invert=False)

            #no_pad_TCHW = _apply_mask_select_TCHW(frames_TCHW, not_pad_mask_list, invert=False)
            #decoded_no_pad = _decode_frames_with_vae(vae, no_pad_TCHW, batch_size=decode_batch_size) if no_pad_TCHW.shape[0] > 0 else no_pad_TCHW.new_zeros((0, 3,)+tuple(frames_TCHW.shape[-2:]))

            out_frames_dir_no_pad_fake = output_dir / "framewise_no_pad" / "fake" / video_name
            _save_frames(decoded_no_pad, out_frames_dir_no_pad_fake, grayscale=grayscale)
            # Real counterpart: resampled real then apply not_pad_mask keep
            keep_np = torch.tensor(not_pad_mask_list, dtype=torch.float32) > 0.5
            real_no_pad = real_T3HW_resampled[keep_np]
            out_frames_dir_no_pad_real = output_dir / "framewise_no_pad" / "real" / real_video_name
            _save_frames(real_no_pad, out_frames_dir_no_pad_real, grayscale=grayscale)

            if "videowise_no_pad" in types:
                out_video_path_no_pad = (output_dir / "videowise_no_pad" / video_name).with_suffix(".mp4")
                _make_video_with_ffmpeg(out_frames_dir_no_pad_fake, out_video_path_no_pad, fps=video_fps, grayscale=grayscale)
        if "framewise_generated" in types:
            # Generated-only video: mask keep of NOT observed_mask
            
            # Gets the generated frames but does not include padded frames i.e., (not observed) minus (padded)
            generated_mask_list = [(1.- x)-(1. - y) for x, y in zip(observed_mask_list, not_pad_mask_list)]
            decoded_generated = _apply_mask_select_TCHW(decoded_base, generated_mask_list, invert=False)

            #generated_TCHW = _apply_mask_select_TCHW(frames_TCHW, generated_mask_list, invert=False)
            #decoded_generated = _decode_frames_with_vae(vae, generated_TCHW, batch_size=decode_batch_size) if generated_TCHW.shape[0] > 0 else generated_TCHW.new_zeros((0, 3,)+tuple(frames_TCHW.shape[-2:]))
            
            out_frames_dir_generated_fake = output_dir / "framewise_generated" / "fake" / video_name
            _save_frames(decoded_generated, out_frames_dir_generated_fake, grayscale=grayscale)
            # Real counterpart: resampled real then apply NOT observed_mask (hidden frames)
            keep_gen = torch.tensor(generated_mask_list, dtype=torch.float32) >= 0.5
            real_generated = real_T3HW_resampled[keep_gen]
            out_frames_dir_generated_real = output_dir / "framewise_generated" / "real" / real_video_name
            _save_frames(real_generated, out_frames_dir_generated_real, grayscale=grayscale)

            if "videowise_generated" in types:
                # generated
                out_video_path_generated = (output_dir / "videowise_generated" / video_name).with_suffix(".mp4")
                _make_video_with_ffmpeg(out_frames_dir_generated_fake, out_video_path_generated, fps=video_fps, grayscale=grayscale)

    df.to_csv(output_dir / 'metadata.csv', index=False)
    print(f"[done] Outputs saved under: {output_dir}")
    return output_dir


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
