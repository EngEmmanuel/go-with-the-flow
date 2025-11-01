import sys
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import cv2
import numpy as np
import pandas as pd

sys.path.append(Path(__file__).parent.parent.as_posix())
from vae.util import load_vae_and_processor
from utils import select_device






# input directory contains many .pt files of sampled videos
# Each .pt file contains a dictionary like:
# sample_results = {
#   '0': {
#       "video_name": reference_batch['video_name'],
#       "cond_image": reference_batch['cond_image'],            # latent video (C,T,H,W) or (T,C,H,W)
#       'reconstructed': (sampled_videos[0,...], ef_value),      # latent video, scalar ef in [0,1]
#       'generated': (sampled_videos[1:,...], [ef1, ef2, ...])   # latent videos (N,C,T,H,W), list of ef values
#   },
#   '1': {...},
#   ...
# }
#
# Notes:
# - latent videos are per-frame latents for a 2D VAE (decode each frame independently)
# - We will save outputs as .mp4 videos using OpenCV


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _ef_to_tag(ef: Union[float, int]) -> str:
    """Format EF float to a safe filename tag like 0p75 for 0.75."""
    try:
        ef_f = float(ef)
    except Exception:
        return str(ef)
    return f"{ef_f:.2f}".replace(".", "p")


def _device_from_vae(vae: torch.nn.Module) -> torch.device:
    return next(vae.parameters()).device


def _to_TCHW(latent: torch.Tensor, latent_channels: int = None) -> torch.Tensor:
    """Convert latent video to shape (T, C, H, W) from (C,T,H,W) or already (T,C,H,W)."""
    #if latent.dim() != 4:
    #    raise ValueError(f"Expected 4D latent video (C,T,H,W) or (T,C,H,W), got shape {tuple(latent.shape)}")

    if latent.shape[0] == latent_channels:
        return latent.permute(1, 0, 2, 3).contiguous()  # (T, C, H, W)
    
    # Else assume it's already (T, C, H, W)
    return latent


def _decode_frames_with_vae(vae: torch.nn.Module, frames_TCHW: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
    """Decode frames with VAE.
    frames_TCHW: (T, C, H, W) latent frames
    returns: (T, 3, H', W') float in [0,1]
    """
    device = _device_from_vae(vae)
    T = frames_TCHW.shape[0]
    outs: List[torch.Tensor] = []
    for i in range(0, T, batch_size):
        chunk = frames_TCHW[i:i+batch_size].to(device)
        with torch.no_grad():
            imgs = vae.decode(chunk).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        outs.append(imgs.detach().cpu())
    return torch.cat(outs, dim=0)


def _save_video_cv2(video_T3HW: torch.Tensor, out_path: Path, fps: int = 16) -> Path:
    """Save a tensor video (T, 3, H, W) in [0,1] to a video file.
    Tries MP4 (mp4v) first; if unavailable, falls back to AVI (XVID).
    Returns the actual file path used (extension may change if fallback).
    """
    _ensure_dir(out_path.parent)
    T, C, H, W = video_T3HW.shape
    assert C == 3, f"Expected 3 channels after decoding, got {C}"
    second_type = ('.mp4', 'mp4v')
    first_type = ('.avi', 'XVID')

    # Helper to attempt a writer
    def _try_writer(path: Path, fourcc_code: str):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(W), int(H)))
        return writer

    # First try MP4
    writer = _try_writer(out_path.with_suffix(first_type[0]), first_type[1])
    used_path = out_path.with_suffix('.mp4')

    if not writer.isOpened():
        # Fallback to AVI
        if writer is not None:
            writer.release()
        writer = _try_writer(out_path.with_suffix(second_type[0]), second_type[1])
        used_path = out_path.with_suffix('.avi')

    if not writer.isOpened():
        if writer is not None:
            writer.release()
        raise RuntimeError("Failed to open a video writer. Your OpenCV build may lack codecs for MP4/AVI.")

    try:
        for t in range(T):
            frame = video_T3HW[t]
            frame_np = (frame.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # HWC RGB
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()

    return used_path


def _decode_and_save_single_video(vae: torch.nn.Module, latent_video: torch.Tensor, out_path: Path, fps: int = 16) -> Path:
    frames_TCHW = _to_TCHW(latent_video, latent_channels=vae.config.latent_channels)
    decoded_T3HW = _decode_frames_with_vae(vae, frames_TCHW)
    return _save_video_cv2(decoded_T3HW, out_path, fps=fps)


def decode_sample_results_file(
    pt_path: Union[str, Path],
    output_root: Union[str, Path],
    vae: torch.nn.Module,
    fps: int = 16,
    fps_map: Optional[Dict[str, float]] = None
) -> Path:
    """Decode one .pt file and save outputs under output_root / <pt_stem> / <video_name>.
    Returns the directory created for this .pt file.
    """
    pt_path = Path(pt_path)
    output_root = Path(output_root)

    sample_root = output_root / pt_path.stem
    _ensure_dir(sample_root)

    data: Dict[str, Any] = torch.load(pt_path, map_location='cpu')

    if not isinstance(data, dict):
        raise ValueError(f"File {pt_path} did not contain a dict as expected.")

    def get_fps_for(video_name: str) -> float:
        if fps_map is not None and video_name in fps_map:
            return float(fps_map[video_name])
        return float(fps)

    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue

        video_name = entry.get('video_name', f"sample_{key}")
        dest_dir = sample_root / str(video_name)
        _ensure_dir(dest_dir)
        target_fps = get_fps_for(str(video_name))

        # 1) cond_image video
        cond_latent = entry.get('cond_image', None)
        if isinstance(cond_latent, torch.Tensor):
            out_path = dest_dir / 'cond_image'
            _decode_and_save_single_video(vae, cond_latent, out_path, fps=target_fps)

        # 2) reconstructed: (latent_video, ef)
        reconstructed = entry.get('reconstructed', None)
        if isinstance(reconstructed, (list, tuple)) and len(reconstructed) == 2:
            rec_latent, rec_ef = reconstructed
            if isinstance(rec_latent, torch.Tensor):
                ef_tag = _ef_to_tag(rec_ef)
                out_path = dest_dir / f'reconstructed_ef{ef_tag}'
                _decode_and_save_single_video(vae, rec_latent, out_path, fps=target_fps)

        # 3) generated: (latent_videos, [ef...])
        generated = entry.get('generated', None)
        if isinstance(generated, (list, tuple)) and len(generated) == 2:
            gen_latents, gen_efs = generated
            if isinstance(gen_latents, torch.Tensor):
                # gen_latents expected shape (N, C, T, H, W) or (N, T, C, H, W)
                if gen_latents.dim() == 5:
                    # Normalize to (N, T, C, H, W)
                    if gen_latents.shape[1] <= 8:  # (N, C, T, H, W)
                        gen_latents = gen_latents.permute(0, 2, 1, 3, 4).contiguous()
                    N = gen_latents.shape[0]
                    if isinstance(gen_efs, (list, tuple)):
                        ef_list = list(gen_efs)
                    else:
                        ef_list = [gen_efs] * N
                    if len(ef_list) != N:
                        if len(ef_list) > N:
                            ef_list = ef_list[:N]
                        else:
                            ef_list = ef_list + [ef_list[-1]] * (N - len(ef_list))

                    for i in range(N):
                        ef_tag = _ef_to_tag(ef_list[i])
                        out_path = dest_dir / f'generated_ef{ef_tag}'
                        # gen_latents[i] is (T, C, H, W) now
                        _decode_and_save_single_video(vae, gen_latents[i], out_path, fps=target_fps)

    return sample_root


def decode_directory(
    directory_path: Union[str, Path],
    vae_locator: str,
    subfolder: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    fps: int = 16,
    metadata_csv_path: Optional[Union[str, Path]] = None
) -> Path:
    """Decode all .pt files in a directory and save videos under sibling 'decoded_sample_videos'.

    directory_path: folder with .pt files
    vae_locator: local path or HF repo id for the VAE
    subfolder: optional subfolder for HF repo
    device: torch device or string ('cuda', 'cpu', etc.)
    fps: default frames per second for saved videos when metadata not provided
    metadata_csv_path: optional CSV with columns including 'video_name' and 'FrameRate' (or 'frame_rate'/'fps')
    """
    directory_path = Path(directory_path)
    assert directory_path.is_dir(), f"Not a directory: {directory_path}"

    # Resolve device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    vae, _ = load_vae_and_processor(vae_locator, subfolder=subfolder, device=device)

    # Optional FPS map from metadata
    fps_map: Optional[Dict[str, float]] = None
    if metadata_csv_path is not None:
        csv_path = Path(metadata_csv_path)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Try to find the correct FPS column
            fps_col_candidates = [c for c in df.columns if c.lower() in ('framerate', 'frame_rate', 'fps')]
            if 'video_name' in df.columns and fps_col_candidates:
                fps_col = fps_col_candidates[0]
                fps_map = {str(row['video_name']): float(row[fps_col]) for _, row in df.iterrows()}
                print(f"Loaded FPS for {len(fps_map)} videos from metadata: {csv_path}")
            else:
                print(f"Warning: Could not find 'video_name' and FPS column in {csv_path}. Using default fps={fps}.")
        else:
            print(f"Warning: Metadata CSV not found: {csv_path}. Using default fps={fps}.")

    output_root = directory_path.parent / 'decoded_sample_videos'
    _ensure_dir(output_root)

    pt_files = sorted(directory_path.glob('*.pt'))
    for pt in pt_files:
        print(f"Decoding {pt.name} ...")
        decode_sample_results_file(pt, output_root, vae, fps=fps, fps_map=fps_map)

    print(f"All decoded videos saved under: {output_root}")
    return output_root


if __name__ == "__main__":
    device = select_device()
    print("Using device:", device)
    vae_res = '4f8'

    decode_directory(
        directory_path="outputs/hydra_outputs/2025-09-02/16-54-08/sample_videos",
        vae_locator="HReynaud/EchoFlow",
        subfolder=f"vae/avae-{vae_res}",
        device=device,
        metadata_csv_path=f"data/CAMUS_Latents_{vae_res}/metadata.csv"
    )

