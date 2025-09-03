"""
Load a VAE (AutoencoderKL) and run reconstructions on images or videos (frame-by-frame).

The VAE locator (`--vae_dir`) may be either:
 - a local folder containing a HuggingFace-style VAE (config.json, safetensors, ...),
 - or a HuggingFace repo id (author/repo). When using a repo id, pass `--subfolder` if the VAE lives in a subfolder.

This script will:
 - load the VAE via `AutoencoderKL.from_pretrained(local_dir)` or `AutoencoderKL.from_pretrained(repo_id, subfolder=...)`
 - create a `VaeImageProcessor` from the VAE config when available
 - accept image globs or video paths and process videos frame-by-frame
 - run frames/images through the VAE and save reconstructions to `--out_dir`

Notes on shapes (torch tensors):
 - input images (after preprocessing) have shape: [B, C, H, W], dtype=float32, values in [0,1]
 - VAE.forward returns an object; this script tries `out.sample` which is typically image-space output
 - postprocessed output images are saved as PNGs

Usage examples:
    # local VAE folder, process images
    python load_vae_reconstruct.py --vae_dir store/weights/vae/avae-4f4 --inputs "images/*.png" --out_dir reconstructions --size 112

    # HF repo id with subfolder, process videos (frame-by-frame)
    python load_vae_reconstruct.py --vae_dir author/repo --subfolder vae/avae-4f4 --inputs "data/videos/*.mp4" --out_dir reconstructions --size 112

"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Optional, List

import torch
from PIL import Image
import torchvision.transforms as T

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from tqdm.auto import tqdm

# allow running this script from anywhere
sys.path.append(str(Path(__file__).resolve().parents[0]))
try:
    from utils.video_utils import load_video
except Exception:
    # fallback: load_video will not be available if run outside the repo; we'll still try to process images
    load_video = None


def select_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_vae_and_processor(vae_locator: str, subfolder: Optional[str], device: torch.device):
    """Load AutoencoderKL from a local folder or a HuggingFace repo id and return (vae, processor).

    vae_locator: local path or HF repo id (author/repo)
    subfolder: optional subfolder inside HF repo where VAE lives
    device: torch device
    """
    p = Path(vae_locator)
    if p.exists():
        print(f"Loading local VAE from: {vae_locator}")
        vae = AutoencoderKL.from_pretrained(str(p))
    else:
        print(f"Loading VAE from HuggingFace repo: {vae_locator}, subfolder={subfolder}")
        vae = AutoencoderKL.from_pretrained(vae_locator, subfolder=subfolder)

    vae = vae.to(device)
    vae.eval()

    processor = VaeImageProcessor.from_config(vae.config)
    print("Created VaeImageProcessor from config")

    return vae, processor


def preprocess_image_with_processor(img_pil: Image.Image, processor, size: int, device: torch.device):
    """Use VaeImageProcessor.preprocess if available; otherwise use a torchvision pipeline.

    Input: PIL.Image
    Output: torch.Tensor [1, C, H, W], dtype=float32, values in [0,1]
    """
    x = processor.preprocess(img_pil, height=size, width=size)


    # Make batch
    x = x.unsqueeze(0).to(device=device, dtype=torch.float32)
    return x


def postprocess_and_save(output_tensor: torch.Tensor, processor, out_path: Path):
    """Postprocess a model output tensor and save image.

    output_tensor: Tensor of shape [B, C, H, W] or [B, 1, H, W]
    """
    # Move to cpu and detach
    out = output_tensor.detach().cpu()

    imgs = processor.postprocess(out, output_type="pil")
    # processor.postprocess returns list of PIL images
    if isinstance(imgs, list) and len(imgs) > 0:
        imgs[0].save(str(out_path))
        return


def _is_video_file(p: str):
    ext = Path(p).suffix.lower()
    return ext in ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_dir", type=str, required=True, help="Local VAE folder or HuggingFace repo id (author/repo). If using HF repo, pass --subfolder")
    parser.add_argument("--subfolder", type=str, default=None, help="optional subfolder inside HF repo where the VAE lives, e.g. vae/avae-4f4")
    parser.add_argument("--inputs", type=str, required=True, help="Glob or comma-separated list of image/video paths, e.g. 'images/*.png' or 'videos/*.mp4' or 'img1.png,vid1.mp4'")
    parser.add_argument("--out_dir", type=str, default="reconstructions", help="Where to save reconstructed images (for videos saved per-frame under a folder)")
    parser.add_argument("--size", type=int, default=112, help="Resolution to resize frames before passing to VAE (H and W)")
    parser.add_argument("--device", type=str, default=None, help="device: cuda, mps, cpu (auto if not set)")
    parser.add_argument("--force_cpu", action="store_true", help="force use of CPU")
    args = parser.parse_args()

    device = select_device(force_cpu=args.force_cpu) if args.device is None else torch.device(args.device)
    print("Using device:", device)

    vae, processor = load_vae_and_processor(args.vae_dir, args.subfolder, device)

    # Resolve input paths (allow comma separated list or glob)
    if "," in args.inputs:
        input_list = [p.strip() for p in args.inputs.split(",")]
    else:
        input_list = sorted(glob.glob(args.inputs))

    if len(input_list) == 0:
        print("No inputs found for pattern:", args.inputs)
        return

    os.makedirs(args.out_dir, exist_ok=True)

    from tqdm.auto import tqdm as _tqdm

    with _tqdm(input_list, desc="Inputs", unit="file") as outer_bar:
        for p in outer_bar:
            pth = Path(p)
            # show current input name in outer bar description
            try:
                outer_bar.set_description(f"Input: {pth.stem}")
            except Exception:
                pass

            if pth.exists() and _is_video_file(str(pth)):
                if load_video is None:
                    print(f"Video support not available (cannot import utils.video_utils). Skipping: {p}")
                    continue

                frames = load_video(str(p), channels=3)  # numpy T,H,W,C uint8
                print(f" -> extracted {frames.shape[0]} frames")
                orig_dir = Path(args.out_dir) / 'original' / pth.stem
                recon_dir = Path(args.out_dir) / 'reconstruction' / pth.stem
                orig_dir.mkdir(parents=True, exist_ok=True)
                recon_dir.mkdir(parents=True, exist_ok=True)
                for i in tqdm(range(frames.shape[0]), desc=f"Video: {pth.stem}", unit="frame"):
                    frame = frames[i]
                    pil = Image.fromarray(frame)
                    # save original frame
                    orig_file = orig_dir / f"frame_{i+1}.png"
                    pil.save(orig_file)

                    x = processor.preprocess(pil, height=args.size, width=args.size).to(device)

                    #print(f"Input tensor shape -> {x.shape}")
                    with torch.no_grad():
                        out = vae(x)
                        if hasattr(out, 'sample'):
                            recon = out.sample
                        elif isinstance(out, (tuple, list)):
                            recon = out[0]
                        else:
                            recon = out

                    recon_file = recon_dir / f"frame_{i+1}.png"
                    postprocess_and_save(recon, processor, recon_file)
            else:
                # treat as image path
                if not pth.exists():
                    print(f"Input not found, skipping: {p}")
                    continue
                print(f"Processing image: {p}")
                img_pil = Image.open(str(pth)).convert("RGB")
                # create per-input dirs
                orig_dir = Path(args.out_dir) / 'original' / pth.stem
                recon_dir = Path(args.out_dir) / 'reconstruction' / pth.stem
                orig_dir.mkdir(parents=True, exist_ok=True)
                recon_dir.mkdir(parents=True, exist_ok=True)

                # save original as frame_1.png for consistency
                orig_file = orig_dir / "frame_1.png"
                img_pil.save(orig_file)

                x = preprocess_image_with_processor(img_pil, processor, args.size, device)
                print(f"Input tensor shape -> {x.shape}")
                with torch.no_grad():
                    out = vae(x)
                    if hasattr(out, 'sample'):
                        recon = out.sample
                    elif isinstance(out, (tuple, list)):
                        recon = out[0]
                    else:
                        recon = out

                recon_file = recon_dir / "frame_1.png"
                postprocess_and_save(recon, processor, recon_file)
                print("Saved original and reconstruction:", orig_file, recon_file)


if __name__ == "__main__":
    print(os.getcwd())
    main()
