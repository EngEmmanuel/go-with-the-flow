from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from datetime import datetime
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity as skimage_ssim

# Masking no longer used; compute unmasked metrics only



def _to_tensor(img: Image.Image, device: Optional[torch.device] = None) -> torch.Tensor:
    x = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                          .view(img.size[1], img.size[0], len(img.getbands()))
                          .numpy().copy()) / 255.0).float()
    x = x.permute(2, 0, 1)  # CHW
    if device is not None:
        x = x.to(device)
    return x


def _load_image(path: Path, resize: Optional[Tuple[int, int]] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if resize is not None:
        w, h = resize[1], resize[0] if isinstance(resize, (list, tuple)) and len(resize) == 2 else resize
        # Accept resize given as (H, W)
        if isinstance(resize, (list, tuple)):
            h, w = resize
        img = img.resize((w, h), Image.BICUBIC)
    return _to_tensor(img, device)


@torch.no_grad()
def _ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """SSIM via scikit-image.structural_similarity.

    Expects img1/img2 as CHW in [0,1].
    """
    if skimage_ssim is None:
        raise ImportError("scikit-image is required for SSIM; install 'scikit-image'.")
    x = img1.permute(1, 2, 0).detach().cpu().numpy()  # HWC
    y = img2.permute(1, 2, 0).detach().cpu().numpy()  # HWC
    return float(skimage_ssim(x, y, data_range=1.0, channel_axis=2))


@torch.no_grad()
def _psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def _try_lpips_device(device: Optional[torch.device]):
    try:
        import lpips  # type: ignore
    except Exception:
        return None
    net = lpips.LPIPS(net='vgg')
    if device is not None:
        net = net.to(device)
    net.eval()
    return net


@torch.no_grad()
def _lpips_vgg(metric_net, img1: torch.Tensor, img2: torch.Tensor) -> float:
    # lpips expects [-1,1] range NCHW
    x = img1.unsqueeze(0) * 2 - 1
    y = img2.unsqueeze(0) * 2 - 1
    d = metric_net(x, y)
    return float(d.squeeze().detach().cpu().item())


def _conf_int(values: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    import statistics
    n = len(values)
    mean = statistics.fmean(values) if n > 0 else float('nan')
    if n <= 1:
        return mean, float('nan'), float('nan')
    stdev = statistics.stdev(values)
    se = stdev / math.sqrt(n)
    # Try Student's t; else fall back to normal z
    try:
        from scipy.stats import t as student_t  # type: ignore
        tcrit = float(student_t.ppf(1 - (1 - confidence) / 2.0, df=n - 1))
    except Exception:
        # 1.96 ~ 95% normal
        from math import erf, sqrt
        # crude mapping for common confidences
        tcrit = 1.96 if abs(confidence - 0.95) < 1e-6 else 1.0
    half_width = tcrit * se
    return mean, mean - half_width, mean + half_width

def get_real_and_fake_video_folders(
    real_root: Path,
    fake_root: Path,
    real_glob: str = '*',
    fake_glob: str = '*',
    regex_pattern: str = r"^patient\d{4}_(?:2CH|4CH)"
):
    # Pair video directories by common names
    real_videos = {vid_folder.name: vid_folder for vid_folder in real_root.glob(real_glob) if vid_folder.is_dir()}
    fake_videos = {vid_folder.name: vid_folder for vid_folder in fake_root.glob(fake_glob) if vid_folder.is_dir()}

    # Regex to capture patient####_<view>
    prefix_pattern = re.compile(regex_pattern)

    # Build prefix → fake_path dict
    fake_prefix_map = {}
    for fake_name, fake_path in fake_videos.items():
        m = prefix_pattern.match(fake_name)
        if not m:
            raise ValueError(f"Unexpected fake video name format: {fake_name}")
        prefix = m.group(0)
        if prefix in fake_prefix_map:
            raise ValueError(f"Multiple fake videos found with prefix {prefix}")
        fake_prefix_map[prefix] = fake_path

    # Build prefix → real_path dict
    real_prefix_map = {}
    for real_name, real_path in real_videos.items():
        m = prefix_pattern.match(real_name)
        if not m:
            raise ValueError(f"Unexpected real video name format: {real_name}")
        prefix = m.group(0)
        if prefix in real_prefix_map:
            raise ValueError(f"Multiple real videos found with prefix {prefix}")
        real_prefix_map[prefix] = real_path

    # Intersect keys to only keep common prefixes
    common_prefixes = set(real_prefix_map.keys()) & set(fake_prefix_map.keys())

    # Build paired_videos
    paired_videos = {
        prefix: (real_prefix_map[prefix], fake_prefix_map[prefix])
        for prefix in common_prefixes
    }

    # Report stats
    print(f"Total pairs: {len(paired_videos)}")
    print(f"Omitted from real_videos: {len(real_prefix_map) - len(common_prefixes)}")
    print(f"Omitted from fake_videos: {len(fake_prefix_map) - len(common_prefixes)}")
    return paired_videos


def compute_metrics_for_datasets(
    inference_root: Path,
    metrics: List[str],
    output_dir: Path,
    confidence: float = 0.95,
    resize: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
    real_glob: str = '*',
    fake_glob: str = '*',
) -> Dict[str, Path]:
    real_root = inference_root / 'real'
    fake_root = inference_root / 'fake'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paired_videos = get_real_and_fake_video_folders(
        real_root=real_root,
        fake_root=fake_root,
        real_glob=real_glob,
        fake_glob=fake_glob,
    )

    # Prepare optional LPIPS model
    want_lpips = any(m.lower() in ("lpips", "lpips_vgg") for m in metrics)
    lpips_net = _try_lpips_device(device) if want_lpips else None
    if want_lpips and lpips_net is None:
        print("[warn] lpips package not found; LPIPS(VGG) will be skipped or set to NaN.")

    # Save under fake_root/metric_results with a single YAML
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(fake_root).parent / 'metric_results'
    out_dir.mkdir(parents=True, exist_ok=True)
    result_yaml = out_dir / f"metrics_{ts}.yaml"

    rows = []
    for real_name, (rdir, fdir) in paired_videos.items():
        rframes = sorted(rdir.glob('*.png'))
        fframes = sorted(fdir.glob('*.png'))
        # Use overlapping frame count only
        n = min(len(fframes), len(rframes))
        if n == 0:
            continue

        # Buckets for metrics (only _all variant retained)
        ssim: List[float] = []
        psnr: List[float] = []
        lp: List[float] = []

        want_ssim = 'ssim' in [m.lower() for m in metrics]
        want_psnr = 'psnr' in [m.lower() for m in metrics]
        want_lp = want_lpips and lpips_net is not None

        for i in range(n):
            fp = fframes[i]
            rp = rframes[i]
            y = _load_image(fp, resize=resize, device=device)
            x = _load_image(rp, resize=resize, device=device)
            if x.shape != y.shape:
                h = min(x.shape[1], y.shape[1])
                w = min(x.shape[2], y.shape[2])
                x = F.interpolate(x.unsqueeze(0), size=(h, w), mode='bicubic', align_corners=False).squeeze(0)
                y = F.interpolate(y.unsqueeze(0), size=(h, w), mode='bicubic', align_corners=False).squeeze(0)

            if want_ssim:
                ssim.append(_ssim(x, y))
            if want_psnr:
                psnr.append(_psnr(x, y))
            if want_lp:
                lp.append(_lpips_vgg(lpips_net, x, y))

        row = {'video_pair': (rdir.name, fdir.name)}
        if ssim:
            row['ssim'] = round(float(sum(ssim) / len(ssim)), 4)
        if psnr:
            row['psnr'] = round(float(sum(psnr) / len(psnr)), 4)
        if want_lpips:
            row['lpips_vgg'] = round(float(sum(lp) / len(lp)), 4) if lp else float('nan')
        rows.append(row)

    per_video_df = pd.DataFrame(rows)

    # Summaries with CIs
    summary_rows = []
    metric_bases = [m.lower() for m in metrics]
    for base in metric_bases:
        col_base = 'lpips_vgg' if base in ('lpips', 'lpips_vgg') else base
        col = col_base
        if col in per_video_df.columns:
            vals = per_video_df[col].dropna().tolist()
            mean, lo, hi = _conf_int(vals, confidence)
            summary_rows.append({'metric': col, 'mean': round(mean, 4), 'ci_low': round(lo, 4), 'ci_high': round(hi, 4), 'n_videos': len(vals)})

    # Prepare YAML payload
    payload: Dict = {
        'config': {
            'real_root': str(real_root),
            'fake_root': str(fake_root),
            'real_glob': real_glob,
            'fake_glob': fake_glob,
            'which': list(metrics),
            'confidence': float(confidence),
            'resize': list(resize) if resize is not None else None,
            'timestamp': ts,
        },
        'summary': summary_rows,
        'per_video': rows,
    }

    OmegaConf.save(config=OmegaConf.create(payload), f=str(result_yaml))

    return {
        'result_yaml': result_yaml,
        'out_dir': out_dir,
    }
