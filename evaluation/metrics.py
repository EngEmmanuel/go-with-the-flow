from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from datetime import datetime
from omegaconf import OmegaConf
try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None


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


def compute_metrics_for_datasets(
    real_root: Path,
    fake_root: Path,
    metrics: List[str],
    output_dir: Path,
    confidence: float = 0.95,
    resize: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
    max_videos: Optional[int] = None,
) -> Dict[str, Path]:
    real_root = Path(real_root)
    fake_root = Path(fake_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare optional LPIPS model
    want_lpips = any(m.lower() in ("lpips", "lpips_vgg") for m in metrics)
    lpips_net = _try_lpips_device(device) if want_lpips else None
    if want_lpips and lpips_net is None:
        print("[warn] lpips package not found; LPIPS(VGG) will be skipped or set to NaN.")

    # Pair video directories by common names
    real_videos = {p.name: p for p in real_root.iterdir() if p.is_dir()}
    fake_videos = {p.name: p for p in fake_root.iterdir() if p.is_dir()}
    common = sorted(set(real_videos.keys()) & set(fake_videos.keys()))
    if max_videos is not None:
        common = common[:max_videos]
    if not common:
        raise ValueError("No common video folders between real and fake roots.")

    # Save under fake_root/metric_results with a single YAML
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(fake_root) / 'metric_results'
    out_dir.mkdir(parents=True, exist_ok=True)
    result_yaml = out_dir / f"metrics_{ts}.yaml"

    rows = []
    # Accumulate per-video metrics

    for vid in common:
        rdir = real_videos[vid]
        fdir = fake_videos[vid]
        rframes = sorted(rdir.glob('*.png'))
        fframes = sorted(fdir.glob('*.png'))
        n = min(len(rframes), len(fframes))
        if n == 0:
            continue
        rframes = rframes[:n]
        fframes = fframes[:n]

        ssim_vals: List[float] = []
        psnr_vals: List[float] = []
        lpips_vals: List[float] = []

        for rp, fp in zip(rframes, fframes):
            x = _load_image(rp, resize=resize, device=device)
            y = _load_image(fp, resize=resize, device=device)
            # Ensure same size
            if x.shape != y.shape:
                h = min(x.shape[1], y.shape[1])
                w = min(x.shape[2], y.shape[2])
                x = F.interpolate(x.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
                y = F.interpolate(y.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

            if 'ssim' in [m.lower() for m in metrics]:
                ssim_vals.append(_ssim(x, y))
            if 'psnr' in [m.lower() for m in metrics]:
                psnr_vals.append(_psnr(x, y))
            if want_lpips and lpips_net is not None:
                lpips_vals.append(_lpips_vgg(lpips_net, x, y))

        row: Dict[str, float | str] = {'video_name': vid}
        if ssim_vals:
            row['ssim'] = float(sum(ssim_vals) / len(ssim_vals))
        if psnr_vals:
            row['psnr'] = float(sum(psnr_vals) / len(psnr_vals))
        if want_lpips:
            row['lpips_vgg'] = float(sum(lpips_vals) / len(lpips_vals)) if lpips_vals else float('nan')
        rows.append(row)

    per_video_df = pd.DataFrame(rows)

    # Summaries with CIs
    summary_rows = []
    for metric in ['ssim', 'psnr', 'lpips_vgg']:
        if metric in per_video_df.columns:
            vals = per_video_df[metric].dropna().tolist()
            mean, lo, hi = _conf_int(vals, confidence)
            summary_rows.append({'metric': metric, 'mean': mean, 'ci_low': lo, 'ci_high': hi, 'n_videos': len(vals)})

    # Prepare YAML payload
    payload: Dict = {
        'config': {
            'real_root': str(real_root),
            'fake_root': str(fake_root),
            'which': list(metrics),
            'confidence': float(confidence),
            'resize': list(resize) if resize is not None else None,
            'max_videos': int(max_videos) if max_videos is not None else None,
            'timestamp': ts,
        },
        'per_video': rows,
        'summary': summary_rows,
    }

    OmegaConf.save(config=OmegaConf.create(payload), f=str(result_yaml))

    return {
        'result_yaml': result_yaml,
        'out_dir': out_dir,
    }
