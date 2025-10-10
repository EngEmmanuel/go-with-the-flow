from __future__ import annotations

import re
import math
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    real_glob: Union[str, List[str]] = '*',
    fake_glob: Union[str, List[str]] = '*',
    payload_kwargs: Optional[Dict[str, Any]] = None,
    just_save_payload: bool = False,
) -> Dict[str, Path]:
    real_root = inference_root / 'real'
    fake_root = inference_root / 'fake'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support multiple globs: normalize to lists
    real_globs: List[str] = [real_glob] if isinstance(real_glob, str) else list(real_glob)
    fake_globs: List[str] = [fake_glob] if isinstance(fake_glob, str) else list(fake_glob)

    if not just_save_payload:
        paired_videos: Dict[str, Tuple[Path, Path]] = {}
        for (rg, fg) in zip(real_globs, fake_globs):
            subset_pairs = get_real_and_fake_video_folders(
                real_root=real_root,
                fake_root=fake_root,
                real_glob=rg,
                fake_glob=fg,
            )
            paired_videos.update(subset_pairs)
    # Prepare optional LPIPS model
    want_lpips = any(m.lower() in ("lpips", "lpips_vgg") for m in metrics)
    lpips_net = _try_lpips_device(device) if want_lpips else None
    if want_lpips and lpips_net is None:
        print("[warn] lpips package not found; LPIPS(VGG) will be skipped or set to NaN.")

    # Save under fake_root/metric_results with a single YAML
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(fake_root).parent / 'metric_results' #/ ts
    out_dir.mkdir(parents=True, exist_ok=True)
    result_yaml = out_dir / "metrics_summary.yaml"

    rows = []
    if not just_save_payload:
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


            row = {'real_video' : rdir.name, 'fake_video' : fdir.name}
            row['ef'] = int(fdir.name.split('_ef')[-1].split('_')[0])
            row['nmf'] = fdir.name.split('_nmf')[-1]
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

    # Save per-video dataframe separately (CSV)
    per_video_csv = out_dir / f"metrics_per_video.csv"
    per_video_df.to_csv(per_video_csv, index=False)

    # YAML now only stores config + summary (aggregate) information
    payload: Dict = {
        'config': {
            'real_root': str(real_root),
            'fake_root': str(fake_root),
            'real_globs': real_globs,
            'fake_globs': fake_globs,
            'which': list(metrics),
            'confidence': float(confidence),
            'resize': list(resize) if resize is not None else None,
            'timestamp': ts,
            'per_video_csv': str(per_video_csv),
        },
        **payload_kwargs,
        'summary': summary_rows,
    }
    OmegaConf.save(config=OmegaConf.create(payload), f=str(result_yaml))

    return {
        'result_yaml': result_yaml,
        'per_video_csv': per_video_csv,
        'out_dir': out_dir,
    }


from typing import Dict, Any, Optional, Union, Tuple, List, Iterable
from tabulate import tabulate  # pip install tabulate

NMF_ROWS = ["nmf25p", "nmf50p", "nmf75p", "nmfmax"]

def collect_metric_results(
    dir_base: Path,
    save_path: Optional[Union[str, Path]] = None,
    *,
    make_tables: bool = False,
    table_format: str = "simple",     # "simple", "grid", "github", "fancy_grid", ...
    tables_filename: str = "all_tables.txt",
    sigfigs: int = 5
) -> Tuple[Dict[str, Any], str]:
    """
    Collect metrics from any path matching:
      <dir_base>/<nmf>/<variant>[/<another_level>/...]/metric_results/metrics_summary.yaml

    - Ignores timestamped subfolders (we only accept files directly inside metric_results/).
    - 'summary' is converted from a list into a dict keyed by metric.
    - Output is nested by the path segments between <nmf> and 'metric_results':
        data[nmf][variant][another_level]... = {stylegan_results, summary, path}
    - If `save_path` is provided, writes nested YAML there (or to <save_path>/all_metrics.yaml if a directory).
    - If `make_tables=True`, writes a single text file with all per-variant-path tables
      (rows = NMF_ROWS; columns = union of metrics; values = stylegan metric or summary['mean']).

    Returns:
        (nested_results_dict, combined_tables_text)
    """
    dir_base = dir_base.resolve()
    out: Dict[str, Any] = {}

    # -------- scan for YAMLs (ignore timestamped subfolders) --------
    for yml in dir_base.rglob("metrics_summary.yaml"):
        if yml.parent.name != "metric_results":
            continue
        try:
            rel = yml.relative_to(dir_base)
        except ValueError:
            continue

        parts = rel.parts
        # Expect: [nmf, <variant and optional deeper keys...>, 'metric_results', 'metrics_summary.yaml']
        if "metric_results" not in parts:
            continue
        mr_idx = parts.index("metric_results")
        if mr_idx < 2:  # need at least nmf + one variant key
            continue

        nmf = parts[0]
        variant_keys = list(parts[1:mr_idx])  # one or more segments

        with open(yml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        stylegan = data.get("stylegan_results") or {}
        summary_list = data.get("summary") or []

        # list[{metric, mean, ...}] -> dict[metric] = {...}
        summary_dict: Dict[str, Dict[str, Any]] = {}
        for row in summary_list:
            if isinstance(row, dict) and row.get("metric"):
                key = row["metric"]
                summary_dict[key] = {k: v for k, v in row.items() if k != "metric"}

        # Insert into nested dict: out[nmf][variant][...]
        leaf = {
            "stylegan_results": stylegan,
            "summary": summary_dict,
            "path": str(yml),
        }
        _nested_set(out, [nmf, *variant_keys], leaf)

    # -------- save nested YAML (optional) --------
    combined_tables_text = ""
    out_dir_for_tables: Optional[Path] = None
    if save_path is not None:
        sp = Path(save_path)
        if sp.exists() and sp.is_dir():
            sp = sp / "all_metrics.yaml"
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, sort_keys=False)
        out_dir_for_tables = sp.parent

    # -------- build tables & write one combined .txt (optional) --------
    if make_tables:
        combined_tables_text = _build_all_tables_text(
            out, table_format=table_format, sigfigs=sigfigs
        )
        if out_dir_for_tables is not None:
            (out_dir_for_tables / tables_filename).write_text(
                combined_tables_text, encoding="utf-8"
            )

    return out, combined_tables_text


# ---------------- helpers ----------------

def _nested_set(d: Dict[str, Any], keys: List[str], value: Dict[str, Any]) -> None:
    """Insert value into nested dict along keys, creating intermediate dicts as needed."""
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value

def _sf(val: Any, sigfigs: int) -> str:
    if val is None:
        return "-"
    try:
        return f"{float(val):.{sigfigs}g}"
    except Exception:
        return str(val)

def _iter_variant_paths(subtree: Dict[str, Any], prefix: Tuple[str, ...] = ()) -> Iterable[Tuple[str, ...]]:
    """
    Yield variant-paths (as tuples of strings) below a single nmf subtree.
    A leaf is identified by having 'stylegan_results' in the dict.
    """
    if isinstance(subtree, dict) and "stylegan_results" in subtree:
        yield prefix
        return
    if isinstance(subtree, dict):
        for k, v in subtree.items():
            yield from _iter_variant_paths(v, prefix + (k,))

def _get_node_at_path(subtree: Dict[str, Any], path: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    cur = subtree
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    if isinstance(cur, dict) and "stylegan_results" in cur:
        return cur
    return None

def _build_all_tables_text(
    data: Dict[str, Any],
    *,
    table_format: str = "simple",
    sigfigs: int = 5
) -> str:
    """
    Build tables per 'variant-path' (e.g., 'framewise', 'framewise_no_pad', or 'framewise/clipA'),
    concatenated into a single text blob with headers. Rows are NMF_ROWS; columns are the union of
    metric names across nmf entries; values come from stylegan_results (direct) or summary['mean'].
    """
    # Collect all variant-paths across all nmf branches
    all_paths: set[Tuple[str, ...]] = set()
    for nmf, subtree in data.items():
        if nmf not in NMF_ROWS:
            continue
        for path in _iter_variant_paths(subtree):
            all_paths.add(path)

    chunks: List[str] = []
    for path in sorted(all_paths):
        title = " / ".join(path) if path else "(root)"
        # Determine columns (union across nmf rows for this path)
        metrics_set = set()
        for nmf in NMF_ROWS:
            node = _get_node_at_path(data.get(nmf, {}), path)
            if not node:
                continue
            metrics_set.update(node.get("stylegan_results", {}).keys())
            metrics_set.update(node.get("summary", {}).keys())
        metrics = sorted(metrics_set)

        # Build table rows
        rows = []
        for nmf in NMF_ROWS:
            node = _get_node_at_path(data.get(nmf, {}), path)
            row = [nmf]
            for m in metrics:
                cell = "-"
                if node:
                    if m in node.get("stylegan_results", {}):
                        cell = _sf(node["stylegan_results"][m], sigfigs)
                    elif m in node.get("summary", {}):
                        cell = _sf(node["summary"][m].get("mean"), sigfigs)
                row.append(cell)
            rows.append(row)

        headers = ["nmf"] + metrics
        table_str = tabulate(rows, headers=headers, tablefmt=table_format, stralign="right")
        chunks.append(f"=== {title} ===")
        chunks.append(table_str)

    return "\n\n".join(chunks)




