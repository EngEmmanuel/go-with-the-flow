import os
import re
import shutil
import tempfile
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image


def get_path_until_fake(input_path: Path) -> Path:
    """
    Given a path, returns the portion of the path up to (and including)
    the directory named 'fake'.

    Example:
        >>> get_path_until_fake(Path("/blah/blah/fake/blah/new.png"))
        PosixPath('/blah/blah/fake')

    """
    input_path = Path(input_path).resolve()
    parts = list(input_path.parts)

    if "fake" not in parts:
        raise ValueError(f"No directory named 'fake' found in {input_path}")

    fake_index = parts.index("fake")
    return Path(*parts[:fake_index + 1])


def dataset_num_to_name(dataset_number: int) -> str:
    """
    Given a dataset number, find the corresponding dataset name
    inside the nnUNet raw directory.
    """
    nnunet_raw = Path(os.environ["nnUNet_raw"])
    pattern = re.compile(rf"^Dataset{dataset_number:03d}_.+")

    for folder in nnunet_raw.iterdir():
        if folder.is_dir() and pattern.match(folder.name):
            return folder.name

    raise FileNotFoundError(
        f"No dataset with number {dataset_number} found in {nnunet_raw}"
    )

class EjectionFractionSegmenter:
    """LV segmentation with a pretrained nnUNet model for EF estimation."""

    def __init__(
                self, 
                dataset_number: int, 
                base_dir: Path, 
                nnunet_env: str = "nonewnet_v3", 
                use_postprocessed: bool = False,
                nnunet_config: str = "2d",
                nnunet_trainer: str = "nnUNetTrainer",
                nnunet_plans: str = "nnUNetPlans",
                folds: int | list[int] = 0,
                postproc_nproc: int = 8,
        ):
        """
        Parameters
        ----------
        dataset_number : int
            The nnUNet dataset number (e.g., 505 for 'Dataset505_LVSeg').
        base_dir : Path
            The working directory containing input videos or frame folders.
        nnunet_env : str
            Name of the conda environment containing nnUNetv2.
        use_postprocessed : bool
            Whether to use nnUNet’s postprocessed outputs if available.
        """
        self.dataset_number = dataset_number
        self.nnunet_env = nnunet_env
        self.use_postprocessed = use_postprocessed

        # Resolve nnUNet environment variables
        self.nnunet_raw = Path(os.environ["nnUNet_raw"])
        self.nnunet_preprocessed = Path(os.environ["nnUNet_preprocessed"])
        self.nnunet_results = Path(os.environ["nnUNet_results"])
        self.nnunet_config = nnunet_config
        self.nnunet_trainer = nnunet_trainer
        self.nnunet_plans = nnunet_plans
        # normalize folds to list[str]

        if isinstance(folds, int):
            self.folds_list = [folds]
        self.folds_list = sorted(self.folds_list)

        self.postproc_nproc = int(postproc_nproc)

        self.temp_dir = None  # Will hold temporary nnUNet dataset path

        print(f"[INFO] Initialised EjectionFractionSegmenter for {dataset_num_to_name(self.dataset_number)}")
        print(f"       nnUNet env: {self.nnunet_env}")
        print(f"       Use postprocessed: {self.use_postprocessed}")
        print(f"       Config: -c {self.nnunet_config} -tr {self.nnunet_trainer} -p {self.nnunet_plans} -f {' '.join(self.folds_list)}")

    # ----------------------------------------------------------------------
    def create_nnunet_format(self, input_dir: Path) -> Path:
        """Convert input frames into a temporary nnUNet-compatible dataset and return its root folder."""
        input_dir = Path(input_dir)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="nnunet_infer_"))
        imagesTs = self.temp_dir / "imagesTs"
        imagesTs.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Creating temporary nnUNet dataset in {self.temp_dir}")

        for video_folder in sorted(input_dir.iterdir()):
            if not video_folder.is_dir():
                continue

            video_name = video_folder.name
            frame_paths = sorted(video_folder.glob("*.png"))

            for idx, frame_path in enumerate(frame_paths):
                new_name = f"{video_name}_frame{idx}_0000.png"
                shutil.copy(frame_path, imagesTs / new_name)

        print(f"[INFO] nnUNet format created successfully at {self.temp_dir}")
        return self.temp_dir

    # ----------------------------------------------------------------------
    def run_inference(self, input_dir: Path) -> pd.DataFrame:
        """Run nnUNet inference, compute LV area traces, and return a summary DataFrame."""
        # Prepare nnUNet input from raw frames
        input_dir = Path(input_dir)
        fake_dir = get_path_until_fake(input_dir)
        metrics_out_dir = fake_dir.parent
        metrics_out_dir.mkdir(parents=True, exist_ok=True)

        nnunet_root = self.create_nnunet_format(input_dir)
        images_dir = nnunet_root / "imagesTs"
        output_dir = Path(tempfile.mkdtemp(prefix="nnunet_output_"))
        print(f"[INFO] Running nnUNet inference...\n       Output: {output_dir}")

        cmd = [
            "conda", "run", "-n", self.nnunet_env,
            "nnUNetv2_predict",
            "-i", str(images_dir),
            "-o", str(output_dir),
            "-d", str(self.dataset_number),
            "-c", self.nnunet_config,
            "-tr", self.nnunet_trainer,
            "-p", self.nnunet_plans,
            "-f", ' '.join(self.folds_list),
        ]

        try:
            subprocess.run(cmd, check=True)
            print("[INFO] nnUNet inference completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] nnUNet inference failed: {e}")
            return pd.DataFrame()

        # ------------------------------------------------------------------
        # Optionally run post-processing using cross-validation artifacts
        used_output_dir = output_dir
        if self.use_postprocessed:
            # locate pp files under nnUNet_results/Dataset{num}*/<trainer>__<plans>__<config>/crossval_results_folds_<fold>
            ds_dir = self.nnunet_results / str(dataset_num_to_name(self.dataset_number))
            fold_for_pp = '_'.join(self.folds_list)
            pp_dir = ds_dir / f"{self.nnunet_trainer}__{self.nnunet_plans}__{self.nnunet_config}" / f"crossval_results_folds_{fold_for_pp}"


            output_dir_pp = Path(str(output_dir) + "_pp")
            output_dir_pp.mkdir(parents=True, exist_ok=True)
            pp_cmd = [
                "conda", "run", "-n", self.nnunet_env,
                "nnUNetv2_apply_postprocessing",
                "-i", str(output_dir),
                "-o", str(output_dir_pp),
                "-pp_pkl_file", str(pp_dir / "postprocessing.pkl"),
                "-np", str(self.postproc_nproc),
                "-plans_json", str(pp_dir / "plans.json"),
            ]
            try:
                subprocess.run(pp_cmd, check=True)
                print("[INFO] nnUNet postprocessing completed successfully.")
                used_output_dir = output_dir_pp
            except subprocess.CalledProcessError as e:
                print(f"[WARN] nnUNet postprocessing failed: {e}; using raw predictions.")
        # Compute LV area traces for each video
        df_rows = []
        output_files = sorted(used_output_dir.glob("*.png"))

        # Group predictions by video name prefix
        video_groups = {}
        for pred_file in output_files:
            match = re.match(r"(.*?)_frame(\d+)_0000", pred_file.stem)
            if match:
                vid_name = match.group(1)
                frame_idx = int(match.group(2))
                video_groups.setdefault(vid_name, []).append((frame_idx, pred_file))

        for vid_name, frames in video_groups.items():
            frames.sort(key=lambda x: x[0])
            lv_areas = []

            for _, fpath in frames:
                seg = np.array(Image.open(fpath))
                lv_mask = seg > 0  # assuming binary mask
                area = np.sum(lv_mask) / lv_mask.size
                lv_areas.append(area)

            pred_lvef = (lv_areas[0] - lv_areas[-1]) / lv_areas[0]

            gt_match = re.search(r"ef(\d+)", vid_name)
            gt_lvef = int(gt_match.group(1))

            df_rows.append({
                "video_name": vid_name,
                "lv_area": lv_areas,
                "pred_lvef": pred_lvef,
                "gt_lvef": gt_lvef,
                "error": abs(pred_lvef - gt_lvef)
            })

        results_df = pd.DataFrame(df_rows)

        results_df.to_csv(metrics_out_dir / f"ef_metrics_d{self.dataset_number}_c{self.nnunet_config}_{'_'.join(self.folds_list)}.csv", index=False)

        # ------------------------------------------------------------------
        # Cleanup temporary directories
        print(f"[INFO] Cleaning up temporary directories...")
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        if used_output_dir != output_dir:
            shutil.rmtree(used_output_dir, ignore_errors=True)
        print(f"[INFO] Temporary files deleted.")

        return results_df


#TODO Work out eval of df results

import math
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def _read_df(p: Path) -> pd.DataFrame:
    p = Path(p)
    if p.suffix.lower() in (".csv",):
        return pd.read_csv(p)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_pickle(p)


def _metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = int(y_true.size)
    if n == 0:
        return {"n": 0, "r2": float("nan"), "mae": float("nan"),
                "mse": float("nan"), "rmse": float("nan"),
                "bias": float("nan"), "bias_stderr": float("nan")}
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rmse = math.sqrt(mse)
    bias = float(np.mean(y_pred - y_true))
    bias_stderr = float(np.std(y_pred - y_true, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {"n": n, "r2": r2, "mae": mae, "mse": mse, "rmse": rmse, "bias": bias, "bias_stderr": bias_stderr}


def summarise_dfs_dict(
    dfs: Dict[str, Dict[str, Dict[str, str]]],
    *,
    pred_col: str = "pred_lvef",
    gt_col: str = "gt_lvef",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Summarise per-file regression metrics and aggregate up the hierarchy.
    Returns (nested_summary, tidy_df).
    """
    nested_summary: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    # per-file metrics
    for scheme, frame_map in dfs.items():
        nested_summary.setdefault(scheme, {})
        for frame_dir, roots in frame_map.items():
            nested_summary[scheme].setdefault(frame_dir, {})
            for root_name, path in roots.items():
                df = _read_df(Path(path))
                y_true = df[gt_col].values
                y_pred = df[pred_col].values
                m = _metrics_from_arrays(y_true, y_pred)
                nested_summary[scheme][frame_dir][root_name] = m
                rows.append({
                    "scheme": scheme,
                    "framewise_dir": frame_dir,
                    "root": root_name,
                    "path": str(path),
                    **m
                })

    if len(rows) == 0:
        return nested_summary, pd.DataFrame()

    per_root_df = pd.DataFrame.from_records(rows)

    # helper to aggregate by concatenating underlying arrays (exact aggregation)
    def _aggregate_group(group_df: pd.DataFrame) -> Dict[str, float]:
        trues = []
        preds = []
        for p in group_df["path"]:
            df = _read_df(Path(p))
            trues.append(df[gt_col].values)
            preds.append(df[pred_col].values)
        if len(trues) == 0:
            return {"n": 0, "r2": float("nan"), "mae": float("nan"),
                    "mse": float("nan"), "rmse": float("nan"),
                    "bias": float("nan"), "bias_stderr": float("nan")}
        y_true_all = np.concatenate(trues)
        y_pred_all = np.concatenate(preds)
        return _metrics_from_arrays(y_true_all, y_pred_all)

    # aggregate per (scheme, frame_dir)
    agg_framewise_rows = []
    for (scheme, frame_dir), g in per_root_df.groupby(["scheme", "framewise_dir"]):
        agg = _aggregate_group(g)
        nested_summary[scheme].setdefault(frame_dir, {})["_aggregate_by_framewise_dir"] = agg
        agg_framewise_rows.append({
            "scheme": scheme,
            "framewise_dir": frame_dir,
            "root": "_AGG_FRAMEWISE_",
            **agg
        })

    # aggregate per scheme
    agg_scheme_rows = []
    for scheme, g in per_root_df.groupby("scheme"):
        agg = _aggregate_group(g)
        nested_summary[scheme]["_aggregate_by_scheme"] = agg
        agg_scheme_rows.append({
            "scheme": scheme,
            "framewise_dir": "_AGG_SCHEME_",
            "root": "_AGG_SCHEME_",
            **agg
        })

    # overall aggregate
    overall = _aggregate_group(per_root_df)
    nested_summary["_overall"] = overall
    overall_row = {"scheme": "_ALL_", "framewise_dir": "_ALL_", "root": "_ALL_", **overall}

    tidy = pd.concat([
        per_root_df,
        pd.DataFrame.from_records(agg_framewise_rows) if agg_framewise_rows else pd.DataFrame(),
        pd.DataFrame.from_records(agg_scheme_rows) if agg_scheme_rows else pd.DataFrame(),
        pd.DataFrame([overall_row])
    ], ignore_index=True, sort=False)

    # keep canonical column order
    cols = ["scheme", "framewise_dir", "root", "n", "r2", "mae", "mse", "rmse", "bias", "bias_stderr", "path"]
    tidy = tidy[[c for c in cols if c in tidy.columns]]

    return nested_summary, tidy

