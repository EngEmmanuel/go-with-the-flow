import os
import re
import math
import json
import yaml
import shutil
import tempfile
import subprocess
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Optional, Tuple, Union

REGRESSION_BIAS_DEFAULT = -0.03275720505553781  # Example bias value for EF regression correction
BOOTSTRAP_ITERS_DEFAULT = 10000

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
        self.base_dir = Path(base_dir)
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
        self.folds_list = sorted(folds)

        self.postproc_nproc = int(postproc_nproc)

        self.temp_dir = None  # Will hold temporary nnUNet dataset path

        print(f"[INFO] Initialised EjectionFractionSegmenter for {dataset_num_to_name(self.dataset_number)}")
        print(f"       Base dir: {self.base_dir}")
        print(f"       nnUNet env: {self.nnunet_env}")
        print(f"       Use postprocessed: {self.use_postprocessed}")
        print(f"       Config: -c {self.nnunet_config} -tr {self.nnunet_trainer} -p {self.nnunet_plans} -f {' '.join(map(str, self.folds_list))}")

    # ----------------------------------------------------------------------
    def create_nnunet_format(self, input_dir: Path, force_grayscale: bool = True) -> Path:
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
                if force_grayscale:
                    img = Image.open(frame_path).convert("L")
                    img.save(imagesTs / new_name)
                else:
                    shutil.copy(frame_path, imagesTs / new_name)

        print(f"[INFO] nnUNet format created successfully at {self.temp_dir}")
        return self.temp_dir

    # ----------------------------------------------------------------------
    def run_inference(self, input_dir: Path) -> pd.DataFrame:
        """Run nnUNet inference, compute LV area traces, and return a summary DataFrame."""
        # Prepare nnUNet input from raw frames
        input_dir = Path(input_dir)
        fake_dir = get_path_until_fake(input_dir)
        metrics_out_dir = (fake_dir.parent / 'nnunet_ef_metrics')
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
            "-p", self.nnunet_plans
        ] + ["-f"] + [str(f) for f in self.folds_list]
        

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
            fold_for_pp = '_'.join(map(str, self.folds_list))
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
            match = re.match(r"(.*?)_frame(\d+)", pred_file.stem)
            if match:
                vid_name = match.group(1)
                frame_idx = int(match.group(2))
                video_groups.setdefault(vid_name, []).append((frame_idx, pred_file))

        print(f"[INFO] Computing EF metrics for {len(video_groups)} videos...")
        for vid_name, frames in tqdm(video_groups.items(), desc="Computing EF metrics", unit="videos"):
            frames.sort(key=lambda x: x[0])
            lv_areas = []

            for _, fpath in frames:
                seg = np.array(Image.open(fpath))
                lv_mask = seg > 0  # assuming binary mask
                area = np.sum(lv_mask) / lv_mask.size
                lv_areas.append(float(area))

            pred_lvef = (lv_areas[0] - lv_areas[-1]) / lv_areas[0]

            gt_match = re.search(r"ef(\d+)", vid_name)
            gt_lvef = int(gt_match.group(1))/100

            df_rows.append({
                "video_name": vid_name,
                "pred_lvef": pred_lvef,
                "gt_lvef": gt_lvef,
                "error": abs(pred_lvef - gt_lvef),
                "pred_ed_idx": int(np.argmax(lv_areas)), # var len list so -ve index from end
                "pred_es_idx": int(np.argmin(lv_areas)) - len(lv_areas),
                "lv_area": lv_areas
            })

        results_df = pd.DataFrame(df_rows)
        if results_df.empty:
            print("[WARN] No results to save; returning empty DataFrame.")
            return results_df

        results_df.to_csv(metrics_out_dir / f"ef_metrics_per_video_d{self.dataset_number}.csv", index=False)
        print(f"[INFO] EF metrics saved to {metrics_out_dir / f'ef_metrics_per_video_d{self.dataset_number}.csv'}")

        ### EF Regression Analysis
        analyser = EFRegressionAnalyser(
            df_data=results_df,
            bootstrap_iters=BOOTSTRAP_ITERS_DEFAULT,
            bias=REGRESSION_BIAS_DEFAULT,  # Example bias value   
            output_dir=metrics_out_dir
        )
        print("[INFO] EF Regression Analysis Summary:")
        print(analyser.summary())
        print("[INFO] Saving EF Regression Analysis results...")
        analyser.save_all()


        # ------------------------------------------------------------------
        # Cleanup temporary directories
        print(f"[INFO] Cleaning up temporary directories...")
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        if used_output_dir != output_dir:
            shutil.rmtree(used_output_dir, ignore_errors=True)
        print(f"[INFO] Temporary files deleted.")

        return results_df

@dataclass
class EFRegressionAnalyser:
    """
    Analyse regression performance (pred_lvef vs gt_lvef) with optional bias correction.

    If `bias` is provided:
      pred_lvef_bias = clip(pred_lvef - bias, 0, 1)
      error_bias     = |pred_lvef_bias - gt_lvef|

    Metrics are computed on the corrected predictions if bias is set.
    Original (uncorrected) metrics are retained under keys with suffix `_raw`.
    """
    df_data: Union[Path, str, pd.DataFrame]
    bootstrap_iters: int = 0
    bootstrap_ci: float = 0.95
    bias: Optional[float] = None
    output_dir: Optional[Path] = None
    results_df: pd.DataFrame = field(init=False)

    def __post_init__(self):
        # load dataframe if a path/string was provided, otherwise copy the DataFrame
        if isinstance(self.df_data, pd.DataFrame):
            self.results_df = self.df_data.copy()
        else:
            csv_p = Path(self.df_data)
            if not csv_p.exists():
                raise FileNotFoundError(f"CSV not found: {csv_p}")
            self.results_df = pd.read_csv(csv_p)

        # basic validation and coercion
        required = {'pred_lvef', 'gt_lvef'}
        if not required.issubset(self.results_df.columns):
            raise ValueError(f"Data must contain columns: {required}")
        self.results_df['pred_lvef'] = self.results_df['pred_lvef'].astype(float)
        self.results_df['gt_lvef'] = self.results_df['gt_lvef'].astype(float)

        # apply bias correction if requested
        if self.bias is not None:
            corrected = np.clip(self.results_df['pred_lvef'] - self.bias, 0.0, 1.0)
            self.results_df['pred_lvef_bias'] = corrected
            self.results_df['error_bias'] = (corrected - self.results_df['gt_lvef']).abs()

        # prepare output directory if provided
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Core metrics ----------------
    def compute_metrics(self) -> Dict[str, float]:
        df = self.results_df

        # Always compute raw metrics
        metrics_raw = self._compute_set(df['gt_lvef'].values, df['pred_lvef'].values, prefix='_raw')

        if self.bias is None:
            # No bias: return raw metrics without prefix normalization
            # Strip '_raw' suffix for main keys
            clean = {k.replace('_raw', ''): v for k, v in metrics_raw.items()}
            return clean
        else:
            # Bias provided: compute corrected metrics as primary; keep raw as reference
            metrics_corr = self._compute_set(df['gt_lvef'].values, df['pred_lvef_bias'].values, prefix='')
            # Merge
            merged = {**metrics_corr, **metrics_raw}
            merged['bias_used'] = float(self.bias)
            return merged

    def _compute_set(self, y: np.ndarray, y_hat: np.ndarray, prefix: str = '') -> Dict[str, float]:
        diff = y_hat - y
        mae = np.mean(np.abs(diff))
        mse = np.mean(diff ** 2)
        rmse = math.sqrt(mse)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_vec = np.abs(diff) / np.where(y == 0, np.nan, np.abs(y))
        mape = np.nanmean(mape_vec)
        bias_val = np.mean(diff)
        abs_bias = np.abs(bias_val)
        sd_diff = np.std(diff, ddof=1)
        loa_lower = bias_val - 1.96 * sd_diff
        loa_upper = bias_val + 1.96 * sd_diff
        pearson_r, pearson_p = stats.pearsonr(y, y_hat)
        spearman_r, spearman_p = stats.spearmanr(y, y_hat)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        out = {
            f'count{prefix}': int(len(y)),
            f'mae{prefix}': float(mae),
            f'mse{prefix}': float(mse),
            f'rmse{prefix}': float(rmse),
            f'mape{prefix}': float(mape),
            f'bias{prefix}': float(bias_val),
            f'abs_bias{prefix}': float(abs_bias),
            f'loa_lower{prefix}': float(loa_lower),
            f'loa_upper{prefix}': float(loa_upper),
            f'r2{prefix}': float(r2),
            f'pearson_r{prefix}': float(pearson_r),
            f'pearson_p{prefix}': float(pearson_p),
            f'spearman_r{prefix}': float(spearman_r),
            f'spearman_p{prefix}': float(spearman_p),
        }
        if self.bootstrap_iters > 0:
            boot = self._bootstrap_metrics(y, y_hat)
            # Add with prefix
            for k, v in boot.items():
                out[f'{k}{prefix}'] = v
        return out

    def _bootstrap_metrics(self, y: np.ndarray, y_hat: np.ndarray) -> Dict[str, Tuple[float, float]]:
        n = len(y)
        alpha = 1 - self.bootstrap_ci
        biases = []
        maes = []
        for _ in range(self.bootstrap_iters):
            idx = np.random.randint(0, n, size=n)
            yy = y[idx]
            pp = y_hat[idx]
            d = pp - yy
            biases.append(np.mean(d))
            maes.append(np.mean(np.abs(d)))
        low_q = alpha / 2 * 100
        high_q = (1 - alpha / 2) * 100
        return {
            'bias_ci': (float(np.percentile(biases, low_q)), float(np.percentile(biases, high_q))),
            'mae_ci': (float(np.percentile(maes, low_q)), float(np.percentile(maes, high_q))),
        }

    # ------------- Augment per-row -------------
    def augment_rows(self) -> pd.DataFrame:
        df = self.results_df.copy()
        df['diff_raw'] = df['pred_lvef'] - df['gt_lvef']
        df['abs_diff_raw'] = np.abs(df['diff_raw'])
        with np.errstate(divide='ignore', invalid='ignore'):
            df['ape_raw'] = df['abs_diff_raw'] / np.where(df['gt_lvef'] == 0, np.nan, np.abs(df['gt_lvef']))
        if self.bias is not None:
            df['diff_bias'] = df['pred_lvef_bias'] - df['gt_lvef']
            df['abs_diff_bias'] = np.abs(df['diff_bias'])
            with np.errstate(divide='ignore', invalid='ignore'):
                df['ape_bias'] = df['abs_diff_bias'] / np.where(df['gt_lvef'] == 0, np.nan, np.abs(df['gt_lvef']))
        return df

    # ------------- Save utilities -------------
    def save_summary(self, filename: str = "ef_reg_summary.yaml"):
        if self.output_dir is None:
            raise ValueError("output_dir not set for EFRegressionAnalyser.")
        out_path = self.output_dir / filename
        metrics = self.compute_metrics()
        if out_path.suffix.lower() in ('.yml', '.yaml'):
            with open(out_path, 'w') as f:
                yaml.safe_dump(metrics, f)
        elif out_path.suffix.lower() == '.json':
            with open(out_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        else:
            pd.DataFrame([metrics]).to_csv(out_path, index=False)

    def save_augmented(self, filename: str = "ef_reg_augmented.csv"):
        if self.output_dir is None:
            raise ValueError("output_dir not set for EFRegressionAnalyser.")
        out_path = self.output_dir / filename
        self.augment_rows().to_csv(out_path, index=False)

    # ------------- Plotting -------------
    def plot_joint_and_bland_altman(
        self,
        filename: str = "ef_reg_plots.png",
        use_bias: bool = True,
        figsize: Tuple[int, int] = (12, 5),
        scatter_kwargs: Optional[Dict] = None,
        annotate: bool = True
    ) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir not set for EFRegressionAnalyser.")
        scatter_kwargs = scatter_kwargs or dict(s=45, edgecolor='k',alpha=0.7)

        df = self.results_df
        if self.bias is not None and use_bias:
            y_hat = df['pred_lvef_bias'].values
            label_pred = "Pred LVEF (bias corrected)"
        else:
            y_hat = df['pred_lvef'].values
            label_pred = "Pred LVEF"

        y = df['gt_lvef'].values
        diff = y_hat - y
        mean_vals = (y_hat + y) / 2.0
        bias_val = np.mean(diff)
        sd_diff = np.std(diff, ddof=1)
        loa_lower = bias_val - 1.96 * sd_diff
        loa_upper = bias_val + 1.96 * sd_diff

        # Jointplot
        temp_df = pd.DataFrame({'gt_lvef': y, 'pred': y_hat})
        jp = sns.jointplot(
            data=temp_df,
            x='gt_lvef',
            y='pred',
            kind='reg',
            scatter_kws=scatter_kwargs,
            line_kws={'color': 'red', 'lw': 1.2},
            height=figsize[1] * 0.9
        )
        jp.ax_joint.set_xlabel("Ground Truth LVEF")
        jp.ax_joint.set_ylabel(label_pred)
        jp.fig.tight_layout()

        # Bland–Altman
        fig_ba, ax_ba = plt.subplots(figsize=(figsize[0] * 0.45, figsize[1] * 0.9))
        ax_ba.scatter(mean_vals, diff, **scatter_kwargs)
        ax_ba.axhline(bias_val, color='red', linestyle='--', label=f"Bias={bias_val:.3f}")
        ax_ba.axhline(loa_lower, color='orange', linestyle='--', label=f"LoA-={loa_lower:.3f}")
        ax_ba.axhline(loa_upper, color='orange', linestyle='--', label=f"LoA+={loa_upper:.3f}")
        ax_ba.set_xlabel("Mean of (Pred, GT)")
        ax_ba.set_ylabel("Pred - GT")
        ax_ba.set_title("Bland–Altman")

        if annotate:
            ax_ba.text(0.02, 0.98, f"Bias={bias_val:.3f}\nSD={sd_diff:.3f}",
                       transform=ax_ba.transAxes,
                       va='top', ha='left', fontsize=10,
                       bbox=dict(boxstyle="round", fc="white", alpha=0.6))
        ax_ba.legend(loc='lower right', fontsize=8)
        fig_ba.tight_layout()

        out_path = self.output_dir / filename
        jp_path = out_path.with_name(out_path.stem + "_jointplot.png")
        ba_path = out_path.with_name(out_path.stem + "_bland_altman.png")
        jp.fig.savefig(jp_path, dpi=750)
        fig_ba.savefig(ba_path, dpi=750)
        plt.close(jp.fig)
        plt.close(fig_ba)

        im1 = Image.open(jp_path)
        im2 = Image.open(ba_path)
        w = im1.width + im2.width
        h = max(im1.height, im2.height)
        composite = Image.new("RGB", (w, h), (255, 255, 255))
        composite.paste(im1, (0, 0))
        composite.paste(im2, (im1.width, 0))
        composite.save(out_path)
        return out_path

    # ------------- Convenience -------------
    def summary(self) -> Dict[str, float]:
        return self.compute_metrics()

    def save_all(self):
        """Save summary YAML/CSV, augmented CSV, and composite plot to output_dir."""
        self.save_summary()
        self.save_augmented()
        self.plot_joint_and_bland_altman()


if __name__ == "__main__":
    # Example usage
    tempfile.tempdir = '/data/spet4299/spet4299_tmp'  # set tmpdir for nnUNet
    base_dir = Path("/data/spet4299/flow-matching/go-with-the-flow/nnunet_test_vid_ef_reg/subset/fake")

    if True:
        segmenter = EjectionFractionSegmenter(
            dataset_number=316,
            base_dir=base_dir,
            nnunet_env="nonewnet_v3",
            use_postprocessed=False,
            nnunet_config="2d",
            nnunet_trainer="nnUNetTrainer",
            nnunet_plans="nnUNetPlans",
            folds=[0,1,2,3,4]
        )

        input_frames_dir = segmenter.base_dir
        results_df = segmenter.run_inference(input_frames_dir)
        print(results_df.head())

    if False:
        csv_path = base_dir.parent / "ef_metrics_per_video_d316.csv"
        analyser = EFRegressionAnalyser(
            df_path=csv_path,
            bootstrap_iters=BOOTSTRAP_ITERS_DEFAULT,
            bootstrap_ci=0.95,
            bias=REGRESSION_BIAS_DEFAULT,
            output_dir=base_dir.parent / "ef_analysis_outputs"
        )
        analyser.save_all()
        summary_metrics = analyser.summary()
        print("Summary Metrics:")
        for k, v in summary_metrics.items():
            print(f"  {k}: {v}")


