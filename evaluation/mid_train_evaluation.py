
import re
import yaml
import wandb
import torch
import hydra

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import select_device
from utils.evaluation import *
from evaluation.latents_to_videos import convert_latents_directory
from evaluation.metrics import compute_metrics_for_datasets, run_stylegan_metrics

class EvaluateTrainProcess():
    def __init__(self, cfg: DictConfig, run_dir=None, output_dir=None):
        self.cfg = cfg
        self.device = select_device()

        # Potentially override dirs
        self.run_dir = Path(run_dir) if run_dir is not None else Path(self.cfg.run_dir)
        self.output_dir = Path(output_dir) if output_dir is not None else Path(self.cfg.output_dir)

        self.ckpt_dir = self.run_dir / "checkpoints"
        self.latent_dir = self.run_dir / "sample_videos"
        self.wandb_dir = self.run_dir / "wandb"
        if not self.wandb_dir.exists():
            self.wandb_dir = None


        self._get_latents_and_checkpoints()

        self.query = {
            'reconstruction-nmfmax': "rec_or_gen == 'rec' and n_missing_frames == 'max'",
            'generation-nmfmax': "rec_or_gen == 'gen' and n_missing_frames == 'max'"
        }

        self.results_dir = self.latent_dir / "mid_train_evaluation_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_latents_and_checkpoints(self):
        'Gets and sorts all checkpoint paths in the run directory'
        latent_dir_names = {x.stem for x in self.latent_dir.glob("*")}
        ckpt_names = {x.stem for x in self.ckpt_dir.glob("*.ckpt")}

        common = latent_dir_names & ckpt_names
        only_in_latents = latent_dir_names - ckpt_names
        only_in_ckpts = ckpt_names - latent_dir_names
        if (only_in_latents) or (only_in_ckpts):
            print('Warning:', 'Only in latents:', only_in_latents, 'Only in ckpts:', only_in_ckpts)
        
        if 'last' in common:
            common.remove('last')
            self.names = sorted(common, key=lambda x: int(x.split('step=')[1])) + ['last']
        else:
            self.names = sorted(common, key=lambda x: int(x.split('step=')[1]))
                        
    def _name_to_ckpt_path(self, name: str) -> Path:
        'Converts a latent directory name to its corresponding checkpoint path'
        return self.ckpt_dir / f"{name}.ckpt"
    
    def _name_to_latent_path(self, name: str) -> Path:
        'Converts a latent directory name to its corresponding latent directory path'
        return self.latent_dir / name

    def _get_wandb_info(self):
        'Gets wandb project, entity, and run id from the run directory'

        run_cfg_dir = Path(self.run_dir) / '.hydra' / 'config.yaml'
        run_cfg = OmegaConf.load(run_cfg_dir)
        project = run_cfg.wandb.init_kwargs.project
        entity = run_cfg.wandb.init_kwargs.entity

        # Get run id
        match = [x for x in self.wandb_dir.glob("run-*")]
        if not match:
            raise ValueError("No wandb run directory found.")
        if len(match) > 1:
            print("Multiple wandb run directories found; using the first one.")
            print([x.name for x in match])
        
        run_id = match[0].name.split("-")[-1]
        return project, entity, run_id

    def load_results_df(self) -> pd.DataFrame:
        'Loads a results dataframe from a CSV file'
        path = self.results_dir / "all_checkpoints_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        df = pd.read_csv(path)
        return df

    def _delete_files(self, paths: list[Path]):
        'Deletes specified files to save space'
        for p in paths:
            if p.exists():
                print(f"Deleting {'/'.join(p.parts[-5:])}")
                p.unlink()

    def _delete_latents(self, names: list[str]):
        for name in names:
            latent_path = self._name_to_latent_path(name)
            files = [x for x in latent_path.glob('*.pt')]
            self._delete_files(files)

    def decode_checkpoint_latents(self, latents_dir: Path):
        'Evaluates a single checkpoint and saves outputs to output_dir'

        decoded_videos_dirs = []
        for name, query in self.query.items():
            decoded_videos_dir = convert_latents_directory(
                real_data_path = Path(self.cfg.real_data_path),
                latents_dir = latents_dir,
                run_dir = self.run_dir,
                repo_id = self.cfg.repo_id,
                query = {'name': name, 'pattern': query},
                output_dir = self.output_dir / latents_dir.name / name,
                types = self.cfg.types,
                fps_metadata_csv = self.cfg.fps_metadata_csv,
                decode_batch_size = self.cfg.get('decode_batch_size', 32),
                device = self.device,
                debugging = self.cfg.get('debugging', False)
                #test_n=4
            )

            decoded_videos_dirs.append(decoded_videos_dir)
        return decoded_videos_dirs

    def calculate_metrics(self, decoded_videos_dir: Path, stylegan_only: bool = False):
        '''
        Evaluates a single checkpoint and saves outputs to output_dir'
        stylegan metrics are combined with pairwise metrics if both are requested
        '''
        stylegan_metrics = [x for x in self.cfg.metrics.get('stylegan_metrics', [])]
        pairwise_metrics = [x for x in self.cfg.metrics.get('pairwise_metrics', [])]
        eval_types = [x for x in self.cfg.types if 'framewise' in x]

        results = {}
        for types in eval_types:
            stylegan_metric_results = {}
            if stylegan_metrics:
                stylegan_metric_results = run_stylegan_metrics(
                    decoded_videos_dir / types,
                    stylegan_metrics=','.join(stylegan_metrics),
                    n_gpus=self.cfg.metrics.get('n_gpus', 1)
                )

                if stylegan_only:
                    results[types] = {'stylegan_results': stylegan_metric_results['results']}

            pairwise_metric_results = {}
            if pairwise_metrics and not stylegan_only:
                stylegan_payload = {'stylegan_results': stylegan_metric_results.get('results', {})}

                pairwise_metric_results = compute_metrics_for_datasets(
                    decoded_videos_dir / types,
                    metrics=pairwise_metrics,
                    payload_kwargs=stylegan_payload
                )

                results[types] = pairwise_metric_results['results']

        return results

# stylegan_metrics_results = {'results': {'fvd2048_10f': 4845.533858185447}, 'metric': 'fvd2048_10f', 'total_time': 79.95496463775635, 'total_time_str': '1m 20s', 'num_gpus': 1, 'snapshot_pkl': None, 'timestamp': 1763237845.608205} 
    
    def process_checkpoint(self, name: str, save_results:bool = True, save_dir: Path | None = None):
        decoded_videos_dirs = self.decode_checkpoint_latents(
            self._name_to_latent_path(name)
        )

        # match queries to decoded dirs
        query_to_video_dir = {k: v for k, v in zip(self.query.keys(), decoded_videos_dirs) if k in v.parts}

        rows = []
        for query, video_dir in query_to_video_dir.items():
            metric_results = self.calculate_metrics(
                video_dir,
                stylegan_only = 'generation' in query
            )
            print(metric_results)
            rows.extend(self._results_to_rows(name, query, metric_results))


        # Potentially save results
        if save_results:
            df = pd.DataFrame(rows)
            if save_dir is None:
                save_dir = self.results_dir / f"ckpt_{name}_results.csv"
        
            df.to_csv(save_dir, index=False)

        return rows

    def _results_to_rows(self, name, task, metric_results):
        rows = [
            {'checkpoint': name, 'task': task, 'type': k, 
             **v.get('stylegan_results', {}), 
             **{ pm['metric']: pm['mean'] for pm in v.get('summary', []) }
            } 
                for k, v in metric_results.items()
        ]
        return rows


    def process_checkpoints(self, save_results: bool = True, save_dir: Path | None = None, delete_latents_after: bool = False):
        rows = []
        for name in tqdm(self.names, desc="Processing checkpoints..."):
            row = self.process_checkpoint(name, save_results=False)
            rows.extend(row)
        
        results_df = pd.DataFrame(rows)
        if save_results:
            if save_dir is None:
                save_dir = self.results_dir / "all_checkpoints_results.csv"

            results_df.to_csv(save_dir, index=False)
            print(f"Saved all checkpoint results to {save_dir}")
        
        if delete_latents_after:
            is_df_empty = results_df.empty

            if is_df_empty:
                print("Results DataFrame is empty; skipping latent deletion in case there was an error.")
            else:
                self._delete_latents(self.names)

        self.results_df = results_df
        return results_df


    def plot_metrics(self, df=None, save_path=None):
        """Plot each metric vs epoch, grouped by task, and optionally save the figure."""
        if df is None:
            df = self.results_df

        df = df.copy()
        df[['epoch','step']] = pd.DataFrame(
            df['checkpoint'].apply(extract_epoch_step_from_checkpoint_str).tolist(), index=df.index
            )

        # support 'last' checkpoint: try to load real epoch/global_step from the saved checkpoint
        df = resolve_last_checkpoint_positions(
            df,
            load_last_ckpt_fn = lambda: torch.load(self._name_to_ckpt_path('last'))
        )

        metric_cols = [c for c in df.columns if c not in ('checkpoint','task','type','epoch','step')]
        metric_cols = [m for m in metric_cols if not df[m].dropna().empty]
        if not metric_cols:
            raise ValueError("No metrics to plot.")

        tasks = df['task'].unique()
        fig, axes = plt.subplots(nrows=len(metric_cols), figsize=(8, 3*len(metric_cols)), constrained_layout=True)
        if len(metric_cols) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metric_cols):
            plotted = False
            for task in tasks:
                sub = df[df['task'] == task].copy()
                if sub[metric].dropna().empty:
                    continue
                sub = sub.sort_values(['epoch','step'], na_position='last')
                ax.plot(sub['epoch'], sub[metric], marker='o', label=task)
                plotted = True
            if plotted:
                ax.set_title(metric)
                ax.set_xlabel('epoch')
                ax.set_ylabel(metric)
                ax.legend()
                ax.grid(True)
            else:
                ax.set_visible(False)

        if not save_path:
            save_path = self.results_dir / "metrics_plot.png"

        fig.savefig(save_path, dpi=800, bbox_inches='tight')

        return fig, axes
    
    def log_results_to_wandb(self, results_df=None):
        """
        Log self.results_df to the W&B run:
        - Upload the results as a W&B Table.
        - For each metric column, log a W&B-generated line plot with x=step and y=<metric>,
          with separate series per task. Supports 'last' checkpoint resolution like plot_metrics.
        """

        if results_df is None:
            if not hasattr(self, 'results_df') or self.results_df is None or self.results_df.empty:
                print("No results_df available; run process_checkpoints() first.")
                return
            df = self.results_df.copy()
        else:
            df = results_df.copy()
        # Attach to the same W&B run
        project, entity, run_id = self._get_wandb_info()
        init_kwargs = dict(project=project, entity=entity, id=run_id, resume='allow')
        if self.wandb_dir:
            init_kwargs['dir'] = str(self.wandb_dir)
        run = wandb.init(**init_kwargs)

        # Prepare dataframe: add epoch/step and resolve 'last'
        df[['epoch','step']] = pd.DataFrame(
            df['checkpoint'].apply(extract_epoch_step_from_checkpoint_str).tolist(), index=df.index
        )
        df = resolve_last_checkpoint_positions(
            df,
            load_last_ckpt_fn=lambda: torch.load(self._name_to_ckpt_path('last'))
        )
        df['step'] = pd.to_numeric(df['step'], errors='coerce')

        # Log the raw results as a W&B table
        table = wandb.Table(dataframe=df)
        wandb.log({"mid_train_evaluation/results_table": table})

        # Identify metric columns (exclude non-metrics)
        exclude = {'checkpoint', 'task', 'type', 'epoch', 'step'}
        metric_cols = [c for c in df.columns if c not in exclude and df[c].notna().any()]

        # Build W&B-generated line plots per metric, series grouped by task, x=step
        tasks = [t for t in df['task'].dropna().unique()]
        for metric in metric_cols:
            xs_list, ys_list, keys = [], [], []
            for task in tasks:
                sub = df[df['task'] == task][['step', metric]].dropna(subset=['step', metric]).copy()
                if sub.empty:
                    continue
                # Aggregate to one value per step per task (average across 'type' etc.)
                agg = sub.groupby('step', as_index=True)[metric].mean().sort_index()
                if agg.empty:
                    continue
                xs_list.append(agg.index.astype(int).tolist())
                ys_list.append(agg.values.astype(float).tolist())
                keys.append(str(task))

            if xs_list:
                chart = wandb.plot.line_series(
                    xs=xs_list,
                    ys=ys_list,
                    keys=keys,
                    title=f"{metric}",
                    xname="step"
                )
                wandb.log({f"mid_train_evaluation/metrics/{metric}": chart})

        run.finish()

def _log_df_to_wandb(eval_ckpt_cfg):
    df_path = Path(eval_ckpt_cfg.run_dir) / "sample_videos" / "mid_train_evaluation_results" / "all_checkpoints_results.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"Results file not found: {df_path}")
    
    df = pd.read_csv(df_path)
    EvaluateTrainProcess(eval_ckpt_cfg).log_results_to_wandb(df)



@hydra.main(version_base=None, config_path='configs', config_name='evaluate_ckpts')
def main(eval_ckpt_cfg: DictConfig):
    _main = False
    if _main:
        evaluator = EvaluateTrainProcess(eval_ckpt_cfg)
        df = evaluator.process_checkpoints(delete_latents_after=True)
        print("Final Results DataFrame:\n", df)

        # Log to wandb
        try:
            evaluator.log_results_to_wandb()
        except Exception as e:
            print(f"[WARN] Failed to log results to wandb: {e}")
        
        # Plot metrics
        try:
            evaluator.plot_metrics()
        except Exception as e:
            print(f"[WARN] Failed to plot metrics: {e}")
    else:
        _log_df_to_wandb(eval_ckpt_cfg)


if __name__ == "__main__":
    main()