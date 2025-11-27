import re
import wandb
import torch
import hydra
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import select_device
from utils.evaluation import *
from dataset.util import default_eval_collate
from dataset.echodataset import EchoDataset
from my_src.custom_callbacks import sample_latents_from_model
from evaluation.functions import load_model_from_run, _get_run_config
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

    def decode_checkpoint_latents(self, latents_dir: Path, kwargs={}) -> list[Path]:
        'Evaluates a single checkpoint and saves outputs to output_dir'
        
        default_kwargs = {
            'real_data_path': Path(self.cfg.real_data_path),
            'run_dir': getattr(self, "run_dir", None),
            'repo_id': self.cfg.repo_id,
            'types': self.cfg.types,
            'fps_metadata_csv': self.cfg.fps_metadata_csv,
            'decode_batch_size': self.cfg.get('decode_batch_size', 32),
            'device': self.device,
            'debugging': self.cfg.get('debugging', False),
        }
        kwargs = {**default_kwargs, **kwargs}

        dv_dir = kwargs.get('dv_dir', "")
        kwargs.pop('dv_dir', None)

        decoded_videos_dirs = []
        for name, query in self.query.items():
            output_dir = self.output_dir / dv_dir / latents_dir.name / name
            decoded_videos_dir = convert_latents_directory(
                latents_dir = latents_dir,
                query = {'name': name, 'pattern': query},
                output_dir = output_dir,
                **kwargs
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
 

    def process_checkpoint(self, name: str | None = None, save_results:bool = True, save_dir: Path | None = None, latent_dir=None, kwargs={}):
        latent_dir = self._name_to_latent_path(name) if latent_dir is None else latent_dir
        decoded_videos_dirs = self.decode_checkpoint_latents(
            latents_dir=latent_dir,
            kwargs=kwargs
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





###########################################################################################
#                               MultiRunAndStepETPWrapper                                 #
###########################################################################################


def get_top_checkpoints(df, task, metric="fid50k_full", top_k=2, smaller_is_better=True):
    sub = df[df["task"] == task]
    sub = sub.sort_values(metric, ascending=smaller_is_better)
    return list(sub.head(top_k)[["checkpoint", metric]].itertuples(index=False, name=None))

class MultiRunAndStepETPWrapper(EvaluateTrainProcess):
    '''
    Class that takes in multiple run directories and a checkpoint selection function,
    evaluates the selected checkpoints from each run, and aggregates the results. Each
    checkpoint is evaluated using a varying number of inference steps specified in cfg.

    Just like EvaluateTrainProcess, an output CSV file is created with all results
    '''
    def __init__(self, cfg, ckpt_selection_fn: callable, debug: bool = False):
        self.cfg = cfg
        self.run_dirs = [Path(rd) for rd in cfg.multirun_eval.run_dirs]
        self.steps = cfg.multirun_eval.n_inference_steps
        self.override_old_run = cfg.multirun_eval.get('override_old_run', True)
        self.ckpt_selection_fn = ckpt_selection_fn
        self.output_dir = Path(cfg.multirun_eval.output_dir)
        self.device = select_device()
        self.debug = debug

        self._get_ckpt_etp_results()

        self.results_dir = self.output_dir / 'multirun_results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.query = {
            'reconstruction-nmfmax': "rec_or_gen == 'rec' and n_missing_frames == 'max'",
            'generation-nmfmax': "rec_or_gen == 'gen' and n_missing_frames == 'max'"
        }

    def _get_wandb_info(self, run_dir, wandb_dir=None):
        'Gets wandb project, entity, and run id from the run directory'
        if self.debug:
            return 'temp_project', 'temp_entity', 'temp_run_id'
        
        run_cfg_dir = Path(run_dir) / '.hydra' / 'config.yaml'
        run_cfg = OmegaConf.load(run_cfg_dir)
        project = run_cfg.wandb.init_kwargs.project
        entity = run_cfg.wandb.init_kwargs.entity

        # Get run id
        if wandb_dir is None:
            wandb_dir = Path(run_dir) / "wandb"
        match = [x for x in wandb_dir.glob("run-*")]
        if not match:
            raise ValueError("No wandb run directory found.")
        if len(match) > 1:
            print("Multiple wandb run directories found; using the first one.")
            print([x.name for x in match])
        
        run_id = match[0].name.split("-")[-1]

        return project, entity, run_id
    
    def _get_run_wandb_name(self, run_dir, wandb_dir=None):
        if self.debug:
            random_n = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=3))
            return f"temp_{random_n}"
        
        'Gets wandb project, entity, and run id from the run directory'
        project, entity, run_id = self._get_wandb_info(run_dir, wandb_dir)
        init_kwargs = dict(project=project, entity=entity, id=run_id, resume='allow')
        run = wandb.init(**init_kwargs)
        return run.name


    def _prepare_dataloader(self, cfg):
        sample_ds = EchoDataset(cfg, split='sample', n_missing_frames='max')
        sample_dl = DataLoader(sample_ds, batch_size=8, shuffle=False, collate_fn=default_eval_collate)
        return sample_dl
    

    def _get_ckpt_etp_results(self):
        ckpts = {}
        for run_dir in self.run_dirs:
            wandb_name = self._get_run_wandb_name(run_dir)
            wandb_dir = run_dir / "wandb"
            etp_results_dir = run_dir / "sample_videos" / "mid_train_evaluation_results" / "all_checkpoints_results.csv"
            if not etp_results_dir.exists():
                print(f"Warning: ETP results file not found for run {run_dir}: {etp_results_dir}")
                continue
            df = pd.read_csv(etp_results_dir)

            ckpts[wandb_name] = []
            ckpt_names = self.ckpt_selection_fn(df)
            for ckpt_name, score in ckpt_names:
                if ckpt_name == 'last':
                    epoch = 2000
                    step = 250_000
                else:
                    epoch, step = extract_epoch_step_from_checkpoint_str(ckpt_name)

                ckpts[wandb_name].append({
                    'name': f"{ckpt_name}.ckpt",
                    'epoch': epoch,
                    'step': step,
                    'score': score,
                    'wandb_name': wandb_name,
                    'run_dir': str(run_dir)
                })

            
        self.ckpts = ckpts


    def sample_latents_from_ckpt(self, ckpt_dict, app='', kwargs={}):
        run_dir = ckpt_dict['run_dir']
        run_cfg = _get_run_config(Path(run_dir))
        sample_dl = self._prepare_dataloader(run_cfg)
        dummy_data = sample_dl.dataset[0]
        model, _ = load_model_from_run(run_dir, dummy_data=dummy_data, ckpt_name=ckpt_dict['name'])
        
        output_dir = self.output_dir / 'latents' / ckpt_dict['wandb_name'] / f"epoch={ckpt_dict['epoch']}-step={ckpt_dict['step']}{app}"
        if not self.override_old_run and output_dir.exists():
            print(f"Latents already exist for {output_dir}, skipping sampling.")
            return output_dir

        latents_dir = sample_latents_from_model(
            model=model,
            dl_list=[sample_dl],
            run_cfg=run_cfg,
            epoch=ckpt_dict['epoch'],
            step=ckpt_dict['step'],
            device=self.device,
            samples_dir=self.output_dir / 'latents' / ckpt_dict['wandb_name'],
            out_name=f"epoch={ckpt_dict['epoch']}-step={ckpt_dict['step']}{app}",
            debug=self.debug,
            kwargs=kwargs
        )

        return latents_dir


    def sample_all_latents(self):
        latents_dict = []
        for wandb_name, ckpts in self.ckpts.items():
            for ckpt_dict in ckpts:
                for steps in self.steps:
                    latents_dir = self.sample_latents_from_ckpt(
                        ckpt_dict, 
                        app=f"-inf_steps={steps}", 
                        kwargs={'model_sample_kwargs': {'steps': steps}}
                    )
                    
                    latents_dict.append({**ckpt_dict, 'inf_steps': steps, 'latents_dir': latents_dir})
                    if self.debug:
                        break

        return latents_dict


    def decode_checkpoint_latents(self, latents_dir: Path, kwargs={}) -> list[Path]:
        return super().decode_checkpoint_latents(latents_dir, kwargs=kwargs)


    def process_checkpoint(self, name=None, save_results = False, save_dir = None, latents_dir=None, **kwargs):
        pc_kwargs = {'run_dir': kwargs['run_dir'], 'dv_dir': f"decoded_videos/{kwargs['wandb_name']}"}
        rows = super().process_checkpoint(name, save_results, save_dir, latents_dir, kwargs=pc_kwargs)
        rows = [{**r, **kwargs} for r in rows]
        return rows


    def plot_metrics_vs_inf_steps(self, df, save_results=True, save_path=None):
        """
        Plot each metric vs inf_steps.
        X-axis: inf_steps
        Y-axis: metric value
        Separate line per (checkpoint, task).
        Aggregates (mean) over duplicate (checkpoint, task, inf_steps) rows (e.g. differing 'type').

        Styling:
        - Color encodes checkpoint
        - Linestyle/marker encodes task
        - Separate compact legend for task linestyles/markers
        """

        work_df = df.copy()
        # Ensure numeric
        work_df['inf_steps'] = pd.to_numeric(work_df['inf_steps'], errors='coerce')
        work_df = work_df.dropna(subset=['inf_steps'])

        exclude = {'checkpoint','task','type','inf_steps','wandb_name','epoch','step','run_dir'}
        metrics = [c for c in work_df.columns if c not in exclude]
        # Keep only columns with at least one non-null numeric value
        metrics = [m for m in metrics if pd.to_numeric(work_df[m], errors='coerce').notna().any()]

        if not metrics:
            raise ValueError("No valid metrics to plot.")

        # Aggregate
        grouped = work_df.groupby(['checkpoint','task','inf_steps']).mean(numeric_only=True).reset_index()

        # Styling maps
        checkpoints = sorted(grouped['checkpoint'].dropna().unique().tolist())
        tasks = sorted(grouped['task'].dropna().unique().tolist())

        cmap = plt.get_cmap('tab20')
        color_map = {ckpt: cmap(i % cmap.N) for i, ckpt in enumerate(checkpoints)}
        # cycle of linestyles/markers for tasks
        style_cycle = [
            ('-', 'o'),
            ('--', 's'),
            ('-.', '^'),
            (':', 'D'),
            ('-', 'v'),
            ('--', 'P'),
            ('-.', 'X'),
            (':', 'h'),
        ]
        task_style = {t: style_cycle[i % len(style_cycle)] for i, t in enumerate(tasks)}

        # Subplot grid: at least two columns
        n = len(metrics)
        cols = max(2, min(3, n))  # at least 2 columns, up to 3 if many metrics
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.2*rows), constrained_layout=True)
        axes = np.array(axes).reshape(-1)
        # Hide any extra axes
        for ax in axes[n:]:
            ax.set_visible(False)

        for ax, metric in zip(axes[:n], metrics):
            sub = grouped[['checkpoint','task','inf_steps', metric]].dropna(subset=[metric])
            if sub.empty:
                ax.set_visible(False)
                continue

            # Plot lines per (checkpoint, task) with color by checkpoint and linestyle/marker by task
            for (ckpt, task), g in sub.groupby(['checkpoint','task']):
                g = g.sort_values('inf_steps')
                ls, mk = task_style.get(task, ('-', 'o'))
                ax.plot(
                    g['inf_steps'], g[metric],
                    label=f"{ckpt}|{task}",  # label not used in legend to reduce clutter
                    color=color_map.get(ckpt, 'gray'),
                    linestyle=ls,
                    marker=mk,
                    linewidth=1.5,
                    markersize=4
                )

            ax.set_title(metric)
            ax.set_xlabel("inf_steps")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)

            # Build a compact legend for tasks (linestyle/marker only)
            task_handles = [
                Line2D([0], [0],
                       color='black',
                       linestyle=task_style[t][0],
                       marker=task_style[t][1],
                       linewidth=1.5,
                       markersize=5,
                       label=str(t))
                for t in tasks
                if not sub[sub['task'] == t].empty
            ]
            if task_handles:
                ax.legend(handles=task_handles, title="Task", fontsize='x-small', title_fontsize='x-small', loc='best')

        if save_results:
            if save_path is None:
                save_path = self.results_dir / "metrics_vs_inf_steps.png"
            fig.savefig(save_path, dpi=600, bbox_inches='tight')
        return fig, axes


    def main(self, latents_dicts=None):
        # Latent sampling
        if latents_dicts is None:
            latents_dicts = self.sample_all_latents()
        print(f"Latents Dict:\n {latents_dicts}")

        # Video decoding and metrics
        rows = []
        for latents_dict in latents_dicts:
            latent_rows = self.process_checkpoint(
                name=latents_dict['name'].replace('.ckpt',''),
                latents_dir=latents_dict['latents_dir'],
                inf_steps=latents_dict['inf_steps'],
                wandb_name=latents_dict['wandb_name'],
                epoch=latents_dict['epoch'],
                step=latents_dict['step'],
                run_dir=latents_dict['run_dir']
            )

            rows.extend(latent_rows)
        print(f"Rows: {rows}")
        
        results_df = pd.DataFrame(rows)
        save_dir = self.results_dir / 'multirun_checkpoints_results.csv'
        results_df.to_csv(save_dir, index=False)
        print(f"Saved multirun checkpoint results to {save_dir}")

        # Results plotting
        self.plot_metrics_vs_inf_steps(results_df)

            
###########################################################################################
#                                           MAIN                                          #
###########################################################################################


@hydra.main(version_base=None, config_path='configs', config_name='multi_run_step_eval') #config_name='evaluate_ckpts')
def main(eval_ckpt_cfg: DictConfig):
    def mrsetp(debug):
        multirun_eval = MultiRunAndStepETPWrapper(
            eval_ckpt_cfg,
            ckpt_selection_fn=lambda df: get_top_checkpoints(
                df, 
                task='reconstruction-nmfmax', 
                metric='fid50k_full', 
                top_k=3, 
                smaller_is_better=True
            ),
            debug=debug
        )

        multirun_eval.main()



    def etp(_main= False):
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


    etp_flag = False

    if etp_flag:
        etp(_main=True)
    else:
        mrsetp(debug=False)


if __name__ == "__main__":
    main()