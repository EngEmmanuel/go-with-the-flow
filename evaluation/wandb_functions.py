import sys
import wandb
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from wandb.apis.public.runs import Runs
from wandb.apis.public import Api
sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation.wandb_utils import yaml_metrics_to_nested_tables
from evaluation.metrics import collect_metric_results
from evaluation.ef_evaluation_schemes import SCHEME_REG
####### Convert yaml to wandb tables ###########

def _get_expected_keys(keys: tuple):
    expected_keys = {}
    for key in keys:
        if 'frame' in key:
            expected_keys['frame'] = key
        elif 'bin' in key:
            expected_keys['bin'] = key
        else:
            print(f"Unrecognized key in wandb table: {key}")
    return expected_keys



all_ = collect_metric_results(
    dir_base = Path('/users/spet4299/code/TEE/flow-matching/go-with-the-flow/outputs/hydra_outputs/2025-10-12/21-13-24/evaluation/decoded_videos/2025-10-20/23-32-43/ef_samples_in_range/reconstruction'),
    save_path=Path('/users/spet4299/code/TEE/flow-matching/go-with-the-flow/outputs/hydra_outputs/2025-10-12/21-13-24/evaluation/decoded_videos/2025-10-20/23-32-43/ef_samples_in_range/reconstruction'),
    make_tables=True
)


# Get all runs from a project that satisfy the filters
#filters = {"state": "finished", "config.optimizer": "adam"}
project = 'go-with-the-flow'
entity='engemmanuel'

api = wandb.Api()
sub_dir= 'evaluation/decoded_videos'
for run in api.runs(f"{entity}/{project}"):
    print(f"Run name: {run.name}")
    print(f"Run ID: {run.id}")
    print("----------")

    if run.name != 'quiet-bee-30':
        continue

    hydra_run_dir = run.summary.get('local_output_dir')
    if hydra_run_dir is None:

        print(
            f"No local_output_dir found in run summary for run {run.name} ({run.createdAt}). Skipping."
        )

        continue
    hydra_run_dir = Path(hydra_run_dir)
    eval_dir = hydra_run_dir / sub_dir if sub_dir else hydra_run_dir

    if not eval_dir.exists():
        if hydra_run_dir.exists():
            print(
                f"Evaluation directory {eval_dir} does not exist for run {run.name} ({run.createdAt}). Skipping.\n \
                Run evaluation code for it"
            )
        else:
            print(
                f"Hydra run directory {hydra_run_dir} does not exist for run {run.name} ({run.createdAt}). Skipping.\n \
                It was most like trained on a different machine."
            )
        continue
    metric_summary_paths = hydra_run_dir.rglob('**/all_metrics.yaml')
    evaluation_results = {}
    for metric_summary_path in metric_summary_paths:
        task = None
        sub_task = None
        for part in metric_summary_path.parts:
            if part in SCHEME_REG:
                task = part
            if part in ['reconstruction', 'generation']:
                sub_task = part
        assert (task is not None) and (sub_task is not None), f'Could not determine task and sub_task from path {metric_summary_path}'

        print(f"Processing metrics from: {metric_summary_path}")
        df, wtable = yaml_metrics_to_nested_tables(metric_summary_path)
        evaluation_results[task][sub_task] = wtable
        
        # SAVE THE DICT OF TABLES TO WANDB. UPDATE WITH NOVEL KEYS, IF DATA ALREADY EXISTS.


            
            
#TODO Finding a way to send the results wandb tables up to the run.




    break


















def log_evaluation_metrics():
    pass