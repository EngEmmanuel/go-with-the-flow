import sys
import subprocess
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dataset.echodataset import EchoDataset
from dataset.util import make_sampling_collate
from evaluation.functions import load_model_from_run, evaluate_to_latents
from evaluation.latents_to_videos import convert_latents_directory
from evaluation.metrics import compute_metrics_for_datasets
from utils import select_device


def main(cfg_path: Path | None = None):
    # Load evaluation config
    if cfg_path is None:
        cfg_path = Path(__file__).with_name("eval_cfg.yaml")
    eval_cfg = OmegaConf.load(cfg_path)

    tasks = set(eval_cfg.get("tasks", []))
    if not tasks:
        print("[info] No tasks specified in eval_cfg.tasks; nothing to do.")
        return

    device = select_device()
    run_dir = Path(eval_cfg.run_dir)

    latents_dir = None
    run_cfg = None
    model = None

    # Task: generate latents
    if "gen_latents" in tasks:
        model, run_cfg, _ = load_model_from_run(run_dir, ckpt_name=eval_cfg.get("ckpt_name", None))

        # Build test loaders per n_missing_frames setting
        test_ds_list = [
            EchoDataset(run_cfg, split="test", cache=False, n_missing_frames=nmf)
            for nmf in eval_cfg.test_n_missing_frames
        ]
        test_dl_list = [
            DataLoader(
                test_ds,
                batch_size=1,
                collate_fn=make_sampling_collate(
                    eval_cfg.n_ef_samples_in_range,
                    ef_gen_range=eval_cfg.test_ef_gen_range,
                ),
            )
            for test_ds in test_ds_list
        ]

        latents_dir = evaluate_to_latents(
            model=model,
            test_dl_list=test_dl_list,
            run_cfg=run_cfg,
            eval_cfg=eval_cfg,
            device=device,
        )

    # Task: convert latents to videos
    if "latents_to_videos" in tasks:
        if latents_dir is None:
            # Use provided directory from config if not generated in this run
            cfg_latents_dir = eval_cfg.get("latents_dir", None)
            if not cfg_latents_dir:
                raise ValueError("latents_to_videos requested but no latents_dir available; set eval_cfg.latents_dir or include gen_latents in tasks.")
            latents_dir = Path(cfg_latents_dir)

        convert_latents_directory(
            latents_dir=latents_dir,
            run_dir=run_dir,
            repo_id=eval_cfg.repo_id,
            output_dir=None,
            types=list(eval_cfg.types),
            query=str(eval_cfg.query) if eval_cfg.get("query", None) is not None else None,
            fps_metadata_csv=str(eval_cfg.fps_metadata_csv) if eval_cfg.get("fps_metadata_csv", None) else None,
            device=device,
        )

    # Task: compute metrics (keeps stylegan-v block, adds our metrics below)
    if "compute_metrics" in tasks:
        # Our image-quality metrics with CIs
        assert eval_cfg.get('metrics') is not None, "eval_cfg.metrics must be set to compute metrics."
        metrics_cfg = eval_cfg.get('metrics')
        real_root = Path(metrics_cfg.get('real_root'))
        fake_root = Path(metrics_cfg.get('fake_root'))


        print("[todo] stylegan-v metrics block: command prepared but not executed.")
        if eval_cfg.get('metrics').get('stylegan_metrics', None) is not None:
            stylegan_metrics = ','.join(eval_cfg['metrics']['stylegan_metrics'])
            n_gpus = 1
            cmd = [
                "python", "stylegan-v/src/scripts/calc_metrics_for_dataset.py",
                "--real_data_path", real_root,
                "--fake_data_path", fake_root,
                "--gpus", str(n_gpus),
                "--resolution", "112",
                "--metrics", stylegan_metrics,
            ]

            out = subprocess.run(cmd, capture_output=True, text=True)
            print(out.stdout)

        # Pairwise metrics with CIs
        if eval_cfg.metrics.get('which') is not None:
            metrics = metrics_cfg.which
            confidence = float(metrics_cfg.get('confidence', 0.95))
            resize = metrics_cfg.get('resize', None)
            if resize is not None:
                resize = tuple(resize)
            max_videos = metrics_cfg.get('max_videos', None)

            out_paths = compute_metrics_for_datasets(
                real_root=real_root,
                fake_root=fake_root,
                metrics=metrics,
                output_dir=Path(eval_cfg.output_dir),
                confidence=confidence,
                resize=resize,
                device=device,
                max_videos=max_videos,
            )
            print(f"[metrics] Wrote summary YAML under: {out_paths['result_yaml']}")


if __name__ == "__main__":
    eval_cfg_path = 'evaluation/eval_cfg.yaml'
    main(eval_cfg_path)