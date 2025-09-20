import sys
import hydra
import subprocess
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

from dataset.echodataset import EchoDataset
from dataset.util import make_sampling_collate
from evaluation.functions import load_model_from_run, evaluate_to_latents
from evaluation.latents_to_videos import convert_latents_directory
from evaluation.metrics import compute_metrics_for_datasets
from utils import select_device

@hydra.main(version_base=None, config_path="configs", config_name="eval_cfg")
def main(eval_cfg: DictConfig):
    tasks = set(eval_cfg.get("tasks", []))
    if not tasks:
        print("[info] No tasks specified in eval_cfg.tasks; nothing to do.")
        return

    device = select_device()
    run_dir = Path(eval_cfg.run_dir)

    latents_dir = None
    decoded_videos_dir = None
    run_cfg = None
    model = None

    # Task: generate latents
    if "gen_latents" in tasks:
        assert eval_cfg.get('latents_dir', None) is None, "eval_cfg.latents_dir should not be set if gen_latents is in tasks."
        print(f"[info] Generating latents for run_dir: {run_dir}")
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
        print('\n'*2, '-'*150)
        print(f"[info] Converting latents to videos for run_dir: {run_dir}")
        if latents_dir is None:
            # Use provided directory from config if not generated in this run
            cfg_latents_dir = eval_cfg.get("latents_dir", None)
            if not cfg_latents_dir:
                raise ValueError("latents_to_videos requested but no latents_dir available; set eval_cfg.latents_dir or include gen_latents in tasks.")
            latents_dir = Path(cfg_latents_dir)

        decoded_videos_dir = convert_latents_directory(
            real_data_path=Path(eval_cfg.real_data_path),
            latents_dir=latents_dir,
            run_dir=run_dir,
            repo_id=eval_cfg.repo_id,
            output_dir=None,
            types=list(eval_cfg.types),
            query=str(eval_cfg.query) if eval_cfg.get("query", None) is not None else None,
            fps_metadata_csv=str(eval_cfg.fps_metadata_csv) if eval_cfg.get("fps_metadata_csv", None) else None,
            device=device,
        )

        print(f"[info] Wrote decoded videos under: {decoded_videos_dir}")
    # Task: compute metrics (keeps stylegan-v block, adds our metrics below)
    if "compute_metrics" in tasks:
        print('\n'*2, '-'*150, flush=True)
        print(f"[info] Computing metrics for run_dir: {run_dir}", flush=True)
        # Our image-quality metrics with CIs
        metrics_cfg = eval_cfg.get('metrics')

        assert metrics_cfg is not None, "eval_cfg.metrics must be set to compute metrics."
        if decoded_videos_dir is None:
            inference_root = Path(metrics_cfg.get('inference_root', None))
            assert inference_root is not None, "eval_cfg.metrics.inference_root must be set if latents_to_videos is not run here."
        else:
            inference_root = decoded_videos_dir 

        real_glob = metrics_cfg.get('real_glob', '*')
        fake_glob = metrics_cfg.get('fake_glob', '*')

        print(f"[metrics] inference root: {inference_root}", flush=True)
        if metrics_cfg.get('stylegan_metrics', None) is not None:
            print('\n'*2, '-'*150, flush=True)
            print(f"[info] Computing StyleGAN metrics", flush=True)
            stylegan_metrics = metrics_cfg['stylegan_metrics']
            n_gpus = 1
            for sub_dir in [f'framewise{x}' for x in ['', '_no_pad', '_generated', '_stitched']]:
                inference_root_sub = inference_root / sub_dir
                if not inference_root_sub.exists():
                    continue
                print('\n'*2,' '*6, '-*'*100, flush=True)
                print(sub_dir.capitalize(), flush=True)

                image_metrics = [x for x in stylegan_metrics if ('fvd' not in x and 'isv' not in x)]
                video_metrics = [x for x in stylegan_metrics if ('fvd' in x or 'isv' in x)]

                # Video metrics
                if 'stitched' in sub_dir and len(video_metrics) > 0:
                    run_stylegan_metrics(
                        inference_root_sub=inference_root_sub,
                        stylegan_metrics=','.join(video_metrics),
                        n_gpus=n_gpus,
                    )

                # Image metrics
                run_stylegan_metrics(
                    inference_root_sub=inference_root_sub,
                    stylegan_metrics=','.join(image_metrics),
                    n_gpus=n_gpus,
                )


                #TODO: Need a way to store the stylegan metrics output in the final yaml
                # Pairwise metrics with CIs
                if eval_cfg.metrics.get('which') is not None:
                    assert eval_cfg.mode == 'rec', "Pairwise metrics can only be used to evaluate reconstruction ['rec'  mode]."
                    metrics = metrics_cfg.which
                    confidence = float(metrics_cfg.get('confidence', 0.95))
                    resize = metrics_cfg.get('resize', None)
                    if resize is not None:
                        resize = tuple(resize)

                    out_paths = compute_metrics_for_datasets(
                        inference_root=inference_root_sub,
                        metrics=metrics,
                        output_dir=Path(eval_cfg.output_dir),
                        confidence=confidence,
                        resize=resize,
                        device=device,
                        real_glob=real_glob,
                        fake_glob=fake_glob,
                    )
                    print(f"[metrics] Wrote summary YAML under: {out_paths['result_yaml']}")


def run_stylegan_metrics(inference_root_sub, stylegan_metrics, n_gpus):
    cmd = [
        "python", "src/scripts/calc_metrics_for_dataset.py",
        "--real_data_path", str(inference_root_sub / 'real'),
        "--fake_data_path", str(inference_root_sub / 'fake'),
        "--gpus", str(n_gpus),
        "--resolution", "112",
        "--metrics", stylegan_metrics,
    ]

    print('StyleGAN Metrics Terminal Command:', ' '.join(cmd))
    try:
        out = subprocess.run(cmd, cwd='stylegan-v', check=True, capture_output=False, text=True)
        print('StyleGAN Metrics Terminal Output:', out.stdout, '\n', out.stderr)
    except subprocess.CalledProcessError as e:
        print('StyleGAN Metrics Terminal Error:', e.stderr, '\n', e.stdout)
    

if __name__ == "__main__":
    main()
    #TODO: get stylegan working