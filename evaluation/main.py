import sys
import json
import time
import hydra
import subprocess
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from typing import Any, Dict
from pprint import pprint
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig


from dataset.echodataset import EchoDataset
from dataset.util import make_sampling_collate
from evaluation.functions import _get_run_config, load_model_from_run
from evaluation.ef_evaluation_schemes import generate_dls_for_evaluation_scheme, run_inference
from evaluation.latents_to_videos import convert_latents_directory
from evaluation.metrics import compute_metrics_for_datasets, collect_metric_results
from utils import select_device


print_line_rule = lambda: print('\n'*2, '-'*150, flush=True)



@hydra.main(version_base=None, config_path="configs", config_name="debug_full_eval_cfg")
def main(eval_cfg: DictConfig):
    tasks = set(eval_cfg.get("tasks", []))
    if not tasks:
        print("[info] No tasks specified in eval_cfg.tasks; nothing to do.")
        return

    device = select_device()
    run_dir = Path(eval_cfg.run_dir)

    latents_dirs = None
    decoded_videos_dir = None
    run_cfg = None
    model = None
    
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    datetime_tuple = hydra_output_dir.parts[-2:]


    # Task: generate latents
    if "gen_latents" in tasks:
        assert eval_cfg.get('latents_dir', None) is None, "eval_cfg.latents_dir should not be set if gen_latents is in tasks."
        print(f"[info] Generating latents for run_dir: {run_dir}")

        
        # Load model from run directory
        run_cfg = _get_run_config(run_dir)
        # Make path agnostic
        run_cfg.paths = eval_cfg.paths
        # prepare dataloaders
        all_dataloaders = generate_dls_for_evaluation_scheme(run_cfg, eval_cfg)

        dummy_data = next(iter(all_dataloaders.values()))[0].dataset[0]
        model, _ = load_model_from_run(run_dir, dummy_data=dummy_data, ckpt_name=eval_cfg.get("ckpt_name"))

        latents_dirs = run_inference(
            eval_cfg=eval_cfg, 
            run_cfg=run_cfg, 
            model=model, 
            device=device, 
            dataloaders=all_dataloaders,
            datetime_tuple=datetime_tuple
        )



    # Task: convert latents to videos
    if "latents_to_videos" in tasks:
        print_line_rule()
        print(f"[info] Converting latents to videos for run_dir: {run_dir}")
        if latents_dirs is None: # if tru, use cfg latents_dirs
            # Use provided directory from config if not generated in this run
            cfg_latents_dirs = eval_cfg.get("latents_dirs", None)
            if not cfg_latents_dirs:
                raise ValueError("latents_to_videos requested but no latents_dir available; set eval_cfg.latents_dir or include gen_latents in tasks.")
            
            #latents_dirs = [Path(p) for p in cfg_latents_dirs]
            latents_dirs = {k: Path(v) for k,v in cfg_latents_dirs.items()}


        decoded_videos_dirs = {k: {} for k in latents_dirs}  # scheme_name -> {query_name: path}
        for scheme_name, latents_dir in latents_dirs.items():
            assert latents_dir.exists(), f"Latents directory does not exist: {latents_dir}"

            queries = eval_cfg.get('queries', {}).get(scheme_name, None)
            if queries is not None:
                queries = OmegaConf.to_container(queries, resolve=True)
            else:
                queries = {"all": "True"}

            start_t = time.perf_counter()
            for name, query in queries.items():
                decoded_videos_dir = convert_latents_directory(
                    real_data_path=Path(eval_cfg.real_data_path),
                    latents_dir=latents_dir,
                    run_dir=run_dir,
                    repo_id=eval_cfg.repo_id,
                    output_dir=None,
                    types=list(eval_cfg.types),
                    query={'name':name,'pattern':query},
                    fps_metadata_csv=str(eval_cfg.fps_metadata_csv) if eval_cfg.get("fps_metadata_csv", None) else None,
                    device=device,
                    debugging=eval_cfg.get('debugging', False),
                )
                decoded_videos_dirs[scheme_name][name] = decoded_videos_dir

                print(f"[info] Wrote decoded videos under: {decoded_videos_dir}")
            end_t = time.perf_counter()
            elapsed = end_t - start_t
            print(f"[timing][L2V] {scheme_name}: {elapsed / 3600:.4f} hours", flush=True)


    if "compute_metrics" in tasks:
        print_line_rule()
        metrics_cfg = eval_cfg.get('metrics')
        print(f"[info] Computing metrics for run_dir: {run_dir}", flush=True)

        if decoded_videos_dir is None:
            inference_roots = metrics_cfg.get('inference_roots', None)
            assert inference_roots is not None, "eval_cfg.metrics.inference_root must be set if latents_to_videos is not run here."

            inference_roots = {
                scheme: {name: Path(path) for name, path in group.items()} 
                for scheme, group in inference_roots.items()
            }
        else:
            inference_roots = decoded_videos_dirs

        real_glob = metrics_cfg.get('real_glob', '*')
        fake_glob = metrics_cfg.get('fake_glob', '*')

        # StyleGAN-V metrics
        print(f"[metrics] inference roots: {inference_roots}", flush=True)
        stylegan_metrics = metrics_cfg.get('stylegan_metrics', None)

        print_line_rule()
        print(f"[info] Computing StyleGAN metrics", flush=True)
        n_gpus = 1

        stylegan_results = {scheme: {'image_metrics': {}, 'video_metrics': {}} for scheme in inference_roots}
        for scheme, group in inference_roots.items():
            print(f"\n[info] Evaluation scheme: {scheme}", flush=True)
            start_t = time.perf_counter()
            for name, inference_root in group.items():
                print(f"\n[info] Computing StyleGAN metrics for: {scheme}:{name}", flush=True)

                # Folder of frames and videos
                for sub_dir in [f'framewise{x}' for x in ['', '_no_pad', '_generated', '_stitched']]:
                    inference_root_sub = inference_root / sub_dir
                    if not inference_root_sub.exists():
                        continue
                    
                    print_line_rule()
                    print('\n', sub_dir.capitalize(), flush=True)

                    image_metrics = [x for x in stylegan_metrics if ('fvd' not in x and 'isv' not in x)]
                    video_metrics = [x for x in stylegan_metrics if ('fvd' in x or 'isv' in x)]

                    # Video metrics
                    vid_results_json = {}
                    if ('stitched' in sub_dir) or ('no_pad' in sub_dir) and len(video_metrics) > 0:
                        vid_results_json = run_stylegan_metrics(
                            inference_root_sub=inference_root_sub,
                            stylegan_metrics=','.join(video_metrics),
                            n_gpus=n_gpus,
                        )
                        stylegan_results[scheme]['video_metrics'][sub_dir] = vid_results_json['results']

                    # Image metrics
                    img_results_json = run_stylegan_metrics(
                        inference_root_sub=inference_root_sub,
                        stylegan_metrics=','.join(image_metrics),
                        n_gpus=n_gpus,
                    )
                    stylegan_results[scheme]['image_metrics'][sub_dir] = img_results_json['results']


                    # Pairwise metrics with CIs
                    pairwise_metrics = eval_cfg.metrics.get('pairwise_metrics')
                    if pairwise_metrics is not None: # pairwise only makes sense for reconstruction

                        just_save_payload = ('generation' in name)
                        print(f"\n[info] Computing pairwise metrics for: {name}", flush=True)
                        
                        confidence = float(metrics_cfg.get('confidence', 0.95))
                        resize = metrics_cfg.get('resize', None)
                        if resize is not None:
                            resize = tuple(resize)

                        out_paths = compute_metrics_for_datasets(
                            inference_root=inference_root_sub,
                            metrics=pairwise_metrics,
                            output_dir=Path(eval_cfg.output_dir),
                            confidence=confidence,
                            resize=resize,
                            device=device,
                            real_glob=real_glob,
                            fake_glob=fake_glob,
                            payload_kwargs={
                                'stylegan_results': img_results_json['results'] | vid_results_json.get('results', {})
                                },
                            just_save_payload=just_save_payload,
                        )
                        print(f"[metrics] Wrote summary YAML under: {out_paths['result_yaml']}")
                    else:
                        print(f"[info] Skipping pairwise metrics for: {name}", flush=True)

                pprint(stylegan_results)

                # Collect metric results into tables
                for parent in inference_root.parents:
                    if parent.name in ['reconstruction', 'generation']:
                        collect_metric_results(
                            dir_base=parent,
                            save_path=parent,
                            make_tables=True,
                        )
            end_t = time.perf_counter()
            elapsed = end_t - start_t
            print(f"[timing][metrics] {scheme}: {elapsed / 3600:.4f} hours", flush=True)



def run_stylegan_metrics(inference_root_sub: Path, stylegan_metrics: str, n_gpus: int) -> Dict[str, Any]:
    cmd = [
        "python", "src/scripts/calc_metrics_for_dataset.py",
        "--real_data_path", str(inference_root_sub / "real"),
        "--fake_data_path", str(inference_root_sub / "fake"),
        "--gpus", str(n_gpus),
        "--resolution", "112",
        "--metrics", stylegan_metrics,
        "--use_cache", '0'
    ]

    try:
        cp = subprocess.run(cmd, cwd="stylegan-v", text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        # Print what you would have seen, then re-raise
        if e.stdout: print(e.stdout, end="")
        if e.stderr: print(e.stderr, end="", file=sys.stderr)
        raise

    # Print everything you would have seen
    if cp.stdout: print(cp.stdout, end="")
    if cp.stderr: print(cp.stderr, end="", file=sys.stderr)

    # Look for the last JSON line in stdout, then stderr
    for stream in (cp.stdout, cp.stderr):
        for line in reversed((stream or "").splitlines()):
            s = line.strip()
            if s.startswith('{"results":') and s.endswith('}'):
                return json.loads(s)

    raise RuntimeError('Couldn\'t find a JSON line starting with {"results": ... } in the output.')

    

if __name__ == "__main__":
    main()


