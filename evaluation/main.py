import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from torch.utils.data import DataLoader
import torch

from dataset.echodataset import EchoDataset
from evaluation.functions import load_model_from_run


run_dir = Path('outputs/hydra_outputs/2025-09-02/16-54-08')
model, cfg, ckpt_path = load_model_from_run(run_dir, ckpt_name=None)

test_ds = EchoDataset(cfg, split="test")
test_dl = DataLoader(test_ds, batch_size=1)

device = next(model.parameters()).device
for i, batch in enumerate(test_dl):
    cond = batch["cond_image"].to(device, non_blocking=True)
    ehs = batch["encoder_hidden_states"].to(device, non_blocking=True)
    B = cond.shape[0]

    with torch.no_grad():
        samples = model.sample(
            encoder_hidden_states=ehs,
            cond_image=cond,
            batch_size=B,
            data_shape=test_ds.shape  # (T, C, H, W)
        )

    print(f"[{i}] sampled shape:", samples.shape)
    # break  # remove to process

