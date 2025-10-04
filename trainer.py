
import torch
import wandb 
import hydra
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.nn import Module, ModuleList
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Literal, Callable
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
#from pytorch_lightning.callbacks.weight_averaging import WeightAveraging

from my_src.models import UNetSTIC, DiffuserSTDiT
from dataset.testdataset import FlowTestDataset
from dataset.echodataset import EchoDataset
from my_src.flows import LinearFlow
from dataset import make_sampling_collate
from vae.util import load_vae_and_processor


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def select_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

#TODO Where is best to include the VAE?

# Model conditioned on time is assumed True

### Trainer
class FlowVideoGenerator(LightningModule):
    def __init__(
            self,
            model,
            cfg,
            **kwargs
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.sample_dir = kwargs.get("sample_dir", None)
        self.sample_dl = kwargs.get("sample_dl", None)
        self.sample_dl = cycle(self.sample_dl) if self.sample_dl is not None else None

        self._val_counter = 0

        # learned 1-D null EF
        self.null_ehs = torch.nn.Parameter(torch.zeros(1, 1))  # shape [1,1]
        self.uncond_prob = float(cfg.trainer.get('uncond_prob', 0.0))

    def training_step(self, batch, batch_idx):
        batch = self.maybe_drop_cond(batch)
        loss = self.model(**batch)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if self.sample_dir is None or batch_idx != 0:
            return

        is_sample_step = (self._val_counter % self.cfg.sample_every_n_val_steps == 0)
        if self.trainer.is_global_zero and is_sample_step:
            self.print(f"Sampling at step {self.global_step}")

            self.sample()

        # Increment counter
        self._val_counter += 1

    @torch.no_grad()
    def sample(self, n_videos_per_sample=2):

        self.model.eval()
        sample_results = {}

        for n in range(n_videos_per_sample):
            reference_batch, repeated_batch = next(self.sample_dl)
            batch_size, *_ = repeated_batch['cond_image'].shape

            repeated_batch = {k: v.to(self.device) for k, v in repeated_batch.items()}

            sampled_videos = self.model.sample(**repeated_batch, batch_size=batch_size)
            sampled_videos = sampled_videos.detach().cpu()
            sampled_videos /= self.cfg.vae.scaling_factor

            cond_image = reference_batch['cond_image'] / self.cfg.vae.scaling_factor
            ef_values = reference_batch['ef_values']

            sample_results[n] = {
                "video_name": reference_batch['video_name'],
                "cond_image": cond_image.contiguous(),
                'reconstructed': (sampled_videos[0,...].contiguous(), round(ef_values[0].item(), 3)),
                'generated': (sampled_videos[1:,...].contiguous(), [round(x.item(), 3) for x in ef_values[1:]])
            }

        torch.save(
            sample_results, 
            self.sample_dir / f"sampled_videos_step_{self.global_step}.pt"
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.cfg.trainer.lr)
        return optimizer
    
    def maybe_drop_cond(self, batch):
        ehs = batch.get('encoder_hidden_states', None)  # [B, 1]
        if ehs is None or self.uncond_prob <= 0.0:
            return batch

        B = ehs.shape[0]
        # boolean mask (no in-place ops)
        drop = (torch.rand(B, device=ehs.device) < self.uncond_prob)  # True => drop EF

        if drop.any():
            m = drop.view(B, 1)                               # [B,1], bool
            null = self.nullf_ehs.expand_as(ehs)               # shape-match; dtype/device handled by PL
            ehs = torch.where(m, null, ehs)                   # functional, no in-place
            batch['encoder_hidden_states'] = ehs

        # log right before returning (only runs when CFG-style dropout is active)
        self.log("train/null_ehs_value", self.null_ehs.item(),
                on_step=True, prog_bar=False, logger=True, rank_zero_only=True)
        self.log("train/ef_drop_rate", drop.float().mean().item(),
                on_step=True, prog_bar=False, logger=True, rank_zero_only=True)
        return batch


def load_model(cfg, dummy_data, device):
    if isinstance(dummy_data, dict):
        C, T, H, W = dummy_data['x'].shape
        Cc, _, _, _ = dummy_data['cond_image'].shape
    else:
        _, C, H, W = dummy_data
        Cc = C + 1
        T =32
    print('input shape:', (C+Cc, T, H, W))

    match (cfg.model.type).lower():
        case "unet":
            return UNetSTIC(
                sample_size=W,
                in_channels=C + Cc, # model expects concatenated x and cond_image along channels
                out_channels=C,
                num_frames=T,
                **cfg.model.kwargs
            ).to(device)
        case "transformer":
            return DiffuserSTDiT(
                input_size=(T, H, W),
                in_channels=C + Cc,
                out_channels=C,
                **cfg.model.kwargs
            ).to(device)
    raise ValueError(f"Unsupported model type: {cfg.model.type}")

# Function to load flow class
def load_flow(cfg, model):
    match (cfg.flow.type).lower():
        case "linear":
            return LinearFlow(
                model=model,
                **cfg.flow.kwargs #TODO Double check it's sent in properly
            )


    raise ValueError(f"Unsupported flow class: {cfg.flow.type}")

def load_vae_processor(cfg, device):
    vae, processor = load_vae_and_processor(
        repo_id=cfg.vae.repo_id,
        subfolder=cfg.vae.subfolder,
        device=device
    )
    return vae, processor


@hydra.main(version_base=None, config_path="configs/flow_train", config_name="flow_train")
def main(cfg: DictConfig):
    # Setup output directories
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    ckpt_dir = (output_dir / "checkpoints")
    sample_dir = (output_dir / "sample_videos")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Datasets and DataLoaders
    train_ds = EchoDataset(cfg, split='train')
    val_ds = EchoDataset(cfg, split='val')
    sample_ds = EchoDataset(cfg, split='sample', n_sample_videos=4)

    train_dl = DataLoader(train_ds, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.dataset.batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
    sample_dl = DataLoader(sample_ds, batch_size=1, collate_fn=make_sampling_collate(n=4))
    dummy_data = train_ds[0]


    # Load model and flow wrapper
    model = load_model(cfg, dummy_data, device)
    model = load_flow(cfg, model)
    model = FlowVideoGenerator(model=model, cfg=cfg, sample_dir=sample_dir, sample_dl=sample_dl)

    # Define callbacks and logger(s)
    callbacks_list = []
    for _,v in OmegaConf.to_container(cfg.ckpt, resolve=True).items():
        if isinstance(v, dict):
            callbacks_list.append(ModelCheckpoint(**v, dirpath=ckpt_dir))

    callbacks_list.append(LearningRateMonitor(logging_interval='step'))

    logger = WandbLogger(
        **cfg.wandb,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    run = logger.experiment
    run.summary['local_output_dir'] = str(output_dir)

    # Instantiate trainers
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks_list,
        **cfg.trainer.kwargs
    )

    # Train the model
    trainer.fit(model, train_dl, val_dl)

if __name__ == '__main__':
    device = select_device()
    main()




#TODO Mask parameter?