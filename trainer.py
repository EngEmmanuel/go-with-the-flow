
import torch
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
import wandb # type: ignore
import sklearn.metrics


from src.models import UNetSTIC, DiffuserSTDiT
from dataset.testdataset import FlowTestDataset
from dataset.echodataset import EchoDataset
from src.flows import LinearFlow
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

    def training_step(self, batch, batch_idx):
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

        if self._val_counter % self.cfg.sample_every_n_val_steps == 0:
            # Use this batch for sampling
            self.print(f"Sampling at step {self.global_step}")
            #batch.pop('x')
            #self.sample_from_batch(batch)

            self.sample()

        # Increment counter
        self._val_counter += 1


    def sample(self, n_videos_per_sample=2):
        self.model.eval()
        sample_results = {}
        for n in range(n_videos_per_sample):
            reference_batch, repeated_batch = next(self.sample_dl)
            batch_size, *_ = repeated_batch['cond_image'].shape

            with torch.no_grad():
                sampled_videos = self.model.sample(**repeated_batch, batch_size=batch_size)
            ef_values = reference_batch['ef_values']
            sample_results[n] = {
                "video_name": reference_batch['video_name'],
                "cond_image": reference_batch['cond_image'],
                'reconstructed': (sampled_videos[0,...], round(ef_values[0].item(), 3)),
                'generated': (sampled_videos[1:,...], [round(x.item(), 3) for x in ef_values[1:]])
            }

            # TODO Check if data is unnormalized somewhere here when sampling
        torch.save(
            sample_results, 
            self.sample_dir / f"sampled_videos_step_{self.global_step}.pt"
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.cfg.trainer.lr)
        return optimizer




    #TODO Add some sort of sample step?
#TODO Complete me. Simple is best. Just see if things run. Use dummy tensors


def load_model(cfg, data_shape, device):
    T, C, H, W = data_shape

    match (cfg.model.type).lower():
        case "unet":
            return UNetSTIC(
                sample_size=W, # (T, C, H, W)
                in_channels=C + C, # model expects concatenated x and cond_image along channels
                out_channels=C,
                num_frames=T,
                **cfg.model.kwargs
            ).to(device)
        case "transformer":
            return DiffuserSTDiT(
                input_size=(T, H, W),
                in_channels=C + C,
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


#TODO LOAD VAE PROCESSOR. SEE HOW THINGS NEED TO BE SCALED FOR TRAIN AND SAMPLING
#TODO REVISIT DECODE_LATENT_IMAGE
#TODO START TRAINING SOMETHING
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

    train_dl = DataLoader(train_ds, batch_size=cfg.dataset.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.dataset.batch_size)
    sample_dl = DataLoader(sample_ds, batch_size=1, collate_fn=make_sampling_collate(n=4))
    data_shape = train_ds.shape # (T, C, H, W)

    # Load model and flow wrapper
    model = load_model(cfg, data_shape, device)
    model = load_flow(cfg, model)
    model = FlowVideoGenerator(model=model, cfg=cfg, sample_dir=sample_dir, sample_dl=sample_dl)

    # Define callbacks and logger(s)
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,

        filename='ckpt-{epoch}-{step}',
        **cfg.ckpt
    )
    lr_callback = LearningRateMonitor(logging_interval='step')

    logger = WandbLogger(
        **cfg.wandb,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Instantiate trainer
    trainer = Trainer(
        logger=logger,
        callbacks=[ckpt_callback, lr_callback],
        **cfg.trainer.kwargs
    )

    # Train the model
    trainer.fit(model, train_dl, val_dl)

if __name__ == '__main__':
    device = select_device()

    main()





    if False:
        arch = 'unet'
        # Mock data
        B, T, C, H, W = 2, 32, 4, 28, 28
        cross_attention_dim = 2
        train_ds = FlowTestDataset(B=B, T=T, C=C, H=H, W=W, device=device, cross_attention_dim=cross_attention_dim)
        val_ds = FlowTestDataset(B=B, T=T, C=C, H=H, W=W, device=device, cross_attention_dim=cross_attention_dim)
        train_dl = DataLoader(train_ds, batch_size=B)
        val_dl = DataLoader(val_ds, batch_size=B)

        if arch == "unet":
            model = UNetSTIC(
                sample_size=H, #H,W of latent frame
                in_channels=C+C+2, # model expects concatenated x and cond_image along channels; use 2*C for test
                out_channels=C, 
                num_frames=T,
                down_block_types = (
                    "CrossAttnDownBlockSpatioTemporal",
                    "DownBlockSpatioTemporal"
                ),
                up_block_types = (
                    "UpBlockSpatioTemporal",
                    "CrossAttnUpBlockSpatioTemporal"
                ),
                block_out_channels=[32, 64],
                num_attention_heads=(2, 4),
                cross_attention_dim=cross_attention_dim
            ).to(device)
        else:
            model = DiffuserSTDiT(
                input_size=(T, H, W),
                in_channels=C+C+2,
                out_channels=C,
                patch_size=(1, 2, 2),
                num_heads=8,
                caption_channels=cross_attention_dim,
                model_max_length=1
            ).to(device)
        model = LinearFlow(model=model)

        # Perform a simple forward through the wrapper to get the model output
        batch = next(iter(train_dl))
        out = model(**batch)
        print('out shape:', out.shape)

        batch.pop('x')
        latent_sample = model.sample(**batch, batch_size=B)











#TODO Mask parameter?