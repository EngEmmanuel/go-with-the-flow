
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
        
        # Increment counter
        self._val_counter += 1

        if self._val_counter % self.cfg.sample_every_n_val_steps == 0:
            # Use this batch for sampling
            self.print(f"Sampling at step {self.global_step}")
            batch.pop('x')
            self.sample_from_batch(batch)

    def sample_from_batch(self, batch):
        #TODO Currently only tests reconstruction. Needs to be updated to test generation
        save_path = self.sample_dir / f"sampled_videos_step_{self.global_step}.pt"

        batch_size, *_ = batch['cond_image'].shape
        self.model.eval()
        with torch.no_grad():
            sampled_videos = self.model.sample(**batch, batch_size=batch_size)
            self.save_latent_videos(sampled_videos, save_path)

        self.print(f"Sampled videos saved to: {save_path}")
        self.model.train()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.cfg.trainer.lr)
        return optimizer

    
    def save_latent_videos(self, latents: torch.Tensor, save_path, metadata: dict = {}):
        """
        Save latent videos and optional metadata to a PyTorch .pt file.
        Args:
            latents (torch.Tensor): Tensor of shape [B, C, T, H, W].
            save_path (str): File path to save the tensor.
            metadata (dict, optional): Additional info to save with latents.
        """
        # Create a dictionary to store
        save_dict = {"latents": latents}
        if metadata:
            save_dict["metadata"] = metadata

        torch.save(save_dict, save_path)



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

    train_dl = DataLoader(train_ds, batch_size=cfg.dataset.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.dataset.batch_size)
    data_shape = train_ds.shape # (T, C, H, W)

    # Load model and flow wrapper
    model = load_model(cfg, data_shape, device)
    model = load_flow(cfg, model)
    model = FlowVideoGenerator(model=model, cfg=cfg, sample_dir=sample_dir)

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