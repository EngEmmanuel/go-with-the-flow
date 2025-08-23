from venv import logger
import torch
import random
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, ModuleList
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Literal, Callable
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb # type: ignore
import sklearn.metrics
from torchdiffeq import odeint
import einx
from pathlib import Path
from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange

from src.models import UNetSTIC, DiffuserSTDiT
from dataset.testdataset import FlowTestDataset
def identity(t):
    return t

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))



def select_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def mask_random_frames(video: torch.Tensor, t:int):
    '''
    Masks 'n_missing_frames' in the range [0, t) of the input video tensor.
    '''
    T, C, H, W = video.shape
    assert t <= T, "t must be less than or equal to T"

    n_missing_frames = torch.randint(low=1, high=t, size=(1,)).item() #Upper bound not inclusive therefore t-1 is max
    mask_indices = random.sample(range(t), n_missing_frames)

    mask = torch.ones_like(video)
    mask[mask_indices, ...] = 0

    masked_video = video * mask
    return masked_video
### ModelClass

#out = model(x_bcthw_for_concat, timesteps, encoder_hidden_states=encoder_hidden_states, cond_image=cond_image)


### FlowClass
class LinearFlow(Module):
    def __init__(
            self,
            model,
            data_shape: tuple[int, ...] | None = None,
            clip_values: tuple[float, float] = (-1., 1.),
            **kwargs
    ):
        super().__init__()
        self.model = model

        self.data_shape = data_shape
        self.noise_schedule = lambda x: x
        self.clip_values = clip_values

        self.loss_fn = nn.MSELoss()
        # objective - either flow or noise
        # self.predict = predict

    @property
    def device(self):
        return next(self.model.parameters()).device

    def sample_times(self, batch):
        pass

    @torch.no_grad()
    def sample(
        self, 
        encoder_hidden_states: torch.Tensor, 
        batch_size=1, 
        steps=16, 
        noise=None, 
        data_shape: tuple[int, ...] | None = None,
        cond_image=None, 
        mask=None,
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        use_ema: bool = False,
        **model_kwargs
    ):

        data_shape = default(data_shape, self.data_shape)
        
        def ode_fn(t, x):
             #TODO Clip flow values
            output = self.predict_flow(model, x, times=t, encoder_hidden_states=encoder_hidden_states, cond_image=cond_image, mask=mask, **model_kwargs)
            return output

        # Start with random gaussian noise - y0
        noise = default(noise, torch.randn(batch_size, *data_shape, device=self.device))

        # time steps
        time_steps = torch.linspace(0., 1., steps, device=self.device)

        # ode
        trajectory = odeint(ode_fn, noise, time_steps, **odeint_kwargs)

        sampled_data = trajectory[-1]  # Get the last state as the sampled data
        
        return sampled_data
    
    # Keep model arg in case of ema
    def predict_flow(self,
                    model:Module, 
                    noised, 
                    *, 
                    times, 
                    encoder_hidden_states=None, 
                    cond_image=None, 
                    mask=None, 
                    eps=1e-10, 
                    **model_kwargs
                ):

        batch = noised.shape[0]
        
        # Prepare time conditioning for model
        times = rearrange(times, '... -> (...)') # Flattens times

        if times.numel() == 1:
            times = repeat(times, '1 -> b', b = batch)

        # Unet and STDiT forward(x, timestep, encoder_hidden_states=None, cond_image=None, mask=None, return_dict=True)

        output = self.model(x=noised, timestep=times, encoder_hidden_states=encoder_hidden_states, cond_image=cond_image, mask=mask, **model_kwargs) # predicted flow / velocity field
        if hasattr(output, 'sample'):
            return output.sample
        return output


    def forward(
            self, 
            x, 
            encoder_hidden_states: torch.Tensor, 
            noise: Tensor | None = None,
            cond_image=None, 
            mask=None, 
            **model_kwargs
        ):

        batch, *data_shape = x.shape
        self.data_shape = default(self.data_shape, data_shape)

        # x0 - gaussian noise, x1 - data
        noise = default(noise, torch.randn_like(x))

        times = torch.rand(batch, device = self.device)
        padded_times = append_dims(times, x.ndim - 1)

        def get_noised_and_flows(model, t):

            # maybe noise schedule
            t = self.noise_schedule(t)
            noised = x * t + noise * (1 - t)

            flow = x - noise
            
            pred_flow = self.predict_flow(model, noised, times=t, encoder_hidden_states=encoder_hidden_states, cond_image=cond_image, **model_kwargs)

            pred_x = noised + pred_flow * (1 - t)

            return flow, pred_flow, pred_x

        # getting flow and pred flow for main model
        flow, pred_flow, pred_x = get_noised_and_flows(self.model, padded_times)

        main_loss = self.loss_fn(pred_flow, flow) #, pred_data = pred_x, times = times, data = x)

        return main_loss
#TODO Where is best to include the VAE?

# Model conditioned on time is assumed True

### Trainer
class FlowVideoGenerator(LightningModule):
    def __init__(
            self,
            flow,
            cfg,
            **kwargs
    ):
        self.model = flow
        self.cfg = cfg


    def training_step(self, batch, batch_idx):
        loss = self.model(**batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch)
        self.log('val_loss', loss)
        return loss

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Increment counter
        self._val_batch_counter += 1

        if self._val_batch_counter % self.sample_every_n_val_steps == 0:
            # Use this batch for sampling
            batch.pop('x')
            self.sample_from_batch(**batch)

    def sample_from_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            sampled_videos = self.model.sample(**batch)
            self.save_latent_videos(sampled_videos, f"sampled_videos_val_step_{self._val_batch_counter}.pt")
        self.model.train()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.cfg.trainer.lr)
        return optimizer

    
    def save_latent_videos(self, latents: torch.Tensor, save_path: str, metadata: dict = {}):
        """
        Save latent videos and optional metadata to a PyTorch .pt file.
        Args:
            latents (torch.Tensor): Tensor of shape [B, C, T, H, W].
            save_path (str): File path to save the tensor.
            metadata (dict, optional): Additional info to save with latents.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=False, exist_ok=False)

        # Create a dictionary to store
        save_dict = {"latents": latents}
        if metadata:
            save_dict["metadata"] = metadata

        torch.save(save_dict, save_path)



    #TODO Add some sort of sample step?
#TODO Complete me. Simple is best. Just see if things run. Use dummy tensors

if __name__ == '__main__':
    device = select_device()
    # Mock data
    B, T, C, H, W = 2, 32, 4, 28, 28
    cross_attention_dim = 2
    ds = FlowTestDataset(B=B, T=T, C=C, H=H, W=W, device=device, cross_attention_dim=cross_attention_dim)
    dl = DataLoader(ds, batch_size=B)
    model = UNetSTIC(
        sample_size=H, #H,W of latent frame
        in_channels=C*2, # model expects concatenated x and cond_image along channels; use 2*C for test
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

    model = LinearFlow(model=model)

    # Perform a simple forward through the wrapper to get the model output
    batch = next(iter(dl))
    out = model(**batch)
    print('out shape:', out.shape)

    batch.pop('x')
    latent_sample = model.sample(**batch, batch_size=B)

    #.
    #
    hydra_run_path = Path('.')
    ckpt_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=hydra_run_path / "checkpoints",
        filename="flow-video-generator-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )

    lr_callback = LearningRateMonitor(logging_interval='step')

    logger = WandbLogger(
        project="go-with-the-flow",
        
    )
    trainer = Trainer(
        max_epochs=2,
        logger=logger,
        accelerator="auto",
        precision=32,
    )