from my_src.model import UNet3D
from my_src.flows import LinearFlow, MeanFlow
from vae.util import load_vae_and_processor




def load_model(cfg, dummy_data, device):
    if isinstance(dummy_data, dict):
        C, T, H, W = dummy_data['x'].shape
        Cc, _, _, _ = dummy_data['cond_image'].shape

    print(f'input shape: {(C+Cc, T, H, W)}, with {Cc} cond channels')

    match (cfg.model.type).lower():
        case "unet":
            return UNet3D(
                sample_size=W,
                in_channels=C + Cc, # model expects concatenated x and cond_image along channels
                out_channels=C,
                num_frames=T,
                **cfg.model.kwargs
            ).to(device)
    raise ValueError(f"Unsupported model type: {cfg.model.type}")

# Function to load flow class
def load_flow(cfg, model):
    match (cfg.flow.type).lower():
        case "linear":
            return LinearFlow(
                model=model,
                **cfg.flow.get('kwargs', {})
            )

        case "mean":
            return MeanFlow(
                model=model,
                **cfg.flow.get('kwargs', {})
            )

    raise ValueError(f"Unsupported flow class: {cfg.flow.type}")

def load_vae_processor(cfg, device):
    vae, processor = load_vae_and_processor(
        repo_id=cfg.vae.repo_id,
        subfolder=cfg.vae.subfolder,
        device=device
    )
    return vae, processor



class SamplerConductor():
    def __init__(self, run_cfg):
        self.max_epochs = run_cfg.trainer.kwargs.max_epochs
        self.sample_every_n_epochs = run_cfg.sample.get('every_n_epochs', self.max_epochs)
        self.last_sample_epoch = 0

        self.roi_freq_multiplier = 2
        self.scheduler = run_cfg.trainer.get('lr_scheduler', None)

    def is_sample_step(
            self,
            epoch, 
            last_sample_epoch,
            last_step
        ):

        # Always sample at the last step
        if last_step:
            return True
        
        epoch_freq = self.sample_every_n_epochs
        progress = epoch / self.max_epochs
        if self.scheduler == 'cosineannealing':
            roi = (0.4, 0.55) # roi for mean flow
            if progress >= roi[0] and progress <= roi[1]:
                epoch_freq = self.sample_every_n_epochs // self.roi_freq_multiplier
                print(f'temp In ROI {roi}, sampling every {epoch_freq} epochs')

            return (
                epoch % epoch_freq == 0
                and epoch != last_sample_epoch
            )
        
        elif self.scheduler is None: # replace with linear decay roi
            return (
                epoch % self.sample_every_n_epochs == 0
                and epoch != last_sample_epoch
		    )
        
        else:
            return (
                epoch % self.sample_every_n_epochs == 0
                and epoch != last_sample_epoch
		    )


    
