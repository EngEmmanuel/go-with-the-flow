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