import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from rectified_flow_pytorch import Trainer, ReflowTrainer, Reflow
from rectified_flow_pytorch import ImageDataset, Unet, Trainer
from dataset import EchoDataset
from rectified_flow_pytorch import MeanFlow, NanoFlow, RectifiedFlow
from pathlib import Path

# Function to load dataset
def load_dataset(cfg):
    if cfg.dataset.name == "TestDataset":
        from dataset.testdataset import TestDataset
        DatasetClass = TestDataset
    elif cfg.dataset.name == "EchoDataset":
        DatasetClass = EchoDataset  
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")
    
    return DatasetClass(cfg.dataset)

# Function to load model
def load_model(cfg):
        return Unet(**cfg.model)


# Function to load flow class
def load_flow(cfg, model):
    if cfg.flow == "MeanFlow":
        return MeanFlow(
            model,
            normalize_data_fn=lambda t: t * 2. - 1.,
            unnormalize_data_fn=lambda t: (t + 1.) / 2.
        )
    elif cfg.flow == "NanoFlow":
        return NanoFlow(
            model,
            normalize_data_fn=lambda t: t * 2. - 1.,
            unnormalize_data_fn=lambda t: (t + 1.) / 2.
        )
    elif cfg.flow == "RectifiedFlow":
        return RectifiedFlow(
            model,
            data_normalize_fn = lambda t: t * 2. - 1.,
            data_unnormalize_fn = lambda t: (t + 1.) / 2.,
        )
    else:
        raise ValueError(f"Unsupported flow class: {cfg.flow.name}")

@hydra.main(version_base=None, config_path="configs/test", config_name="test")
def main(cfg: DictConfig):
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    trainer_kwargs = {
        "checkpoints_folder": output_dir / "checkpoints",
        "results_folder": output_dir / "results"
    }
    

    # Load dataset
    dataset = load_dataset(cfg)

    # Load model
    model = load_model(cfg)

    # Load flow class
    flow = load_flow(cfg, model)

    # Initialize trainer
    trainer = Trainer(
        flow,
        dataset=dataset,
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        **trainer_kwargs
    )

    # Start training
    trainer()

if __name__ == "__main__":
    main()