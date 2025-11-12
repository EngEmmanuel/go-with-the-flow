import os, shutil, sys
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger


sys.path.append(str(Path(__file__).resolve().parent.parent))

from ef_regression.dataset import CAMUSVideoEF #type: ignore
from ef_regression.model import EFRegressor #type: ignore


if __name__ == "__main__":
    DEFAULT_CONFIG_PATH = "ef_regression/config_reference/camus_112_32.yaml"

    torch.hub.set_dir("/users/spet4299/code/TEE/flow-matching/go-with-the-flow/.cache")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=(DEFAULT_CONFIG_PATH is None),
                        help="Path to config YAML file")
    args = parser.parse_args()

        # 🔹 Pick CLI config if given, otherwise use default
    config_path = args.config if args.config else DEFAULT_CONFIG_PATH
    if not config_path or not os.path.isfile(config_path):
        sys.exit(f"❌ Config file not found: {config_path}")

    # Load config
    config = OmegaConf.load(config_path)
    seed_everything(config.seed)

    # Generate run name
    run_name = "ef_reg_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + os.path.basename(config_path).split(".")[0]

    # Init model
    model = EFRegressor(config)

    # Save config with checkpoints
    os.makedirs(os.path.join(config.checkpoint.path, run_name), exist_ok=True)
    shutil.copy(config_path, os.path.join(config.checkpoint.path, run_name, "config.yaml"))

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.checkpoint.path, run_name, "checkpoints"),
        filename='{epoch}',
        save_top_k=3,
        monitor='val/loss',
        mode='min',
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    logger = WandbLogger(
        name=run_name,
        project=config.wandb.project,
        config=OmegaConf.to_container(config, resolve=True),
        entity=config.wandb.entity,
    )


    train_ds = CAMUSVideoEF(config.dataset, splits=["TRAIN"]) #type: ignore
    val_ds = CAMUSVideoEF(config.dataset, splits=["VAL"])

    train_dl = DataLoader(train_ds, batch_size=config.dataloader.batch_size, shuffle=True) #, num_workers=config.dataloader.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=config.dataloader.batch_size, shuffle=False) #, num_workers=config.dataloader.num_workers, pin_memory=True)


    trainer = Trainer(
        max_epochs=config.trainer.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        accelerator=config.trainer.accelerator, 
        precision=config.trainer.precision,
        strategy=config.trainer.strategy,
        accumulate_grad_batches= config.trainer.accumulate_grad_batches
    )

    trainer.fit(model, train_dl, val_dl)

    print("Done")