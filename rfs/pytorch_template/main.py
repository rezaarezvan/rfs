import os
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from data_loader.data_loader import get_dataloaders
from model.model import get_model
from trainer.trainer import LitModule

def flatten_config(cfg):
    flat_config = {}
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            for sub_key, sub_value in flatten_config(value).items():
                flat_config[f"{key}.{sub_key}"] = sub_value
        else:
            flat_config[key] = value
    return flat_config

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    print(OmegaConf.to_yaml(cfg))

    flat_config = flatten_config(cfg)

    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    model = get_model(cfg)

    lit_model = LitModule(model, cfg)

    logger = WandbLogger(project=cfg.project_name, name=cfg.run_name, config=flat_config)

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoint_dir,
            filename="{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
            monitor="val_loss"
        ),
        EarlyStopping(monitor="val_loss", patience=cfg.training.patience),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu" if cfg.training.use_gpu else "cpu",
        devices=1 if cfg.training.use_gpu else None,
        logger=logger,
        callbacks=callbacks,
        precision=16 if cfg.training.use_amp else 32,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(lit_model, train_loader, val_loader)
    trainer.test(lit_model, test_loader)

if __name__ == "__main__":
    main()
