import os
import sys
import pickle
import datetime
import logging

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from src.utils import (
    print_options,
    setup_neptune_logger,
    import_datamodule,
    seed_everything,
    bcolor_prints
)

from ssl_utilities.utils_finetune import (
    import_ssl_finetune_lightmodule,
    collect_predict_epoch_end_supervised,
    collect_permute_test_sets_dir,
)


torch.set_printoptions(sci_mode=False)
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="ssl_conf", config_name="fullfinetune_cfg")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    logger = setup_neptune_logger(cfg)
    print_options(cfg)
    # Save model checkpoint on this path
    checkpoint_path = os.path.join("checkpoints/", str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    # set up pytorch lightning model
    model = import_ssl_finetune_lightmodule(cfg, log)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor=cfg.light_model.callbacks.monitor,
        save_top_k=1,
        filename=cfg.model.model_name + "_{epoch:02d}",
        mode=cfg.light_model.callbacks.monitor_mode,
    )
    early_stop_callback = EarlyStopping(
        monitor=cfg.light_model.callbacks.monitor,
        patience=cfg.light_model.callbacks.patience,
        verbose=True,
        mode=cfg.light_model.callbacks.monitor_mode,
        min_delta=cfg.light_model.callbacks.min_delta,
    )
    model_summary_callback = ModelSummary(max_depth=1)

    dm = import_datamodule(cfg, log)
    dm.setup(stage="fit")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[cfg.gpu_id],
        benchmark=cfg.benchmark,
        deterministic=cfg.deterministic,
        check_val_every_n_epoch=cfg.light_model.callbacks.check_val_every_n_epoch,
        max_epochs=cfg.light_model.callbacks.max_epochs,
        max_steps=cfg.light_model.callbacks.max_steps,
        callbacks=[checkpoint_callback, early_stop_callback, model_summary_callback],
        logger=logger,
        fast_dev_run=cfg.light_model.dataset.fast_dev_run,
        limit_train_batches=cfg.light_model.dataset.limit_train_batches,
        limit_val_batches=cfg.light_model.dataset.limit_val_batches,
    )
    log.info("Start training")
    trainer.fit(model, dm)
    best_epoch = trainer.early_stopping_callback.stopped_epoch
    log.info(f"Best epoch: {best_epoch}")


    log.info("Start testing")
    topk_checkpoint_paths = os.listdir(checkpoint_path)
    dm.setup("test")
    trainer.test(model, dm.test_dataloader(), ckpt_path=checkpoint_path + "/" + topk_checkpoint_paths[0])

    log.info("Finished testing")



if __name__ == "__main__":
    # Set hyrda configuration not to change the directory by default. This is needed for the output directory to work.
    sys.argv.append("hydra.job.chdir=False")
    main()
