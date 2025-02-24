import os
import sys
import pickle
import datetime
import logging
import gc

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
from supervised_utilities.utilities import (
    import_supervised_lightmodule,
    collect_permute_test_sets_dir,
    collect_predict_epoch_end_supervised
)


torch.set_printoptions(sci_mode=False)
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="supervised_conf", config_name="supervised_config")
def main(cfg: DictConfig) -> None:
    bcolor_prints(cfg, show_full_cfg=True)
    seed_everything(cfg.seed)
    logger = setup_neptune_logger(cfg)
    # print_options(cfg)
    # Save model checkpoint on this path
    checkpoint_path = os.path.join("checkpoints/", str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + f"_{os.getpid()}")

    # set up pytorch lightning model
    model = import_supervised_lightmodule(cfg, log)

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
        deterministic=cfg.deterministic,
        check_val_every_n_epoch=cfg.light_model.callbacks.check_val_every_n_epoch,
        max_epochs=cfg.light_model.callbacks.max_epochs,
        max_steps=cfg.light_model.callbacks.max_steps,
        callbacks=[checkpoint_callback, early_stop_callback, model_summary_callback],
        logger=logger,
        fast_dev_run=cfg.task.fast_dev_run,
        limit_train_batches=cfg.task.limit_train_batches,
        limit_val_batches=cfg.task.limit_val_batches,
    )
    log.info("Start training")
    trainer.fit(model, dm)

    log.info("Start testing")
    topk_checkpoint_paths = os.listdir(checkpoint_path)
    dm.setup("test")
    trainer.test(model, dm.test_dataloader(), ckpt_path=checkpoint_path + "/" + topk_checkpoint_paths[0])

    # Perform Permutation Test
    if cfg.permute_test.perform_test_on_permute_sets:
        data_dirs = collect_permute_test_sets_dir(cfg, use_all_test_sets=cfg.permute_test.use_all_test_sets)
        if len(data_dirs) == 0:
            raise ValueError("No data found for permutation test. Please check the data directory.")
        # make directory
        if not os.path.exists(f"{cfg.save_output_path}/permute_test"):
            os.makedirs(f"{cfg.save_output_path}/permute_test")

        for data_dir in data_dirs:
            data_name = data_dir.split("/")[-1].split(".")[0]  # collect without format name (e.g. .pkl)
            predict_outputs = trainer.predict(
                model,
                dm.predict_dataloader(permute_predict_name=data_name),
                ckpt_path=checkpoint_path + "/" + topk_checkpoint_paths[0],
                return_predictions=True,
            )
            test_output_df = collect_predict_epoch_end_supervised(cfg, trainer, predict_outputs, data_name)

            # save the test results in the output directory
            test_output_df.to_pickle(
                f"{cfg.save_output_path}/permute_test/{data_name}_cv{cfg.data.validation_cv_num}.pkl"
            )
            log.info(
                f"Saved the test results in {cfg.save_output_path}/permute_test/{data_name}_cv{cfg.data.validation_cv_num}.pkl"
            )

    torch.cuda.empty_cache()  # Clear cache
    gc.collect()  # Collect garbage
if __name__ == "__main__":
    # Set hyrda configuration not to change the directory by default. This is needed for the output directory to work.
    sys.argv.append("hydra.job.chdir=False")
    main()
