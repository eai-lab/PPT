import os
import sys
import pandas as pd
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
from ssl_utilities.utils_pretrain import (
    import_ssl_lightmodule,
    collect_ssl_predict_epoch_end,
    collect_permute_test_sets_dir,
    eval_classification,
    eval_classification_with_permuted_test,
    construct_output_results,
    save_attention_files,
    save_patch_embeds
)


torch.set_printoptions(sci_mode=False)
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="ssl_conf", config_name="pretrain_cfg")
def main(cfg: DictConfig) -> None:
    bcolor_prints(cfg)
    seed_everything(cfg.seed)
    logger = setup_neptune_logger(cfg)
    print_options(cfg)
    # Save model checkpoint on this path
    checkpoint_path = os.path.join("checkpoints/", str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    # set up pytorch lightning model
    model = import_ssl_lightmodule(cfg, log)
    
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
        log_every_n_steps=50,
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

    model.save_model_weight()


    log.info("Start Linear Probing....")
    topk_checkpoint_paths = os.listdir(checkpoint_path)

    # # for linear probing
    # 1. collect train embeddings and fit clf
    # 2. collect test embeddings and predict
    if cfg.light_model.light_name in ("light_patchtst_pretrain_permute", "light_patchtst_pretrain_mask", "light_patchtst_pretrain_cl",
                                      "light_pits_pretrain_permute", "light_pits_pretrain_mask"):
        # 1. collect train embeddings and fit clf
        log.info("Start collecting train embeddings and fit clf")
        predict_train_outputs = trainer.predict(
            model,
            dm.train_dataloader(batch_size=1024),
            ckpt_path=checkpoint_path + "/" + topk_checkpoint_paths[0],
            return_predictions=True,
        )
        predict_test_outputs = trainer.predict(
            model,
            dm.test_dataloader(batch_size=1024),
            ckpt_path=checkpoint_path + "/" + topk_checkpoint_paths[0],
            return_predictions=True,
        )
        if cfg.save_attention_file:
            save_attention_files(cfg, predict_test_outputs)
        
        train_outs = collect_ssl_predict_epoch_end(predict_train_outputs)
        test_outs = collect_ssl_predict_epoch_end(predict_test_outputs)

        log.info("Start evaluating with linear classifiers")
        len_stages = len(train_outs["patch_embed_out"]) # layers of model. e.g. 3 for patchtst
        embed_dicts = []
        for i in range(len_stages):
            _, embed_dict = eval_classification(
                cfg,
                train_outs["patch_embed_out"][i],
                train_outs["y_true"],
                test_outs["patch_embed_out"][i],
                test_outs["y_true"],
                eval_protocol="linear",
            )
            embed_dicts.append(embed_dict)
            log.info(f"Patch Embed {i} SSL accuracy: {embed_dict['acc']}")

        # test on averaged embeddings as well
        avg_clf, embed_dict_avg = eval_classification(
            cfg,
            train_outs["all_average_embed_out"],
            train_outs["y_true"],
            test_outs["all_average_embed_out"],
            test_outs["y_true"],
            eval_protocol="linear",
        )
        log.info(f"All Average Embed SSL accuracy: {embed_dict_avg['acc']}")
        embed_dicts.append(embed_dict_avg)

        _, embed_dict_max = eval_classification(
            cfg,
            train_outs["all_maxed_embed_out"],
            train_outs["y_true"],
            test_outs["all_maxed_embed_out"],
            test_outs["y_true"],
            eval_protocol="linear",
        )
        log.info(f"All Maxed Embed SSL accuracy: {embed_dict_max['acc']}")
        embed_dicts.append(embed_dict_max)

        output_ssl_df = construct_output_results(cfg, embed_dicts)

        # save output results

        output_ssl_df.to_csv(f"{cfg.save_output_path}/cv{cfg.data.validation_cv_num}_pretrain.csv", index=False)
        print(f"Saved output results to {cfg.save_output_path}/cv{cfg.data.validation_cv_num}_pretrain.csv")

        if cfg.permute_test.perform_test_on_permute_sets:
            print("Performing Permutation Test in Self-Supervised Learning with avg_clf")
            data_dirs = collect_permute_test_sets_dir(cfg, use_all_test_sets=cfg.permute_test.use_all_test_sets)
            if len(data_dirs) == 0:
                raise ValueError("No data found for permutation test. Please check the data directory.")
            # make directory
            if not os.path.exists(f"{cfg.save_output_path}/permute_test"):
                os.makedirs(f"{cfg.save_output_path}/permute_test")

            permute_dict_results_list = []
            for data_dir in data_dirs:
                data_name = data_dir.split("/")[-1].split(".")[0]  # collect without format name (e.g. .pkl)
                predict_outputs = trainer.predict(
                    model,
                    dm.predict_dataloader(permute_predict_name=data_name),
                    ckpt_path=checkpoint_path + "/" + topk_checkpoint_paths[0],
                    return_predictions=True,
                )
                predict_outs = collect_ssl_predict_epoch_end(predict_outputs)
                permute_result_dict = eval_classification_with_permuted_test(cfg, 
                                                       predict_outs['all_average_embed_out'],
                                                       predict_outs['y_true'],
                                                       avg_clf)
                permute_result_dict["data_name"] = data_name
                permute_dict_results_list.append(permute_result_dict)
                log.info(f"Permutation Test on {data_name} accuracy: {permute_result_dict['acc']}")
            # concat all the results
            permute_result_df = pd.DataFrame(permute_dict_results_list)
            permute_result_df.to_csv(f"{cfg.save_output_path}/permute_test/cv{cfg.data.validation_cv_num}_pretrain.csv", index=False)
            


        torch.cuda.empty_cache()  # Clear cache
        gc.collect()  # Collect garbage


if __name__ == "__main__":
    # Set hyrda configuration not to change the directory by default. This is needed for the output directory to work.
    sys.argv.append("hydra.job.chdir=False")
    main()
