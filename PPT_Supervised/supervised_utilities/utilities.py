import numpy as np
import pandas as pd
import glob
from omegaconf import DictConfig

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

def import_supervised_lightmodule(cfg: DictConfig, log):
    """
    Load the data module based on the data name
    :param cfg: Configuration dictionary. [omegaconf.dictconfig.DictConfig]
    :return: Data module. [pytorch_lightning.LightningDataModule]
    """
    if cfg.light_model.light_name == "light_patchtst_permute":
        from supervised_light_model.light_patchtst_permute import LitModel
    elif cfg.light_model.light_name == "light_lstm":
        from supervised_light_model.light_lstm import LitModel
    else:
        raise ValueError(f"Unknown lightning model {cfg.light_model.light_name}. ")

    return LitModel(cfg)


def collect_permute_test_sets_dir(cfg, use_all_test_sets=False):
    """Collect the directories of permute test sets"""
    test_patch_size_list = cfg.permute_test.patch_size_sets
    permute_sets_dir = cfg.permute_test.permute_sets_dir
    data_dirs = []

    all_test_set_dirs = glob.glob(f"src/data/{permute_sets_dir}/" + "random_*")
    # all_test_set_dirs = glob.glob(f"src/data/{permute_sets_dir}/" + "*_010_*.pkl")
    if use_all_test_sets:
        return all_test_set_dirs
    else:
        for test_set_dir in all_test_set_dirs:
            patch_size = int(test_set_dir.split("/")[-1].split("_")[2])
            if isinstance(test_patch_size_list, int):
                test_patch_size_list = [test_patch_size_list]
            if patch_size in test_patch_size_list:
                data_dirs.append(test_set_dir)
        return data_dirs
    

def collect_predict_epoch_end_supervised(cfg, trainer, outputs, data_name):
    test_loss = torch.stack([x["test_loss"] for x in outputs]).cpu().mean()
    y_pred = torch.cat([torch.argmax(x["y_pred"], 1) for x in outputs]).cpu()
    y_pred_proba = torch.nn.Softmax(dim=1)(torch.cat([x["y_pred"] for x in outputs])).cpu()
    y_pred_proba_df = pd.DataFrame(y_pred_proba.numpy(), columns=[f"pred_{i}" for i in range(cfg.data.num_class)])
    y_true = torch.cat([x["y_true"] for x in outputs]).cpu()

    test_accuracy = accuracy_score(y_true, y_pred)
    if cfg.data.num_class == 2:
        test_auroc = roc_auc_score(y_true, y_pred_proba[:, 1])
        test_auprc = average_precision_score(y_true, y_pred_proba[:, 1])
    else:
        test_auroc = roc_auc_score(y_true, y_pred_proba, multi_class="ovo")
        test_auprc = average_precision_score(y_true, y_pred_proba)
    print(f"Test Accuracy: {test_accuracy:.3f}, Test AUROC: {test_auroc:.3f}, Test AUPRC: {test_auprc:.3f}")
    # Save the test results in the output directory
    test_label = pd.read_csv(f"{cfg.save_output_path}/test_label.csv")
    test_label = pd.concat([test_label, y_pred_proba_df], axis=1)
    test_label["y_pred"] = y_pred.numpy()
    test_label["y_true"] = y_true.numpy()
    test_label["test_loss"] = test_loss.item()
    test_label["test_accuracy"] = test_accuracy
    test_label["test_auroc"] = test_auroc
    test_label["test_auprc"] = test_auprc
    test_label["stop_epoch"] = trainer.early_stopping_callback.stopped_epoch
    test_label["cv_num"] = cfg.data.validation_cv_num
    test_label["exp_num"] = cfg.exp_num
    test_label["model_name"] = cfg.model.model_name
    test_label["train_data_name"] = cfg.data.data_name

    # parse data_name
    parsed_data_name = data_name.split("_")
    # test_label["permute_strategy"] = parsed_data_name[0]
    test_label["permute_patch_len"] = int(parsed_data_name[2])
    # test_label["permute_frequency"] = int(parsed_data_name[4])
    test_label["permute_dataset_name"] = data_name

    # parse shuffler and lambda value
    # if cfg.light_model.light_name == "light_patchtst_permute":
    #     test_label["use_consistency_loss"] = cfg.light_model.ssl_loss.use_consistency_loss
    #     test_label["use_margin_loss"] = cfg.light_model.ssl_loss.use_margin_loss

    return test_label
