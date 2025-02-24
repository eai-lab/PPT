import numpy as np
import pandas as pd
import glob
import omegaconf
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import NeptuneLogger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

import torch


def seed_everything(seed: int = 42):
    """
    Seed everything for reproducibility.
    :param seed: Random seed. [int]
    :return:
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def import_datamodule(cfg: DictConfig, log):
    """
    Load the data module based on the data name
    :param cfg: Configuration dictionary. [omegaconf.dictconfig.DictConfig]
    :return: Data module. [pytorch_lightning.LightningDataModule]
    """
    if cfg.data.data_name in ["gilon_activity"]:
        log.info(f"Setting up GILON Dataset with CV number {cfg.data.validation_cv_num}")
        from src.dataset.gilon_dataset import DataModule
    elif cfg.data.data_name in ["ptb"]:
        log.info(f"Setting up PTB Dataset with pre-defined split")
        from src.dataset.ptb_dataset import DataModule
    elif cfg.data.data_name in ["ms_har"]:
        log.info(f"Setting up MS-HAR Dataset with pre-defined split")
        from src.dataset.ms_har_dataset import DataModule
    else:
        log.info(f"Setting up UEA style dataset with pre-defined split")
        from src.dataset.uea_dataset import DataModule
         
    return DataModule(cfg)


def setup_neptune_logger(cfg: DictConfig, tags: list = None):
    """
    Nettune AI loger configuration. Needs API key.
    :param cfg: Configuration dictionary. [omegaconf.dictconfig.DictConfig]
    :param tags: List of tags to log a particular run. [list]
    :return:
    """

    # setup logger
    neptune_logger = NeptuneLogger(
        api_key=cfg.logger.api_key,
        project=cfg.logger.project_name,
        mode=cfg.logger.mode,
    )

    neptune_logger.experiment["parameters/model"] = cfg.model.model_name

    return neptune_logger


def print_options(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ""
    message += "----------------- Options ---------------\n"
    for k, v in sorted(vars(opt).items()):
        comment = ""
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    print(message)

def fix_config(cfg: DictConfig):
    num_gpu = torch.cuda.device_count()
    if cfg.gpu_id >= 0:
        print(f"{bcolors.HEADER}{bcolors.WARNING} The gpu_id [{cfg.gpu_id}] exceeds the total GPUs [{num_gpu}]{bcolors.ENDC}{bcolors.ENDC}")
        print(f"{bcolors.HEADER}{bcolors.WARNING} Replace the gpu_id [{cfg.gpu_id}] to [{cfg.gpu_id % num_gpu}]{bcolors.ENDC}{bcolors.ENDC}")
        
        setattr(cfg, 'gpu_id', cfg.gpu_id % num_gpu)

def bcolor_prints(cfg, show_full_cfg=False):
    fix_config(cfg)
    print(f"{bcolors.HEADER}=====> {cfg.task.task_name.upper()} <===== setting {bcolors.ENDC}")
    print(f"{bcolors.HEADER}=====> running {cfg.model.model_name.upper()} model {bcolors.ENDC}")
    print(f"{bcolors.HEADER}=====> running {cfg.light_model.light_name.upper()} light model {bcolors.ENDC}")
    print(f"{bcolors.HEADER}=====> with {cfg.data.data_name} data {bcolors.ENDC}")
    if cfg.model.model_name in ["patchtst", "pits"]:
        print(f"{bcolors.HEADER}=====> with patch size {cfg.model.patch_len} {bcolors.ENDC}")
    if show_full_cfg:
        print(f"\n\n{bcolors.HEADER}<========== Full Configurations ==========>\n{OmegaConf.to_yaml(cfg)} \n<=========================================>{bcolors.ENDC}\n\n")

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"



