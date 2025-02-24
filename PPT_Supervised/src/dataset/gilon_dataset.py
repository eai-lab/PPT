import os
import pandas as pd
import numpy as np
import pickle
from typing import Optional
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L

from src.utils import bcolors


class DataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str], permute_predict_name=None):
        if stage == "fit":
            self.train = GilonDataset("train", self.cfg)
            self.val = GilonDataset("val", self.cfg)
        elif stage == "test":
            self.test = GilonDataset("test", self.cfg)
            # self.test_sampler = ConstantRandomSampler(self.train)
        elif stage == "predict":
            pass
        else:
            raise ValueError(f"Unknown stage {stage}")

    def train_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.cfg.light_model.dataset.batch_size
        else:
            batch_size = batch_size # for linear probing
        return DataLoader(
            self.train,
            batch_size=batch_size,
            num_workers=self.cfg.light_model.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.cfg.light_model.dataset.batch_size,
            num_workers=self.cfg.light_model.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.cfg.light_model.dataset.batch_size
        else:
            batch_size = batch_size # for linear probing
        return DataLoader(
            GilonDataset("test", self.cfg),
            batch_size=batch_size,
            num_workers=self.cfg.light_model.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self, permute_predict_name=None, noise_predict_name=None):
        return DataLoader(
            GilonDataset("predict", self.cfg, permute_predict_name=permute_predict_name, noise_predict_name=noise_predict_name),
            batch_size=512,
            num_workers=self.cfg.light_model.dataset.num_workers,
            shuffle=False,
            drop_last=False,
        )


class GilonDataset(Dataset):
    """Load Gilon sensor data and meta data using global_id value. index is mapped to global id through label_df"""

    def __init__(self, mode, cfg, permute_predict_name=None, noise_predict_name=None):
        print(bcolors.OKBLUE + bcolors.BOLD + f"{mode} Mode" + bcolors.ENDC + bcolors.ENDC)
        self.mode = mode
        self.cfg = cfg
        self.cv_val_num = cfg.data.validation_cv_num

        if self.cfg.data.data_name == "gilon_activity":
            self.activity_map = activity_map

        # Create train dataset. Val is just a subset of train dataset
        if mode in ("train", "val"):
            self.label_df = pd.read_csv(f"src/data/{cfg.data.features_save_dir}/train_label_chunk40_window160_72users.csv")
            sensor_feature_save_path = f"src/data/{cfg.data.features_save_dir}/weight_train_sensors_normalized.pkl"
            meta_feature_save_path = f"src/data/{cfg.data.features_save_dir}/weight_train_meta.pkl"
            if os.path.isfile(sensor_feature_save_path):
                print(f"{sensor_feature_save_path} Exists!, loading..")
                with open(sensor_feature_save_path, "rb") as f:
                    self.sensor_dict = pickle.load(f)
                with open(meta_feature_save_path, "rb") as m:
                    self.meta_dict = pickle.load(m)

            cv_dict_path = f"src/data/{cfg.data.features_save_dir}/cv_dict.pkl"
            with open(cv_dict_path, "rb") as c:
                self.cv_dict = pickle.load(c)

            if mode in {"train"}:
                self.label_df = self.label_df[~self.label_df["ID"].isin(self.cv_dict[self.cv_val_num])]
            elif mode in {"val"}:
                self.label_df = self.label_df[self.label_df["ID"].isin(self.cv_dict[self.cv_val_num])]

        elif mode in ("test"):
            self.label_df = pd.read_csv(f"src/data/{cfg.data.features_save_dir}/test_label_chunk40_window160_72users.csv")
            sensor_test_feature_save_path = f"src/data/{cfg.data.features_save_dir}/weight_test_sensors_normalized.pkl"
            if os.path.isfile(sensor_test_feature_save_path):
                print(f"{sensor_test_feature_save_path} Exists!, loading..")
                with open(sensor_test_feature_save_path, "rb") as f:
                    self.sensor_dict = pickle.load(f)

        elif mode in ("predict"):
            self.label_df = pd.read_csv(f"src/data/{cfg.data.features_save_dir}/test_label_chunk40_window160_72users.csv")
            if permute_predict_name is not None:
                data_dict_path = f"src/data/{cfg.data.features_save_dir}/permute_testdata/{permute_predict_name}.pkl"
            elif noise_predict_name is not None:
                data_dict_path = f"src/data/{cfg.data.features_save_dir}/noise_testdata/{noise_predict_name}.pkl"
            else:
                raise ValueError(f"Unknown permute_predict_name and noise_predict_name are None")

            if os.path.isfile(data_dict_path):
                print(f"{data_dict_path} Exists!, loading..")
                with open(data_dict_path, "rb") as f:
                    self.sensor_dict = pickle.load(f)

        """If you want to perform any ablation on the datasets, please do it here. all the features will be based on the label_df"""
        get_label_statistics(self.label_df)
        self.label_df = self.label_df.reset_index(drop=True)

        if mode == "test":
            self.label_df = self.label_df[["ID", "GLOBAL_ID", "EXP", "Weight", "SPEED", "ACTIVITY"]]
            self.label_df.to_csv(f"{self.cfg.save_output_path}/test_label.csv", index=False)

        if mode in ("train") and self.cfg.task.task_name in ("fullfinetune", "fullfinetune_with_ppt"):
            self.train_ratio = self.cfg.task.train_ratio
            if self.train_ratio < 1.0:
                print(f"Original Number of samples: {len(self.label_df)}")
                self.label_df = self.label_df.sample(frac=self.train_ratio, random_state=42).reset_index(drop=True)
                print(
                    bcolors.OKBLUE
                    + bcolors.BOLD
                    + f"Using {self.train_ratio} fraction of the train data, with seed 42"
                    + bcolors.ENDC
                    + bcolors.ENDC
                )
                print(f"Sampled number of samples: {len(self.label_df)}")
            else:
                print(f"Using full train data, without subsampling")

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, index):
        # TODO: Check if the weighted sampling is working properly
        global_id = self.label_df.iloc[index].GLOBAL_ID
        label = self.label_df[self.label_df["GLOBAL_ID"] == global_id]

        feature = self.sensor_dict[global_id]
        activity_label = self.activity_map[label.EXP.item()]
        y_true = torch.tensor(activity_label, dtype=torch.long)

        return {
            "feature": torch.tensor(feature, dtype=torch.float32),
            "y_true": y_true,
            "global_id": global_id,
        }


def get_label_statistics(label_df):
    """for sanity check"""
    weight_list = list(label_df.Weight.unique())
    weight_list.sort()
    print("-" * 50)
    print(f"Number of users:{len(label_df.ID.unique())}")
    print(f"Number of unique global ID: {len(label_df.GLOBAL_ID.unique())}")
    print(f"Unique weights: {weight_list}")
    print(f"Site Distribution:{label_df.groupby('ID').head(1).Site.value_counts()}")
    print("-" * 50)


activity_map = {
    "EXP01": 0,
    "EXP02": 0,
    "EXP03": 1,
    "EXP04": 1,
    "EXP05": 1,
    "EXP06": 1,
    "EXP07": 1,
    "EXP08": 1,
    "EXP09": 2,
    "EXP10": 2,
    "EXP11": 2,
    "EXP12": 3,
    "EXP13": 3,
    "EXP14": 4,
    "EXP15": 5,
    "EXP16": 6,
}
