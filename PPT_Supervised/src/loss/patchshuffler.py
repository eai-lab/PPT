"""patchshuffler: shuffles at the patch level"""
import pandas as pd
import numpy as np
import glob
import os
import torch
from torch import nn
import random
from abc import ABC, abstractmethod

from einops import rearrange
import time


class BaseShuffler(nn.Module, ABC):
    """Pre-compute shuffled idx to save sorting time during training"""

    def __init__(self, cfg, data_sample, permute_freq=None, permute_strategy="random"):
        super().__init__()
        assert (
            len(data_sample.shape) == 4
        ), f"Data sample must be of shape [B, Channel, Embedding, Patch_num], but got {data_sample.shape}"
        self.cfg = cfg
        self.shuffled_idx_base_path = cfg.shuffler.shuffled_idx_base_path
        assert permute_strategy in [
            "random",
            "vicinity",
            "farthest",
        ], f"Unknown permutation strategy: {permute_strategy}"

        self.B, self.C, _, self.P = data_sample.shape
        self.num_permutations = 1000

        if permute_freq is None:
            # if None, use config setting
            self.permute_freq = self.cfg.shuffler.permute_freq
            self.permute_strategy = self.cfg.shuffler.permute_strategy.lower()
        else:
            # if not None, use the passed in value
            self.permute_freq = permute_freq
            self.permute_strategy = permute_strategy.lower()

        file_name = self.load_data_name()

        if os.path.isfile(file_name):
            self.shuffled_idx = torch.load(file_name)
        else:
            print(f"Shuffled idx not found at {file_name}. Creating new shuffled idx...")
            now = time.time()
            self.shuffled_idx = self.construct_shuffled_idx()
            print(f"Time taken to create shuffled idx: {time.time() - now:.3f} seconds")
            self.save_data(self.shuffled_idx, file_name)
        self.shuffled_idx = self.shuffled_idx.to(f"cuda:{cfg.gpu_id}")
        # print device
        print(f"Shuffled idx device: {self.shuffled_idx.device}")

    def save_data(self, shuffled_idx, file_name):
        torch.save(shuffled_idx, file_name)
        print(f"Saved shuffled idx to {file_name}")

    def forward(self, X):
        """X: [B, C, E, P]"""
        X = rearrange(X, "b c e p -> b e c p")
        rand_idx = torch.randint(0, self.num_permutations - 1, (1,)).to(X.device)
        # print(f"Random idx: {rand_idx}")
        shuffle_order = self.shuffled_idx[rand_idx]
        expanded_shuffle_order = shuffle_order.expand(X.size(0), X.size(1), -1, -1)
        X_prime = torch.gather(X, 3, expanded_shuffle_order)
        X_shuffled = rearrange(X_prime, "b e c p -> b c e p")
        return X_shuffled

    def load_data_name(self):
        data_name = (
            f"C{self.cfg.data.c_in}_P{self.P}_permute{str(self.permute_freq).zfill(2)}_{self.permute_strategy}.pt"
        )
        print("*" * 80)
        print(f"Loading shuffled idx from {self.shuffled_idx_base_path}/{data_name}")
        print("*" * 80)
        return f"{self.shuffled_idx_base_path}/{data_name}"

    @abstractmethod
    def construct_shuffled_idx(self):
        pass


# Shuffle at Patch level
class PatchSoftShuffler(BaseShuffler):
    def __init__(self, cfg, data_sample, permute_freq, permute_strategy="random"):
        super().__init__(cfg, data_sample, permute_freq, permute_strategy)

    def construct_shuffled_idx(self):
        # create a shuffled tensor of shape [1000, c_in, patch_num]
        # 500 is the number of possible soft shuffle options. We sample from this tensor
        #
        data = np.arange(0, self.P)

        total_shuffled_idx = []
        print(
            f"Constructing shuffled idx with strategy {self.permute_strategy} / freq:{self.permute_freq} for {self.num_permutations} times"
        )
        for i in range(self.num_permutations):
            shuffled_idx_list = []
            for feat in range(self.C):
                copy_data = data.copy()
                num_patches = len(copy_data)
                for _ in range(self.permute_freq):
                    if self.permute_strategy == "random":
                        idx1, idx2 = random.sample(range(num_patches), 2)
                    elif self.permute_strategy == "vicinity":
                        idx1 = random.randint(0, num_patches - 2)
                        idx2 = idx1 + 1
                    elif self.permute_strategy == "farthest":
                        idx1, idx2 = random.sample(range(num_patches), 2)
                        while abs(idx1 - idx2) < 2:
                            idx1, idx2 = random.sample(range(num_patches), 2)
                    else:
                        raise ValueError("Unknown permutation strategy: {}".format(self.permute_strategy))

                    tmp_value = copy_data[idx1]
                    copy_data[idx1] = copy_data[idx2]
                    copy_data[idx2] = tmp_value

                    del idx1, idx2, tmp_value

                shuffled_idx_list.append(copy_data)
            shuffled_idx_list = np.stack(shuffled_idx_list, axis=0)
            total_shuffled_idx.append(shuffled_idx_list)
        total_shuffled_idx = torch.tensor(np.stack(total_shuffled_idx, axis=0))

        return total_shuffled_idx
