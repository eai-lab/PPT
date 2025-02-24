"""sequence shuffler: shuffles at the sequence level"""
import pandas as pd
import numpy as np
import glob
import os
import torch
from torch import nn
import random
from src.loss.patchshuffler import PatchSoftShuffler
from easydict import EasyDict
from einops import rearrange


class SequenceShuffler(nn.Module):
    def __init__(self, cfg, permute_freq=None, permute_strategy="random"):
        """Shuffles at the sequence level"""
        super().__init__()
        self.cfg = cfg
        self.shuffled_idx_base_path = cfg.shuffler.shuffled_idx_base_path

        patch_len = cfg.model.patch_len
        patch_num = int(cfg.data.seq_len // patch_len)
        # set as attribute to make it accessible
        self.permute_freq = permute_freq
        self.permute_strategy = permute_strategy

        data_sample = torch.randn(1, cfg.data.c_in, 1, patch_num)
        self.patchifier = Patchify(cfg, patch_len)
        self.patchshuffler = PatchSoftShuffler(cfg, data_sample, permute_freq, permute_strategy)
        self.unpatchifier = UnPatchify()

    def forward(self, x):
        # x: [bs x seq_len x c_in]
        x = rearrange(x, "b s c -> b c s")
        patched_x = self.patchifier(x)
        shuffled_patched_x = self.patchshuffler(patched_x)
        x = self.unpatchifier(shuffled_patched_x)
        x = rearrange(x, "b c s -> b s c")
        return x  # [bs x seq_len x c_in]


class Patchify(nn.Module):
    def __init__(self, cfg, patch_len):
        super().__init__()
        # RevIn
        self.m_cfg = cfg.model
        # no overlapping
        self.patch_len = patch_len
        self.stride = patch_len

    def forward(self, x):  # z: [bs x seq_len x c_in]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x c_in x patch_num x patch_len]
        x = rearrange(x, "b c pn pl -> b c pl pn")  # z: [bs x c_in x patch_len x patch_num]

        return x  # z: [bs x c_in x patch_len x patch_num]


class UnPatchify(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = rearrange(x, "b c pl pn -> b c (pn pl)")
        return x


if __name__ == "__main__":
    cfg = EasyDict()
    cfg.gpu_id = 0
    cfg.model = EasyDict()
    cfg.model.patch_len = 5
    cfg.model.stride = 5
    cfg.model.padding_patch = None
    cfg.data = EasyDict()
    cfg.data.c_in = 2
    cfg.data.seq_len = 20
    cfg.shuffler = EasyDict()
    cfg.shuffler.shuffled_idx_base_path = "shuffled_idx"
    cfg.shuffler.permute_freq = 1
    cfg.shuffler.permute_strategy = "random"
    sequence_shuffler = SequenceShuffler(cfg)
    x = torch.arange(40).reshape(1, cfg.data.seq_len, cfg.data.c_in)  # batch_size x seq_len x c_in
    print(f"original x: {x}")
    print(f"Patch Len: {cfg.model.patch_len}, Stride: {cfg.model.stride}")
    y = sequence_shuffler(x.to(f"cuda:{cfg.gpu_id}"))
    print(f"shuffled x: {y}")
