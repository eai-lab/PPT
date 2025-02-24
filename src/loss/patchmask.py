"""sequence shuffler: shuffles at the sequence level"""
import pandas as pd
import numpy as np
import glob
import os
import torch
from torch import nn
import random
from easydict import EasyDict
from einops import rearrange
from src.utils import bcolors


class PatchMask(nn.Module):
    def __init__(self, cfg):
        """Mask at sequence level"""
        super().__init__()
        self.cfg = cfg

        patch_len = cfg.model.patch_len
        patch_num = int(cfg.data.seq_len // patch_len)
        data_sample = torch.randn(1, cfg.data.c_in, 1, patch_num) # pass in a dummy patch sample shape
        self.mask_ratio = cfg.light_model.ssl_loss.mask_ratio
        self.patchifier = Patchify(cfg, patch_len)
        if self.cfg.light_model.light_name == "light_patchtst_pretrain_cl":
            print(f"{bcolors.OKGREEN}Using Complementary Masking{bcolors.ENDC}")
            self.masker = ComplementaryMaskModel(cfg, data_sample)
        else:
            self.masker = MaskModel(cfg,data_sample)
        self.unpatchifier = UnPatchify()

    def forward(self, x):
        # x: [bs x seq_len x c_in]
        x = rearrange(x, "b s c -> b c s")
        patched_x = self.patchifier(x)
        unmasked_patched_x, masked_patched_x, mask = self.masker(patched_x)
        unmasked_input_x = self.unpatchifier(unmasked_patched_x)
        unmasked_input_x = rearrange(unmasked_input_x, "b c s -> b s c") # This is the unmasked part.

        masked_input_x = self.unpatchifier(masked_patched_x)
        masked_input_x = rearrange(masked_input_x, "b c s -> b s c")
        return unmasked_input_x, masked_input_x, masked_patched_x, mask
    
class MaskModel(nn.Module):
    def __init__(self, cfg, data_sample):
        super().__init__()
        self.cfg = cfg
        self.mask_ratio = cfg.light_model.ssl_loss.mask_ratio
        self.B, self.C, _, self.P = data_sample.shape
        
    def forward(self, x):
        """x: [B, C, Patch_len, Patch_num],"""
        mask = torch.rand(self.B, self.C, self.P) < self.mask_ratio
        mask = mask.to(x.device)
        # expand mask to the same shape as x
        mask = mask.unsqueeze(2).expand_as(x)
        masked_x = x * mask # masked_x is the patch that needs to be reconstructed
        unmasked_x = x * (~mask) # unmasked_x is the patch that is used for reconstruction
        return unmasked_x, masked_x, mask

class ComplementaryMaskModel(nn.Module):
    def __init__(self, cfg, data_sample):
        super().__init__()
        self.cfg = cfg
        self.B, self.C, _, self.P = data_sample.shape
        
    def forward(self, x):
        """x: [B, C, Patch_len, Patch_num],"""
        # mask only the even patches
        mask = torch.ones(self.B, self.C, self.P)
        mask[:, :, 1::2] = 0
        # convert to boolean
        mask = mask.bool().to(x.device)
        # expand mask to the same shape as x
        mask = mask.unsqueeze(2).expand_as(x)
        masked_x = x * mask
        unmasked_x = x * (~mask)
        return unmasked_x, masked_x, mask


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
