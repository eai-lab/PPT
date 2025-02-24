# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from src.layers.PITS_layers import PITS_backbone
from einops import rearrange, reduce, repeat


class PITS(nn.Module):
    def __init__(self, cfg, verbose: bool = False):
        super().__init__()

        # load parameters
        self.cfg = cfg
        self.model = PITS_backbone(cfg)
        self.d_model = self.cfg.model.d_model
        self.num_class = self.cfg.data.num_class

        if self.cfg.task.task_name == "supervised":
            print(f"Using {self.cfg.model.head_type} head for finetune")
            if self.cfg.model.head_type == "lstm":
                self.head = LSTMHead(d_model=self.d_model, n_classes=self.num_class)
            elif self.cfg.model.head_type == "last":
                self.head = ClassificationHead(
                    n_vars=cfg.data.c_in, d_model=self.d_model, n_classes=self.num_class, head_dropout=0.1
                )
            else:
                raise ValueError(f"Unknown head_type {self.cfg.model.head_type}")
        else:
            raise ValueError(f"Unknown head_type {self.cfg.model.head_type} or task_name {self.cfg.task.task_name}")
        

    def forward(self, x, return_outputs=False):  # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        patch_embd_from_layers = self.model(x)
        y = self.head(patch_embd_from_layers)

        if return_outputs:
            return y, {
                "patch_embd_from_layers": [patch_embd_from_layers], 
                }
        else:
            return y

class LSTMHead(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )
        self.linear = nn.Linear(d_model, n_classes)
        
    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        z_sequence = reduce(x, "b c pl pn-> b pl pn", reduction="mean")
        z_sequence = rearrange(z_sequence, "b pl pn -> b pn pl")
        _, (_, c_n) = self.lstm(z_sequence)
        # get the last output
        x = c_n.squeeze()
        y = self.linear(x)
        return y
        
        
    
class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x[:, :, :, -1]  # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)  # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)  # y: bs x n_classes
        return y
    

class ClassificationHead_max(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

    def forward(self, x):
        x = self.flatten(x)
        x, _ = torch.max(x,dim=2) # (64,1,128,125) -> (64,128,125) -> (64,128)
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y
    
class ClassificationHead_avg(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.mean(x,dim=2) # (64,1,128,125) -> (64,128,125) -> (64,128)
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y
    
class ClassificationHead_concat(nn.Module):
    def __init__(self, n_vars, d_model, num_patch_new, n_classes, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model*num_patch_new, n_classes)
        self.flatten = nn.Flatten(start_dim=1,end_dim=3)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y