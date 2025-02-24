from torch import nn
import torch.nn.functional as F

from src.layers.PatchTST_backbone import PatchTST_backbone

# Cell
from einops import rearrange, reduce

class PatchTST(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = PatchTST_backbone(cfg)
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
            elif self.cfg.model.head_type == "cls":
                self.head = CLSHead(d_model=self.d_model, n_classes=self.num_class)
            else:
                raise ValueError(f"Unknown head_type {self.cfg.model.head_type}")
        else:
            raise ValueError(f"Unknown head_type {self.cfg.model.head_type} or task_name {self.cfg.task.task_name}")

        
        self.use_only_last_layer = self.cfg.light_model.ssl_loss.use_only_last_layer
        if self.use_only_last_layer:
            print("@@@@@@@@@@@@@ Using only the last layer for SSL loss @@@@@@@@@@@@@")

    def forward(self, x, return_outputs=False):  # x: [Batch, seq_len, c_in]
        x = x.permute(0, 2, 1)
        z, patch_embd_from_layers, attention_outs = self.model(x)
        if self.use_only_last_layer:
            patch_embd_from_layers = [z]
        y = self.head(z)

        if return_outputs:
            return y, {
                "patch_embd_from_layers": patch_embd_from_layers, 
                "attention_outs": attention_outs
                }
        else:
            return y
        
class PretrainMaskHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """
        x = rearrange(x, 'b c d p -> b c p d')  # [bs x nvars x num_patch x d_model]
        x = self.linear(self.dropout(x))      # [bs x nvars x num_patch x patch_len]
        x = rearrange(x, 'b c p d -> b c d p')  # [bs x nvars x d_model x num_patch]
        return x
    
class CLSHead(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)
    
    def forward(self, x):
        cls_token = x[:, :, :, 0]
        # average along the channel dimension
        cls_token_mean = reduce(cls_token, "b c pl -> b pl", reduction="mean")
        y = self.linear(cls_token_mean)
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
        x, _ = self.lstm(z_sequence)
        # get the last output
        x = x[:, -1, :]
        # _, (_, c_n) = self.lstm(z_sequence)
        # x = c_n.squeeze()
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