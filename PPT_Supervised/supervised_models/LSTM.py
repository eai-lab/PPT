import torch
import torch.nn as nn
import torch.nn.functional as F

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_first = True


class LSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg = cfg.model
        assert self.model_cfg.model_name == "lstm"
        self.drop = nn.Dropout(self.model_cfg.dropout)
        self.classifier = nn.Linear(self.model_cfg.hidden_size, cfg.data.num_class)
        self.lstm = nn.LSTM(
            input_size=cfg.data.c_in,
            hidden_size=self.model_cfg.hidden_size,
            num_layers=self.model_cfg.num_layers,
            bidirectional=self.model_cfg.bidirectional,
            batch_first=True,
        )

    def forward(self, x, return_outputs=False):
        h0 = torch.zeros(self.model_cfg.num_layers, x.size(0), self.model_cfg.hidden_size).to(device)
        c0 = torch.zeros(self.model_cfg.num_layers, x.size(0), self.model_cfg.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        avg_pool = torch.mean(out, 1)

        x_drop = self.drop(avg_pool)
        out = self.classifier(x_drop)
        if return_outputs:
            return out, {"channel_attention": None}
        else:
            return out
