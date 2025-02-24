import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce, repeat
from src.loss.infonce_loss import InfoNCE


class InfoNCEFeatureLossSSL(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = torch.nn.CrossEntropyLoss()
        self.lstm = torch.nn.LSTM(
            input_size=cfg.model.d_model,
            hidden_size=cfg.model.d_model,
            num_layers=1,
            batch_first=True,
            dropout=0.1,
            bidirectional=False,
        )
        self.loss = InfoNCE(temperature=0.1, reduction="mean", negative_mode="unpaired")


    def forward(self, anchor, positive, negative):
        # anchor, positive, negative: [batch, c_in, embed_dim, patch_num]

        anchor = rearrange(anchor, "b c e p -> (b p) c e")
        positive = rearrange(positive, "b c e p -> (b p) c e")
        negative = rearrange(negative, "b c e p -> (b p) c e")

        patch_concat = torch.cat([anchor, positive, negative], dim=0)
        # lstm
        _, (_, c_n) = self.lstm(patch_concat)
        # get the context vector
        lstm_out = c_n.squeeze()

        anchor, positive, negative = torch.split(lstm_out, anchor.shape[0], dim=0)

        loss = self.loss(anchor, positive, negative)

        return loss


class InfoNCETimeLossSSL(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = torch.nn.CrossEntropyLoss()
        self.lstm = torch.nn.LSTM(
            input_size=cfg.model.d_model,
            hidden_size=cfg.model.d_model,
            num_layers=1,
            batch_first=True,
            dropout=0.1,
            bidirectional=False,
        )
        self.loss = InfoNCE(temperature=0.1, reduction="mean", negative_mode="unpaired")


    def forward(self, anchor, positive, negative):
        # anchor, positive, negative: [batch, c_in, embed_dim, patch_num]

        # anchor = reduce(anchor, "b c e p -> b p e", reduction="mean")
        # positive = reduce(positive, "b c e p -> b p e", reduction="mean")
        # negative = reduce(negative, "b c e p -> b p e", reduction="mean")
        anchor = rearrange(anchor, "b c e p -> (b c) p e")
        positive = rearrange(positive, "b c e p -> (b c) p e")
        negative = rearrange(negative, "b c e p -> (b c) p e")

        patch_concat = torch.cat([anchor, positive, negative], dim=0)
        # lstm
        _, (_, c_n)  = self.lstm(patch_concat)
        # get the context vector
        lstm_out = c_n.squeeze()

        anchor, positive, negative = torch.split(lstm_out, anchor.shape[0], dim=0)

        loss = self.loss(anchor, positive, negative)

        return loss


class TimeOrderLossSSL(torch.nn.Module):
    """Self Supervised-Setting"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = torch.nn.CrossEntropyLoss()
        self.lstm = torch.nn.LSTM(
            input_size=cfg.model.d_model,
            hidden_size=cfg.model.d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.linear_head = torch.nn.Linear(cfg.model.d_model, 1)
        self.loss = torch.nn.BCEWithLogitsLoss()


    def forward(self, patch_embed, patch_embed_shuffled):
        # patch_embed: [batch, c_in, embed_dim, patch_num]
        # perform average pooling along c_in
        # patch_temporal_embed = reduce(patch_embed, "b c e p -> b p e", reduction="mean")
        patch_temporal_embed = rearrange(patch_embed, "b c e p -> (b c) p e")
        patch_shuffled_embed = rearrange(patch_embed_shuffled, "b c e p -> (b c) p e")
        label = torch.cat([torch.ones(patch_temporal_embed.shape[0]), torch.zeros(patch_temporal_embed.shape[0])]).float()
        label = label.to(patch_embed.device)

        # patch_shuffled_embed = reduce(patch_embed_shuffled, "b c e p -> b p e", reduction="mean")

        # concat the original and permuted patch embed along batch dim
        patch_temporal_embed_concat = torch.cat([patch_temporal_embed, patch_shuffled_embed], dim=0)
        # lstm
        _, (_, c_n)  = self.lstm(patch_temporal_embed_concat)
        # lstm_out: [batch, seq_len, hidden_size]
        # perform average pooling along seq_len
        # lstm_out = torch.mean(lstm_out, dim=1)
        lstm_out = c_n.squeeze()
        # lstm_out: [batch, hidden_size]
        # linear head
        logits = self.linear_head(lstm_out)
        # logits: [batch, 2]
        loss = self.loss(logits.squeeze(), label)

        # measure accuracy
        acc = torch.sum(torch.round(torch.sigmoid(logits.squeeze())) == label).float() / label.shape[0]
        # print(f"Time Order Loss Accuracy: {acc}")

        return loss


class FeatureOrderLossSSL(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = torch.nn.CrossEntropyLoss()
        self.lstm = torch.nn.LSTM(
            input_size=cfg.model.d_model,
            hidden_size=cfg.model.d_model,
            num_layers=1,
            batch_first=True,
            dropout=0.1,
            bidirectional=False,
        )
        self.linear_head = torch.nn.Linear(cfg.model.d_model, 1)
        self.loss = torch.nn.BCEWithLogitsLoss()


    def forward(self, patch_embed, patch_embed_shuffled):
        # patch_embed: [batch, c_in, embed_dim, patch_num]

        patch_feature_embed = rearrange(patch_embed, "b c e p -> (b p) c e")
        patch_shuffled_feature_embed = rearrange(patch_embed_shuffled, "b c e p -> (b p) c e")

        label = torch.cat(
            [torch.ones(patch_feature_embed.shape[0]), torch.zeros(patch_feature_embed.shape[0])]
        ).float()
        label = label.to(patch_embed.device)

        # concat the original and permuted patch embed along batch dim
        patch_temporal_embed_concat = torch.cat([patch_feature_embed, patch_shuffled_feature_embed], dim=0)
        # lstm
        _, (_, c_n) = self.lstm(patch_temporal_embed_concat)
        # lstm_out: [batch, seq_len, hidden_size]
        # perform average pooling along seq_len
        # lstm_out = torch.mean(lstm_out, dim=1)
        lstm_out = c_n.squeeze()
        # lstm_out: [batch, hidden_size]
        # linear head
        logits = self.linear_head(lstm_out)
        # logits: [batch, 2]
        loss = self.loss(logits.squeeze(), label)

        # measure accuracy
        acc = torch.sum(torch.round(torch.sigmoid(logits.squeeze())) == label).float() / label.shape[0]
        # print(f"Feature Order Loss Accuracy: {acc}")

        return loss


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)
        self.counter =0
    def forward(self, *x):
        if self.counter % 100 == 0:
            print(f"params: {self.params}")
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        self.counter += 1
        return loss_sum