
import torch
from torch import nn
from torch import Tensor


from einops import rearrange, reduce, repeat


# Cell
class PITS_backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.m_cfg = cfg.model

        # Patching
        self.patch_len = self.m_cfg.patch_len
        self.stride = self.m_cfg.stride
        self.padding_patch = self.m_cfg.padding_patch
        context_window = cfg.data.seq_len
        patch_num = int((context_window - self.patch_len)/self.stride + 1)
        if self.padding_patch == 'end': # can be modified to general case
            pad_num = self.patch_len - (context_window % self.patch_len)
            self.padding_patch_layer = nn.ReplicationPad1d((0, pad_num)) 
            patch_num += 1

        c_in = cfg.data.c_in
        d_model = self.m_cfg.d_model
        shared_embedding = self.m_cfg.shared_embedding

        # Backbone
        self.backbone = FC2Encoder(c_in=c_in, patch_len=self.patch_len, d_model=d_model, shared_embedding=shared_embedding)

    def forward(self, z):
        # do patching
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)
        #  In PITS CLS tokens don't make sense as there are no self attention layers

        # model
        z = self.backbone(z)
        return z


class FC2Encoder(nn.Module):
    def __init__(self, c_in, patch_len, d_model=128, shared_embedding=True):
        super().__init__()
        self.c_in = c_in
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        self.act = nn.ReLU(inplace=True)
        if not shared_embedding:
            self.W_P1 = nn.ModuleList()
            self.W_P2 = nn.ModuleList()
            for _ in range(self.c_in):
                self.W_P1.append(nn.Linear(patch_len, d_model))
                self.W_P2.append(nn.Linear(d_model, d_model))
        else:
            self.W_P1 = nn.Linear(patch_len, d_model)
            self.W_P2 = nn.Linear(d_model, d_model)

    def forward(self, x) -> Tensor:
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        # [128, 7, 12, 56]
        """
        x = x.permute(0, 3, 1, 2)
        bs, num_patch, c_in, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(c_in):
                z = self.W_P1[i](x[:, :, i, :])
                x_out.append(z)
                z = self.act(z)
                z = self.W_P2[i](z)  # ??
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P1(x)
            x = self.act(x)
            x = self.W_P2(x)
        x = x.transpose(1, 2)
        x = x.permute(0, 1, 3, 2)
        return x
