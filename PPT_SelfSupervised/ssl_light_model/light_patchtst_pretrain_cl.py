import numpy as np
import os

import lightning as L
import torch
import torch.nn as nn
import torchmetrics

from ssl_models.PatchTST import PatchTST
from src.loss.patchmask import PatchMask
from src.loss.infonce_loss import InfoNCE
from einops import reduce, rearrange
from ssl_utilities.save_embeddings import save_embedding_files  

class LitModel(L.LightningModule):
    """
    A Lightning Module for a machine learning model with various loss functions and shuffling mechanisms.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.light_model.optimizer.lr
        self.PatchMasker = PatchMask(cfg)
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.data.num_class)
        self.loss = InfoNCE(temperature=0.1, reduction="mean", negative_mode="paired")
        self.select_model()

    def training_step(self, batch, batch_idx):
        feature = batch["feature"]

        # masked_feature [bs x seq_len x c_in], target_patch [bs x c_in x patch_num x patch_len], mask [bs x c_in x patch_num x patch_len]
        masked_feature, masked_complementary_feature,  target_patch, mask = self.PatchMasker(feature)

        # Supervised Loss
        patch_embed_last_layer_masked, _  = self.model(masked_feature, return_outputs=True)

        # Get only the predicted patches
        patch_embed_last_layer_complementary, _ = self.model(masked_complementary_feature, return_outputs=True)

        b, c, e, p = patch_embed_last_layer_masked.shape
        # max pool for every two patches
        patch_embed_masked = rearrange(patch_embed_last_layer_masked, "b c e (p1 p2) -> b c e p1 p2", p1=2, p2=p//2)
        patch_embed_complementary = rearrange(patch_embed_last_layer_complementary, "b c e (p1 p2) -> b c e p1 p2", p1=2, p2=p//2)

        # max pool for every two patches
        patch_embed_masked = reduce(patch_embed_masked, "b c e p1 p2 -> b e p2", reduction="max")
        patch_embed_complementary = reduce(patch_embed_complementary, "b c e p1 p2 -> b e p2", reduction="max")

        patch_embed_masked = rearrange(patch_embed_masked, "b e p -> (b p) e")
        patch_embed_complementary = rearrange(patch_embed_complementary, "b e p -> (b p) e")

        loss = self.loss(patch_embed_masked, patch_embed_complementary)

        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch["y_true"]
        _, outputs = self.model(feature, return_outputs=True)
        patch_embd_from_layers = outputs["patch_embd_from_layers"]
        attention_outs = outputs["attention_outs"]
        loss = torch.tensor(0.0)
        # obtain instance level representation by average pooling along c_in, and patch_num
        if self.cfg.save_embedding:
            save_embedding_files(y_true, patch_embd_from_layers, batch_idx, self.cfg)
            
        len_stages = len(patch_embd_from_layers)
        patch_embed_outs = []
        for i in range(len_stages):
            patch_embed_outs.append(reduce(patch_embd_from_layers[i], "b c e p -> b e", reduction="mean"))
        

        if self.cfg.save_embedding and batch_idx == 0:
            if not os.path.exists(self.cfg.save_embedding_path):
                os.makedirs(self.cfg.save_embedding_path)
            last_layer_patch_embeds = patch_embd_from_layers[-1]
            np.save(f"{self.cfg.save_embedding_path}/patch_embeds_{batch_idx}.npy", last_layer_patch_embeds.cpu().numpy())
            # save y_true 
            np.save(f"{self.cfg.save_embedding_path}/y_true_{batch_idx}.npy", y_true.cpu().numpy())
            
        return {
            "loss": loss,
            "y_true": y_true,
            "patch_embed_out": patch_embed_outs,
            "attention_outs": attention_outs,
        }
    

    def save_model_weight(self, path=None):
        model_weightname = f"cv{self.cfg.data.validation_cv_num}_pretrain.pt"
        encoder_ckpt = {'epoch': self.current_epoch, 'model_state_dict': self.model.state_dict()}
        if path is None: 
            root_path = self.cfg.save_pt_path
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            path = os.path.join(root_path, model_weightname)
        torch.save(encoder_ckpt, path)
        print("=====================================")
        print(f"Best pretrain model epoch: {self.current_epoch} saved to {path}")
        print("=====================================")



    def select_model(self):
        if self.cfg.model.model_name == "patchtst":
            self.model = PatchTST(self.cfg)
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
