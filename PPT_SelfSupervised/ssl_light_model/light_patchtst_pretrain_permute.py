import numpy as np
import os

import lightning as L
import torch
import torch.nn as nn
import torchmetrics

from ssl_models.PatchTST import PatchTST
from src.loss.patchorder_ssl_loss import (
    TimeOrderLossSSL, 
    FeatureOrderLossSSL, 
    InfoNCETimeLossSSL, 
    InfoNCEFeatureLossSSL, 
    AutomaticWeightedLoss,
)
from src.loss.sequenceshuffler import SequenceShuffler
from einops import reduce
from src.utils import bcolors
from ssl_utilities.save_embeddings import save_embedding_files  

class LitModel(L.LightningModule):
    """
    A Lightning Module for Self-Supervised Pre-Training
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.light_model.optimizer.lr
        
        self.initialize_shufflers(cfg)
        self.initialize_losses(cfg)
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.data.num_class)

        self.select_model()
        self.c_in = cfg.data.c_in

    def training_step(self, batch, batch_idx):
        feature = batch["feature"]

        # Supervised Loss
        _, outputs = self.model(feature, return_outputs=True)
        patch_embd_from_layers = outputs["patch_embd_from_layers"]

        # initialize losses
        consistency_loss, margin_loss = self.calculate_ssl_losses(feature, patch_embd_from_layers)

        if self.cfg.light_model.ssl_loss.use_awl_loss:
            loss = self.awl(consistency_loss, margin_loss)
        else:
            loss = self.lambda_consistency * consistency_loss + self.lambda_margin * margin_loss

        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log_losses(
            consistency_loss=consistency_loss,
            margin_loss=margin_loss,
            loss=loss,
            mode="train",
        )
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
    

    def calculate_ssl_losses(self, feature, patch_embd_from_layers):
        """Calculates the Self-Supervised Losses.
        If there is 1 channel, featureconsistency_loss, featuremargin_loss will be 0.
        """
        consistency_loss = 0
        margin_loss = 0

        if self.cfg.light_model.ssl_loss.use_consistency_loss or self.cfg.light_model.ssl_loss.use_margin_loss:
            shuffled_feature_strong, _ = self.shuffle_feature(feature, self.strong_shufflers)
            _, outputs_strong = self.model(shuffled_feature_strong, return_outputs=True)
            patch_embd_from_layers_strong = outputs_strong["patch_embd_from_layers"]

        if self.cfg.light_model.ssl_loss.use_consistency_loss:
            timeconsistency_loss = self.calculate_timestep_loss(patch_embd_from_layers, patch_embd_from_layers_strong)
            featureconsistency_loss = self.calculate_featurestep_loss(patch_embd_from_layers, patch_embd_from_layers_strong) if self.c_in != 1 else 0
            consistency_loss = timeconsistency_loss + featureconsistency_loss

        if self.cfg.light_model.ssl_loss.use_margin_loss:
            shuffled_feature_weak, _ = self.shuffle_feature(feature, self.weak_shufflers)
            _, outputs_weak = self.model(shuffled_feature_weak, return_outputs=True)
            patch_embd_from_layers_weak = outputs_weak["patch_embd_from_layers"]

            timemargin_loss = self.calculate_timemargin_loss(patch_embd_from_layers, patch_embd_from_layers_weak, patch_embd_from_layers_strong)
            featuremargin_loss = self.calculate_featuremargin_loss(patch_embd_from_layers, patch_embd_from_layers_weak, patch_embd_from_layers_strong) if self.c_in != 1 else 0
            margin_loss = timemargin_loss + featuremargin_loss

        return consistency_loss, margin_loss


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


    def initialize_shufflers(self, cfg):
        """Initialize sequence shufflers."""
        strong_lower_bound = cfg.shuffler.permute_freq
        weak_lower_bound = 1
        self.strong_shufflers = [
            SequenceShuffler(cfg, permute_freq=freq, permute_strategy=self.cfg.shuffler.strong_permute_strategy)
            for freq in [strong_lower_bound, strong_lower_bound + 1, strong_lower_bound + 2]
        ]
        print(f"Strong shufflers with permute_freq: {[shuffler.permute_freq for shuffler in self.strong_shufflers]}")
        if cfg.light_model.ssl_loss.use_margin_loss:
            self.weak_shufflers = [
                SequenceShuffler(cfg, permute_freq=freq, permute_strategy= self.cfg.shuffler.weak_permute_strategy)
                for freq in [weak_lower_bound, weak_lower_bound + 1, weak_lower_bound + 2]
            ]
            print(f"Weak shufflers with permute_freq: {[shuffler.permute_freq for shuffler in self.weak_shufflers]}")

    def initialize_losses(self, cfg):
        """Initialize loss functions based on configuration."""
        # initialize lambda
        if cfg.light_model.ssl_loss.use_awl_loss:
            #use bcolor to print
            print(f"{bcolors.WARNING}=====> Using Automatic Weighted Loss {bcolors.ENDC}")
            print(f"{bcolors.WARNING}=====> Lambda will be ignored {bcolors.ENDC}")
            self.awl = AutomaticWeightedLoss(num=2) # Lambda will be ignored. 
        self.lambda_consistency, self.lambda_margin =  0.0, 0.0
        
        if cfg.light_model.ssl_loss.use_consistency_loss:
            self.lambda_consistency = cfg.light_model.ssl_loss.lambda_consistency
            self.timestep_loss = TimeOrderLossSSL(cfg)
            self.featurestep_loss = FeatureOrderLossSSL(cfg)
            print(f"Using Consistency Loss with lambda: {self.lambda_consistency}")

        if cfg.light_model.ssl_loss.use_margin_loss:
            self.lambda_margin = cfg.light_model.ssl_loss.lambda_margin
            self.triplet_timemargin_loss = InfoNCETimeLossSSL(cfg)
            self.triplet_featuremargin_loss = InfoNCEFeatureLossSSL(cfg)
            print(f"Using Margin Loss with lambda: {self.lambda_margin}")


    def shuffle_feature(self, feature, shufflers):
        """Shuffle features using a randomly selected shuffler."""
        coin_toss = np.random.randint(0, len(shufflers))
        return shufflers[coin_toss](feature), shufflers[coin_toss].permute_freq

    def calculate_timemargin_loss(self, patch_embeds, patch_embeds_shuffled_weak, patch_embeds_shuffled_strong):
        """Calculate margin Time loss."""
        timemargin_loss = sum(
            self.triplet_timemargin_loss(pe, pew, pes)
            for pe, pew, pes in zip(patch_embeds, patch_embeds_shuffled_weak, patch_embeds_shuffled_strong)
        )
        return timemargin_loss / len(patch_embeds)

    def calculate_featuremargin_loss(self, patch_embeds, patch_embeds_shuffled_weak, patch_embeds_shuffled_strong):
        """Calculate margin Feature loss."""
        featuremargin_loss = sum(
            self.triplet_featuremargin_loss(pe, pew, pes)
            for pe, pew, pes in zip(patch_embeds, patch_embeds_shuffled_weak, patch_embeds_shuffled_strong)
        )
        return featuremargin_loss / len(patch_embeds)

    def calculate_timestep_loss(self, patch_embeds, patch_embeds_shuffled_strong):
        """Calculate time loss."""
        return sum(self.timestep_loss(pe, pes) for pe, pes in zip(patch_embeds, patch_embeds_shuffled_strong)) / len(
            patch_embeds
        )

    def calculate_featurestep_loss(self, patch_embeds, patch_embeds_shuffled_strong):
        """Calculate feature loss."""
        return sum(
            self.featurestep_loss(pe, pes) for pe, pes in zip(patch_embeds, patch_embeds_shuffled_strong)
        ) / len(patch_embeds)


    def log_losses(
        self,
        consistency_loss,
        margin_loss,
        loss,
        mode="train",
    ):
        """Log losses."""
        self.log(f"{mode}_consistency_loss", consistency_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_margin_loss", margin_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def select_model(self):
        if self.cfg.model.model_name == "patchtst":
            self.model = PatchTST(self.cfg)
        elif self.cfg.model.model_name == "pits":
            raise ValueError(f"Please use light_pits_pretrain_permute or light_pits_pretrain_mask instead of light_patchtst_pretrain.")
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
