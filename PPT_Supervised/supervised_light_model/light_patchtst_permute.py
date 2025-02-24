import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)
from collections import defaultdict
import time

import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics


from supervised_models.PatchTST import PatchTST
from supervised_models.PITS import PITS
from src.loss.patchorder_ssl_loss import (
    TimeOrderLossSSL, 
    FeatureOrderLossSSL, 
    InfoNCETimeLossSSL, 
    InfoNCEFeatureLossSSL, 
    AutomaticWeightedLoss,
)
from src.loss.sequenceshuffler import SequenceShuffler
from src.utils import bcolors

from src.viz_confusion import make_confusion_matrix


class LitModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.light_model.optimizer.lr
        self.initialize_shufflers(cfg)
        self.initialize_losses(cfg)
        self.initialize_metrics(cfg)

        self.select_model()
        self.c_in = cfg.data.c_in

    def training_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch["y_true"]

        # Supervised Loss
        y_pred, outputs = self.model(feature, return_outputs=True)
        patch_embd_from_layers = outputs["patch_embd_from_layers"]

        supervised_loss = self.loss(y_pred.squeeze(), y_true)

        consistency_loss, margin_loss = self.calculate_ssl_losses(feature, patch_embd_from_layers)


        if self.cfg.light_model.ssl_loss.use_awl_loss:
            ssl_loss = self.awl(consistency_loss, margin_loss)
        else:
            ssl_loss = self.lambda_consistency * consistency_loss + self.lambda_margin * margin_loss
        
        loss = supervised_loss + ssl_loss
        train_acc = self.train_accuracy(y_pred, y_true)
        self.log("train_acc", train_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log_losses(
            supervised_loss=supervised_loss,
            consistency_loss=consistency_loss,
            margin_loss=margin_loss,
            loss=loss,
            mode="train",
        )
        return loss

    def validation_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch["y_true"]

        y_pred, outputs = self.model(feature, return_outputs=True)
        patch_embd_from_layers = outputs["patch_embd_from_layers"]

        val_supervised_loss = self.loss(y_pred.squeeze(), y_true)

        # initialize losses
        val_consistency_loss, val_margin_loss = self.calculate_ssl_losses(feature, patch_embd_from_layers)
        
        if self.cfg.light_model.ssl_loss.use_awl_loss:
            val_ssl_loss = self.awl(val_consistency_loss, val_margin_loss)
        else:
            val_ssl_loss = self.lambda_consistency * val_consistency_loss + self.lambda_margin * val_margin_loss

        val_loss = val_supervised_loss + val_ssl_loss
        val_acc = self.val_accuracy(y_pred, y_true)
        self.log("val_acc", val_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log_losses(
            supervised_loss=val_supervised_loss,
            consistency_loss=val_consistency_loss,
            margin_loss=val_margin_loss,
            loss=val_loss,
            mode="val",
        )
        return val_loss

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.test_output_list = defaultdict(list)
        return

    def test_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch["y_true"]
        y_pred  = self.model(feature, return_outputs=False)
        test_loss = self.loss(y_pred.squeeze(), y_true)
        self.test_output_list["test_loss"].append(test_loss)
        self.test_output_list["y_pred"].append(y_pred)
        self.test_output_list["y_true"].append(y_true)

    def on_test_epoch_end(self):
        outputs = self.test_output_list
        test_loss = torch.stack(outputs["test_loss"]).mean().cpu()
        y_pred = torch.argmax(torch.cat(outputs["y_pred"]), dim=1).cpu()
        y_pred_proba = torch.nn.Softmax(dim=1)(torch.cat(outputs["y_pred"])).cpu()
        y_pred_proba_df = pd.DataFrame(
            y_pred_proba.numpy(), columns=[f"pred_{i}" for i in range(self.cfg.data.num_class)]
        )
        y_true = torch.cat(outputs["y_true"]).cpu()

        test_accuracy = accuracy_score(y_true, y_pred)
        if self.cfg.data.num_class == 2:
            test_auroc = roc_auc_score(y_true, y_pred_proba[:, 1])
            test_auprc = average_precision_score(y_true, y_pred_proba[:, 1])
            test_precision = precision_score(y_true, y_pred)
            test_recall = recall_score(y_true, y_pred)
            test_f1 = f1_score(y_true, y_pred)
            print(f"test acc: {test_accuracy:.4f}, test auroc: {test_auroc:.4f}, test auprc: {test_auprc:.4f}")
            print(f"test precision: {test_precision:.4f}, test recall: {test_recall:.4f}, test f1: {test_f1:.4f}")
        else:
            test_auroc = roc_auc_score(y_true, y_pred_proba, multi_class="ovo")
            test_auprc = average_precision_score(y_true, y_pred_proba)
            print(f"test acc: {test_accuracy:.4f}, test auroc: {test_auroc:.4f}, test auprc: {test_auprc:.4f}")

        # Save the test results in the output directory
        test_label = pd.read_csv(f"{self.cfg.save_output_path}/test_label.csv")
        test_label = pd.concat([test_label, y_pred_proba_df], axis=1)
        test_label["y_pred"] = y_pred.numpy()
        test_label["y_true"] = y_true.numpy()
        test_label["test_loss"] = test_loss.item()
        test_label["test_accuracy"] = test_accuracy
        test_label["test_auroc"] = test_auroc
        test_label["test_auprc"] = test_auprc
        test_label["stop_epoch"] = self.trainer.early_stopping_callback.stopped_epoch
        test_label["cv_num"] = self.cfg.data.validation_cv_num
        test_label["model_name"] = self.cfg.model.model_name
        test_label["data_name"] = self.cfg.data.data_name
        test_label.to_csv(
            f"{self.cfg.save_output_path}/cv{self.cfg.data.validation_cv_num}_test_label.csv", index=False
        )
        activity_cf_mat = confusion_matrix(y_true, y_pred)
        make_confusion_matrix(
            activity_cf_mat, f"{self.cfg.save_output_path}/cv{self.cfg.data.validation_cv_num}_confusion_matrix.png"
        )

        if self.cfg.save_pt_file:
            self.save_model_weight()

        del self.test_output_list, outputs

    def predict_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch["y_true"]
        y_pred = self.model(feature, return_outputs=False)
        test_loss = self.loss(y_pred.squeeze(), y_true)

        return {"y_pred": y_pred, "y_true": y_true, "test_loss": test_loss}


    def save_patch_embd(self, patch_embeds, y_true, y_pred, hidden_last):
        current_epoch = self.current_epoch
        patch_embeds = patch_embeds.squeeze().cpu().detach().numpy()
        y_true = y_true.squeeze().cpu().detach().numpy()
        y_pred = y_pred.squeeze().cpu().detach().numpy()
        hidden_last = hidden_last.squeeze().cpu().detach().numpy()
        np.save(
            f"{self.cfg.save_output_path}/patch_embeds_epoch{current_epoch}_{self.cfg.data.validation_cv_num}.npy",
            patch_embeds,
        )
        np.save(
            f"{self.cfg.save_output_path}/y_true_epoch{current_epoch}_{self.cfg.data.validation_cv_num}.npy", y_true
        )
        np.save(
            f"{self.cfg.save_output_path}/y_pred_epoch{current_epoch}_{self.cfg.data.validation_cv_num}.npy", y_pred
        )
        np.save(
            f"{self.cfg.save_output_path}/hidden_last_epoch{current_epoch}_{self.cfg.data.validation_cv_num}.npy",
            hidden_last,
        )


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
        # Set base loss
        self.loss = nn.CrossEntropyLoss()
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


        if cfg.light_model.ssl_loss.use_awl_loss:
            #use bcolor to print
            print(f"{bcolors.WARNING}=====> Using Automatic Weighted Loss {bcolors.ENDC}")
            print(f"{bcolors.WARNING}=====> Lambda will be ignored {bcolors.ENDC}")
            self.awl = AutomaticWeightedLoss(num=2) # Lambda will be ignored.
        else: 
            self.lambda_consistency, self.lambda_margin =  0.0, 0.0

    def initialize_metrics(self, cfg):
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.data.num_class)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.data.num_class)

    def calculate_ssl_losses(self, feature, patch_embd_from_layers):
        """Calculates the Self-Supervised Losses.
        If there is 1 channel, featureconsistency_loss, featuremargin_loss will be 0.
        """
        consistency_loss = 0
        margin_loss = 0

        if self.cfg.light_model.ssl_loss.use_consistency_loss or self.cfg.light_model.ssl_loss.use_margin_loss:
            # now = time.time()
            shuffled_feature_strong, _ = self.shuffle_feature(feature, self.strong_shufflers)
            # print(f"Time taken to shuffle feature: {time.time() - now:.3f} seconds")
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


    def save_model_weight(self, path=None):
        model_weightname = f"cv{self.cfg.data.validation_cv_num}_model.pt"
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

    def log_losses(
        self,
        supervised_loss,
        consistency_loss,
        margin_loss,
        loss,
        mode="train",
    ):
        """Log losses."""
        self.log(f"{mode}_supervised_loss", supervised_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_consistency_loss", consistency_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_margin_loss", margin_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def select_model(self):
        if self.cfg.model.model_name == "patchtst":
            self.model = PatchTST(self.cfg)
        elif self.cfg.model.model_name == "pits":
            self.model = PITS(self.cfg)
        else:
            raise NotImplementedError

        if self.cfg.model.model_name in ["pits_lstm", "patchtst_lstm"]:
            assert self.cfg.data.seq_len % self.cfg.model.patch_len == 0, "seq_len should be divisible by patch_len"
            assert self.cfg.model.patch_len == self.cfg.model.stride, "patch_len should be equal to stride"
            print(f"Using {self.cfg.model.model_name} with patch_len: {self.cfg.model.patch_len}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
