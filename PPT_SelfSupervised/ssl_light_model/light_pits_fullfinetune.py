import pandas as pd
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

import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics

from ssl_models.PITS import PITS
from src.utils import bcolors
from src.viz_confusion import make_confusion_matrix


class LitModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.light_model.optimizer.lr
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.data.num_class)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.data.num_class)
        self.select_model()
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.load_model_weight()

    def training_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch["y_true"]
        if batch_idx == 0:
            print(y_true[:5])

        y_pred = self.model(feature, return_outputs=False)
        loss = self.loss(y_pred.squeeze(), y_true)
        train_acc = self.train_accuracy(y_pred, y_true)
        self.log("train_acc", train_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch["y_true"]

        y_pred = self.model(feature, return_outputs=False)
        val_loss = self.loss(y_pred.squeeze(), y_true)


        val_acc = self.val_accuracy(y_pred, y_true)
        self.log("val_acc", val_acc, on_epoch=True, prog_bar=True, logger=True)
        return val_loss


    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.test_output_list = defaultdict(list)
        return

    def test_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch["y_true"]
        y_pred = self.model(feature, return_outputs=False)
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
            test_f1 = f1_score(y_true, y_pred, average="macro")
            test_precision = precision_score(y_true, y_pred, average="macro")
            test_recall = recall_score(y_true, y_pred, average="macro")
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
        test_label["test_precision"] = test_precision
        test_label["test_recall"] = test_recall
        test_label["test_f1"] = test_f1
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

        del self.test_output_list, outputs

    def predict_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch["y_true"]
        y_pred, _, attention_outs = self.model(feature, return_outputs=False)
        test_loss = self.loss(y_pred.squeeze(), y_true)

        return {"y_pred": y_pred, "y_true": y_true, "test_loss": test_loss}

    def load_model_weight(self):
        pretrain_file = self.cfg.pretrain_pt_file
        if pretrain_file is None:
            print("=====> No pretrained weight is loaded")
            print("Training from scratch!!!!!, Is this what you want? Check the config file.")
            print('===============================')
            pass
        else:
            root_path = self.cfg.load_pt_path
            weight_path = f"{root_path}/{pretrain_file}"
            # add bcolors
            print(f"{bcolors.WARNING}=====>  Fine-Tune data name: {self.cfg.data.data_name} {bcolors.ENDC}")
            # print(f"{bcolors.WARNING}=====> Using Cross-Patch Scenario?: {self.cfg.task.cross_patch_scenario} {bcolors.ENDC}")
            model_dict = self.model.state_dict()
            pretrained_ckpt = torch.load(weight_path)
            best_epoch = pretrained_ckpt["epoch"]
            pretrained_dict = pretrained_ckpt["model_state_dict"]

            matched_layers =0
            unmatched_layers = []
            transferred_params = 0
            for name, param in model_dict.items():
                if 'head' in name: continue
                if name in pretrained_dict:
                    matched_layers += 1
                    input_param = pretrained_dict[name]
                    if input_param.shape == param.shape:
                        param.copy_(input_param)
                        transferred_params += param.numel()
                    else:
                        unmatched_layers.append(name)
                else:
                    unmatched_layers.append(name)
                    pass # these are weights that weren't in the original model, such as a new head (SIMMTM)
            if matched_layers == 0:
                raise Exception("No shared weight names were found between the models")
            else:
                if len(unmatched_layers) > 0:
                    print(f'check unmatched_layers: {unmatched_layers}')
                else:
                    print(f"weights from {weight_path} (Best Pretraining Epoch: {best_epoch}) successfully transferred!\n")
            # print the total number of params transferred and not transferred
            print(f"Transferred params: {transferred_params}, Not transferred params: {self.num_params - transferred_params}")
            print(f"Percentage of transferred params: {transferred_params/self.num_params*100:.2f}%")
            self.model = self.model.to(self.device)
        

    def select_model(self):
        if self.cfg.model.model_name == "pits":
            self.model = PITS(self.cfg)
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
