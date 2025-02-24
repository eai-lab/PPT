import numpy as np
import pandas as pd
import glob
from omegaconf import DictConfig

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from ssl_utilities import _eval_protocols as eval_protocols

def import_ssl_lightmodule(cfg: DictConfig, log):
    """
    Load the data module based on the data name
    :param cfg: Configuration dictionary. [omegaconf.dictconfig.DictConfig]
    :return: Data module. [pytorch_lightning.LightningDataModule]
    """
    if cfg.light_model.light_name == "light_patchtst_pretrain_permute":
        from ssl_light_model.light_patchtst_pretrain_permute import LitModel
    elif cfg.light_model.light_name == "light_patchtst_pretrain_mask":
        from ssl_light_model.light_patchtst_pretrain_mask import LitModel
    elif cfg.light_model.light_name == "light_patchtst_pretrain_cl":
        from ssl_light_model.light_patchtst_pretrain_cl import LitModel
    elif cfg.light_model.light_name == "light_pits_pretrain_permute":
        from ssl_light_model.light_pits_pretrain_permute import LitModel
    elif cfg.light_model.light_name == "light_pits_pretrain_mask":
        from ssl_light_model.light_pits_pretrain_mask import LitModel
    else:
        raise ValueError(f"Unknown lightning model {cfg.light_model.light_name}. ")

    return LitModel(cfg)


def eval_classification(cfg, train_repr, train_labels, test_repr, test_labels, eval_protocol="linear"):
    assert train_labels.ndim == 1 or train_labels.ndim == 2

    if eval_protocol == "linear":
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == "svm":
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == "knn":
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, "unknown evaluation protocol"

    def merge_dim01(array):
        return array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    clf = fit_clf(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)
    if eval_protocol == "linear":
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max() + 1))

    if cfg.data.num_class == 2:
        auroc = roc_auc_score(test_labels_onehot.reshape(-1), y_score[:, 1])
        auprc = average_precision_score(test_labels_onehot.reshape(-1), y_score[:, 1])
        f1 = f1_score(test_labels, clf.predict(test_repr))
        precision = precision_score(test_labels, clf.predict(test_repr))
        recall = recall_score(test_labels, clf.predict(test_repr))

    else:
        auroc = roc_auc_score(test_labels_onehot, y_score, multi_class="ovo")
        auprc = average_precision_score(test_labels_onehot, y_score)
        f1 = f1_score(test_labels, clf.predict(test_repr), average="macro")
        precision = precision_score(test_labels, clf.predict(test_repr), average="macro")
        recall = recall_score(test_labels, clf.predict(test_repr), average="macro")
    print(
        f"acc: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, auroc: {auroc:.3f}, auprc: {auprc:.3f}"
    )
    return clf, {
        "acc": acc,
        "auroc": auroc,
        "auprc": auprc,
        "f1_score": f1,
        "precision_score": precision,
        "recall_score": recall,
    }

def collect_ssl_predict_epoch_end(outputs):
    test_loss = torch.stack([x["loss"] for x in outputs]).cpu().mean()
    y_true = torch.cat([x["y_true"] for x in outputs], dim=0).cpu().numpy()
    patch_embed_outs_cat = [x["patch_embed_out"] for x in outputs]
    len_stages = len(patch_embed_outs_cat[0])

    patch_embed_outs = []
    for i in range(len_stages):
        patch_embed_outs.append(torch.cat([x[i] for x in patch_embed_outs_cat], dim=0).cpu().numpy())

    all_average_embed_out = np.stack(patch_embed_outs).mean(axis=0)
    all_maxed_embed_out = np.stack(patch_embed_outs).max(axis=0)

    return {
        "test_loss": test_loss.item(),
        "y_true": y_true,
        "patch_embed_out": patch_embed_outs,
        "all_average_embed_out": all_average_embed_out,
        "all_maxed_embed_out": all_maxed_embed_out,
    }


def save_patch_embeds(cfg, outputs):
    # obtain the last layer
    patch_embed_outs = torch.cat([x["patch_embed_out"][-1] for x in outputs], dim=0).cpu().numpy()
    np.save(f"{cfg.save_embedding_path}/patch_embed_cv{cfg.data.validation_cv_num}.npy", patch_embed_outs)
    print(f"patch_embed_cv{cfg.data.validation_cv_num}.npy saved")


def save_attention_files(cfg, outputs):
    for layer_i in range(cfg.model.n_layers):
        attention_outs = torch.cat([x["attention_outs"][layer_i] for x in outputs], dim=0).cpu().numpy()
        np.save(f"{cfg.save_output_path}/attention_cv{cfg.data.validation_cv_num}_{layer_i}.npy", attention_outs)
        print(f"attention_outs_layer_{layer_i}.npy saved")


def construct_output_results(cfg, embed_dicts):
    # construct output results
    len_stages = len(embed_dicts) - 2  # exclude all_average_embed_out and all_maxed_embed_out
    seq_len = cfg.data.seq_len
    patch_len = cfg.model.patch_len
    stride = cfg.model.stride
    patch_num = int((seq_len - patch_len)/stride + 1)
    output_results_dict = {
        "exp_num": cfg.exp_num,
        "cv_num": cfg.data.validation_cv_num,
        "model_name": cfg.model.model_name,
        "light_model_name": cfg.light_model.light_name,
        "data_name": cfg.data.data_name,
        "lr": cfg.light_model.optimizer.lr,
        "scheduler": cfg.light_model.scheduler.scheduler_type,
        "strong_permute_strategy": cfg.shuffler.strong_permute_strategy,
        "weak_permute_strategy": cfg.shuffler.weak_permute_strategy,
        "permutation_freq": cfg.shuffler.permute_freq,
        "pretrain_max_step": cfg.light_model.callbacks.max_steps, 
        "pretrain_max_epochs": cfg.light_model.callbacks.max_epochs,
        "pretrain_batch_size": cfg.light_model.dataset.batch_size,
        "pretrain_train_ratio": cfg.light_model.dataset.train_ratio,
        "patch_len": cfg.model.patch_len,
        "stride": cfg.model.stride,
        "patch_num": patch_num,
        "d_model": cfg.model.d_model,
    }
    for i in range(len_stages):
        output_results_dict[f"embed_{i}_acc"] = embed_dicts[i]["acc"]
        output_results_dict[f"embed_{i}_auroc"] = embed_dicts[i]["auroc"]
        output_results_dict[f"embed_{i}_auprc"] = embed_dicts[i]["auprc"]
        output_results_dict[f"embed_{i}_f1_score"] = embed_dicts[i]["f1_score"]
        output_results_dict[f"embed_{i}_precision_score"] = embed_dicts[i]["precision_score"]
        output_results_dict[f"embed_{i}_recall_score"] = embed_dicts[i]["recall_score"]

    output_results_dict[f"embed_average_acc"] = embed_dicts[-2]["acc"]
    output_results_dict[f"embed_average_auroc"] = embed_dicts[-2]["auroc"]
    output_results_dict[f"embed_average_auprc"] = embed_dicts[-2]["auprc"]
    output_results_dict[f"embed_average_f1_score"] = embed_dicts[-2]["f1_score"]
    output_results_dict[f"embed_average_precision_score"] = embed_dicts[-2]["precision_score"]
    output_results_dict[f"embed_average_recall_score"] = embed_dicts[-2]["recall_score"]

    output_results_dict[f"embed_max_acc"] = embed_dicts[-1]["acc"]
    output_results_dict[f"embed_max_auroc"] = embed_dicts[-1]["auroc"]
    output_results_dict[f"embed_max_auprc"] = embed_dicts[-1]["auprc"]
    output_results_dict[f"embed_max_f1_score"] = embed_dicts[-1]["f1_score"]
    output_results_dict[f"embed_max_precision_score"] = embed_dicts[-1]["precision_score"]
    output_results_dict[f"embed_max_recall_score"] = embed_dicts[-1]["recall_score"]

    output_results = pd.DataFrame(output_results_dict, index=[0])
    return output_results



def collect_permute_test_sets_dir(cfg, use_all_test_sets=False):
    """Collect the directories of permute test sets"""
    test_patch_size_list = cfg.permute_test.patch_size_sets
    permute_sets_dir = cfg.permute_test.permute_sets_dir
    data_dirs = []

    all_test_set_dirs = glob.glob(f"src/data/{permute_sets_dir}/" + "random_*")
    all_test_set_dirs.sort()
    # all_test_set_dirs = glob.glob(f"src/data/{permute_sets_dir}/" + "*_010_*.pkl")
    if use_all_test_sets:
        return all_test_set_dirs
    else:
        for test_set_dir in all_test_set_dirs:
            patch_size = int(test_set_dir.split("/")[-1].split("_")[2])
            if isinstance(test_patch_size_list, int):
                test_patch_size_list = [test_patch_size_list]
            if patch_size in test_patch_size_list:
                data_dirs.append(test_set_dir)
        return data_dirs

def eval_classification_with_permuted_test(cfg, test_repr, test_labels, clf):
    assert test_labels.ndim == 1 or test_labels.ndim == 2

    def merge_dim01(array):
        return array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])

    if test_labels.ndim == 2:
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    acc = clf.score(test_repr, test_labels)
    y_score = clf.predict_proba(test_repr)

    test_labels_onehot = label_binarize(test_labels, classes=np.arange(test_labels.max() + 1))

    if cfg.data.num_class == 2:
        auroc = roc_auc_score(test_labels_onehot.reshape(-1), y_score[:, 1])
        auprc = average_precision_score(test_labels_onehot.reshape(-1), y_score[:, 1])
        f1 = f1_score(test_labels, clf.predict(test_repr))
        precision = precision_score(test_labels, clf.predict(test_repr))
        recall = recall_score(test_labels, clf.predict(test_repr))

    else:
        auroc = roc_auc_score(test_labels_onehot, y_score, multi_class="ovo")
        auprc = average_precision_score(test_labels_onehot, y_score)
        f1 = f1_score(test_labels, clf.predict(test_repr), average="macro")
        precision = precision_score(test_labels, clf.predict(test_repr), average="macro")
        recall = recall_score(test_labels, clf.predict(test_repr), average="macro")
    print(
        f"acc: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, auroc: {auroc:.3f}, auprc: {auprc:.3f}"
    )
    return {
        "acc": acc,
        "auroc": auroc,
        "auprc": auprc,
        "f1_score": f1,
        "precision_score": precision,
        "recall_score": recall,
    }