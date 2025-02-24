#!/bin/bash
START_EXP_NUM=20
pretrain_exp_num=200 # Loading the pretrained exp_num
model_name=patchtst
data_name=gilon_activity
patch_size_sets=5
shuffler=random
lr_rate=0.0002
batch_size=128
patience=5
light_model=light_patchtst_fullfinetune

d_model=128
n_heads=4


for train_ratio in 1.0 0.1 0.01
do
    echo "Start experiment $START_EXP_NUM, patch_size_sets=$patch_size_sets"
    python main_finetune.py data=$data_name model=$model_name data.validation_cv_num=0\
                    gpu_id=0 light_model=$light_model  \
                    exp_num=$START_EXP_NUM pretrain_exp_num=$pretrain_exp_num \
                    task.train_ratio=$train_ratio \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    light_model.optimizer.lr=$lr_rate \
                    pretrain_pt_file=cv0_pretrain.pt \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.callbacks.patience=$patience \
                    light_model.dataset.batch_size=$batch_size seed=2020 deterministic=True &
    sleep 3
    python main_finetune.py data=$data_name model=$model_name data.validation_cv_num=1\
                    gpu_id=1 light_model=$light_model  \
                    exp_num=$START_EXP_NUM pretrain_exp_num=$pretrain_exp_num \
                    task.train_ratio=$train_ratio \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    light_model.optimizer.lr=$lr_rate \
                    pretrain_pt_file=cv1_pretrain.pt \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.callbacks.patience=$patience \
                    light_model.dataset.batch_size=$batch_size seed=2021 deterministic=True &
    sleep 3
    python main_finetune.py data=$data_name model=$model_name data.validation_cv_num=2\
                    gpu_id=0 light_model=$light_model  \
                    exp_num=$START_EXP_NUM pretrain_exp_num=$pretrain_exp_num \
                    task.train_ratio=$train_ratio \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    light_model.optimizer.lr=$lr_rate \
                    pretrain_pt_file=cv2_pretrain.pt \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.callbacks.patience=$patience \
                    light_model.dataset.batch_size=$batch_size seed=2022 deterministic=True &
    sleep 3
    python main_finetune.py data=$data_name model=$model_name data.validation_cv_num=3\
                    gpu_id=1 light_model=$light_model  \
                    exp_num=$START_EXP_NUM pretrain_exp_num=$pretrain_exp_num \
                    task.train_ratio=$train_ratio \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    light_model.optimizer.lr=$lr_rate \
                    pretrain_pt_file=cv3_pretrain.pt \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.callbacks.patience=$patience \
                    light_model.dataset.batch_size=$batch_size seed=2023 deterministic=True &
    sleep 3
    python main_finetune.py data=$data_name model=$model_name data.validation_cv_num=4\
                    gpu_id=0 light_model=$light_model  \
                    exp_num=$START_EXP_NUM pretrain_exp_num=$pretrain_exp_num \
                    task.train_ratio=$train_ratio \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    light_model.optimizer.lr=$lr_rate \
                    pretrain_pt_file=cv4_pretrain.pt \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.callbacks.patience=$patience \
                    light_model.dataset.batch_size=$batch_size seed=2024 deterministic=True &
    wait
    echo "finish $START_EXP_NUM"
    START_EXP_NUM=$(( START_EXP_NUM + 1 ))
done


for train_ratio in 1.0 0.1 0.01
do
    echo "Start experiment $START_EXP_NUM, patch_size_sets=$patch_size_sets"
    python main_finetune.py data=$data_name model=$model_name data.validation_cv_num=0\
                    gpu_id=0 light_model=$light_model  \
                    exp_num=$START_EXP_NUM pretrain_exp_num=$pretrain_exp_num \
                    task.train_ratio=$train_ratio \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    light_model.optimizer.lr=$lr_rate \
                    pretrain_pt_file=null \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.callbacks.patience=$patience \
                    light_model.dataset.batch_size=$batch_size seed=2020 deterministic=True &
    sleep 3
    python main_finetune.py data=$data_name model=$model_name data.validation_cv_num=1\
                    gpu_id=1 light_model=$light_model  \
                    exp_num=$START_EXP_NUM pretrain_exp_num=$pretrain_exp_num \
                    task.train_ratio=$train_ratio \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    light_model.optimizer.lr=$lr_rate \
                    pretrain_pt_file=null \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.callbacks.patience=$patience \
                    light_model.dataset.batch_size=$batch_size seed=2021 deterministic=True &
    sleep 3
    python main_finetune.py data=$data_name model=$model_name data.validation_cv_num=2\
                    gpu_id=0 light_model=$light_model  \
                    exp_num=$START_EXP_NUM pretrain_exp_num=$pretrain_exp_num \
                    task.train_ratio=$train_ratio \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    light_model.optimizer.lr=$lr_rate \
                    pretrain_pt_file=null \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.callbacks.patience=$patience \
                    light_model.dataset.batch_size=$batch_size seed=2022 deterministic=True &
    sleep 3
    python main_finetune.py data=$data_name model=$model_name data.validation_cv_num=3\
                    gpu_id=1 light_model=$light_model  \
                    exp_num=$START_EXP_NUM pretrain_exp_num=$pretrain_exp_num \
                    task.train_ratio=$train_ratio \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    light_model.optimizer.lr=$lr_rate \
                    pretrain_pt_file=null \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.callbacks.patience=$patience \
                    light_model.dataset.batch_size=$batch_size seed=2023 deterministic=True &
    sleep 3
    python main_finetune.py data=$data_name model=$model_name data.validation_cv_num=4\
                    gpu_id=0 light_model=$light_model  \
                    exp_num=$START_EXP_NUM pretrain_exp_num=$pretrain_exp_num \
                    task.train_ratio=$train_ratio \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    light_model.optimizer.lr=$lr_rate \
                    pretrain_pt_file=null \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.callbacks.patience=$patience \
                    light_model.dataset.batch_size=$batch_size seed=2024 deterministic=True &
    wait
    echo "finish $START_EXP_NUM"
    START_EXP_NUM=$(( START_EXP_NUM + 1 ))
done
