#!/bin/bash
START_EXP_NUM=700
model_name=patchtst
patch_size_sets=10
shuffler=random
permute_freq=40
lr_rate=0.002
batch_size=128
light_model=light_patchtst_permute
use_awl_loss=True
data_name=ptb
n_heads=8
d_model=128
loss=True

for strong_strategy in vicinity farthest random
do
    for weak_strategy in vicinity farthest random
    do 
        echo "Start experiment $START_EXP_NUM, patch_size_sets=$patch_size_sets"
        python main_supervised.py data=$data_name model=$model_name data.validation_cv_num=0\
                        gpu_id=0 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        permute_test.patch_size_sets=[$patch_size_sets] \
                        permute_test.perform_test_on_permute_sets=True \
                        shuffler.permute_freq=$permute_freq \
                        shuffler.strong_permute_strategy=$strong_strategy \
                        shuffler.weak_permute_strategy=$weak_strategy \
                        model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                        task.limit_train_batches=1.0 \
                        light_model.ssl_loss.use_awl_loss=$use_awl_loss \
                        light_model.ssl_loss.use_consistency_loss=$loss  \
                        light_model.ssl_loss.use_margin_loss=$loss  \
                        light_model.callbacks.patience=3 \
                        save_pt_file=True \
                        light_model.dataset.batch_size=$batch_size seed=2020 deterministic=True &
        sleep 3
        python main_supervised.py data=$data_name model=$model_name data.validation_cv_num=1\
                        gpu_id=1 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        permute_test.patch_size_sets=[$patch_size_sets] \
                        permute_test.perform_test_on_permute_sets=True \
                        shuffler.permute_freq=$permute_freq  \
                        shuffler.strong_permute_strategy=$strong_strategy \
                        shuffler.weak_permute_strategy=$weak_strategy \
                        model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                        task.limit_train_batches=1.0 \
                        light_model.ssl_loss.use_awl_loss=$use_awl_loss \
                        light_model.ssl_loss.use_consistency_loss=$loss  \
                        light_model.ssl_loss.use_margin_loss=$loss  \
                        light_model.callbacks.patience=3 \
                        save_pt_file=True \
                        light_model.dataset.batch_size=$batch_size seed=2021 deterministic=True &
        sleep 3
        python main_supervised.py data=$data_name model=$model_name data.validation_cv_num=2\
                        gpu_id=2 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        permute_test.patch_size_sets=[$patch_size_sets] \
                        permute_test.perform_test_on_permute_sets=True \
                        shuffler.permute_freq=$permute_freq  \
                        shuffler.strong_permute_strategy=$strong_strategy \
                        shuffler.weak_permute_strategy=$weak_strategy \
                        model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                        light_model.ssl_loss.use_awl_loss=$use_awl_loss \
                        light_model.ssl_loss.use_consistency_loss=$loss  \
                        light_model.ssl_loss.use_margin_loss=$loss  \
                        light_model.callbacks.patience=3 \
                        save_pt_file=True \
                        light_model.dataset.batch_size=$batch_size seed=2022 deterministic=True &

        sleep 3
        python main_supervised.py data=$data_name model=$model_name data.validation_cv_num=3\
                        gpu_id=3 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        permute_test.patch_size_sets=[$patch_size_sets] \
                        permute_test.perform_test_on_permute_sets=True \
                        shuffler.permute_freq=$permute_freq  \
                        shuffler.strong_permute_strategy=$strong_strategy \
                        shuffler.weak_permute_strategy=$weak_strategy \
                        model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                        light_model.ssl_loss.use_awl_loss=$use_awl_loss \
                        light_model.ssl_loss.use_consistency_loss=$loss  \
                        light_model.ssl_loss.use_margin_loss=$loss  \
                        light_model.callbacks.patience=3 \
                        save_pt_file=True \
                        light_model.dataset.batch_size=$batch_size seed=2023 deterministic=True &

        wait
        python main_supervised.py data=$data_name model=$model_name data.validation_cv_num=4\
                        gpu_id=1 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        permute_test.patch_size_sets=[$patch_size_sets] \
                        permute_test.perform_test_on_permute_sets=True \
                        shuffler.permute_freq=$permute_freq  \
                        shuffler.strong_permute_strategy=$strong_strategy \
                        shuffler.weak_permute_strategy=$weak_strategy \
                        model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                        light_model.ssl_loss.use_awl_loss=$use_awl_loss \
                        light_model.ssl_loss.use_consistency_loss=$loss  \
                        light_model.ssl_loss.use_margin_loss=$loss  \
                        light_model.callbacks.patience=3 \
                        save_pt_file=True \
                        light_model.dataset.batch_size=$batch_size seed=2024 deterministic=True &
        wait
        echo "finish $START_EXP_NUM"
        START_EXP_NUM=$(( START_EXP_NUM + 1 ))
    done
done
