#!/bin/bash
START_EXP_NUM=0
model_name=patchtst
data_name=gilon_activity
patch_size_sets=5
shuffler=random
permute_freq=40
lr_rate=0.002
batch_size=32
light_model=light_patchtst_pretrain_permute
use_awl_loss=True
consistency_loss=True
margin_loss=True


for patch_size_sets in 5 10
do
    for max_steps in 2000
    do 
        for d_model in 64 128 256
        do
            for n_heads in 4 8
            do
                    echo "Start experiment $START_EXP_NUM, patch_size_sets=$patch_size_sets"
                    python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=0\
                                    gpu_id=0 light_model=$light_model exp_num=$START_EXP_NUM \
                                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                                    light_model.optimizer.lr=$lr_rate \
                                    shuffler.permute_freq=$permute_freq \
                                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                                    light_model.ssl_loss.use_awl_loss=True \
                                    light_model.ssl_loss.use_consistency_loss=True  \
                                    light_model.ssl_loss.use_margin_loss=True  \
                                    light_model.callbacks.max_steps=$max_steps \
                                    save_pt_file=False \
                                    light_model.dataset.batch_size=$batch_size seed=2020 deterministic=True &
                    sleep 3
                    python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=1\
                                    gpu_id=1 light_model=$light_model exp_num=$START_EXP_NUM \
                                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                                    light_model.optimizer.lr=$lr_rate \
                                    shuffler.permute_freq=$permute_freq \
                                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                                    light_model.ssl_loss.use_awl_loss=True \
                                    light_model.ssl_loss.use_consistency_loss=True  \
                                    light_model.ssl_loss.use_margin_loss=True  \
                                    light_model.callbacks.max_steps=$max_steps \
                                    save_pt_file=False \
                                    light_model.dataset.batch_size=$batch_size seed=2021 deterministic=True &
                    wait
                    echo "finish $START_EXP_NUM"
                    START_EXP_NUM=$(( START_EXP_NUM + 1 ))
            done 
        done 
    done
done

