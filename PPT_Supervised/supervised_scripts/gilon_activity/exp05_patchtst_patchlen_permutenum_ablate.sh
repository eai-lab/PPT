#!/bin/bash
START_EXP_NUM=1200
model_name=patchtst
patch_size_sets=5
shuffler=random
permute_freq=10
lr_rate=0.002
batch_size=128
light_model=light_patchtst_permute
use_awl_loss=True

loss=True

for patch_size_sets in 5 8 16 20
do
    for permute_freq in 5 10 20 40
    do
        echo "Start experiment $START_EXP_NUM, patch_size_sets=$patch_size_sets"
        python main_supervised.py data=gilon_activity model=$model_name data.validation_cv_num=0\
                        gpu_id=0 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        permute_test.patch_size_sets=[$patch_size_sets] \
                        permute_test.perform_test_on_permute_sets=True \
                        shuffler.permute_freq=$permute_freq \
                        model.pe=sincos model.d_model=128 model.n_heads=4 model.n_layers=3 \
                        light_model.ssl_loss.use_awl_loss=$use_awl_loss \
                        light_model.ssl_loss.use_consistency_loss=$loss  \
                        light_model.ssl_loss.use_margin_loss=$loss  \
                        save_pt_file=True \
                        light_model.dataset.batch_size=$batch_size seed=2020 deterministic=True &
        wait
        python main_supervised.py data=gilon_activity model=$model_name data.validation_cv_num=1\
                        gpu_id=0 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        permute_test.patch_size_sets=[$patch_size_sets] \
                        permute_test.perform_test_on_permute_sets=True \
                        shuffler.permute_freq=$permute_freq  \
                        model.pe=sincos model.d_model=128 model.n_heads=4 model.n_layers=3 \
                        light_model.ssl_loss.use_awl_loss=$use_awl_loss \
                        light_model.ssl_loss.use_consistency_loss=$loss  \
                        light_model.ssl_loss.use_margin_loss=$loss  \
                        save_pt_file=True \
                        light_model.dataset.batch_size=$batch_size seed=2021 deterministic=True &
        wait
        python main_supervised.py data=gilon_activity model=$model_name data.validation_cv_num=2\
                        gpu_id=0 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        permute_test.patch_size_sets=[$patch_size_sets] \
                        permute_test.perform_test_on_permute_sets=True \
                        shuffler.permute_freq=$permute_freq  \
                        model.pe=sincos model.d_model=128 model.n_heads=4 model.n_layers=3 \
                        light_model.ssl_loss.use_awl_loss=$use_awl_loss \
                        light_model.ssl_loss.use_consistency_loss=$loss  \
                        light_model.ssl_loss.use_margin_loss=$loss  \
                        save_pt_file=True \
                        light_model.dataset.batch_size=$batch_size seed=2022 deterministic=True &

        wait
        python main_supervised.py data=gilon_activity model=$model_name data.validation_cv_num=3\
                        gpu_id=0 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        permute_test.patch_size_sets=[$patch_size_sets] \
                        permute_test.perform_test_on_permute_sets=True \
                        shuffler.permute_freq=$permute_freq  \
                        model.pe=sincos model.d_model=128 model.n_heads=4 model.n_layers=3 \
                        light_model.ssl_loss.use_awl_loss=$use_awl_loss \
                        light_model.ssl_loss.use_consistency_loss=$loss  \
                        light_model.ssl_loss.use_margin_loss=$loss  \
                        save_pt_file=True \
                        light_model.dataset.batch_size=$batch_size seed=2023 deterministic=True &

        wait
        python main_supervised.py data=gilon_activity model=$model_name data.validation_cv_num=4\
                        gpu_id=0 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        permute_test.patch_size_sets=[$patch_size_sets] \
                        permute_test.perform_test_on_permute_sets=True \
                        shuffler.permute_freq=$permute_freq  \
                        model.pe=sincos model.d_model=128 model.n_heads=4 model.n_layers=3 \
                        light_model.ssl_loss.use_awl_loss=$use_awl_loss \
                        light_model.ssl_loss.use_consistency_loss=$loss  \
                        light_model.ssl_loss.use_margin_loss=$loss  \
                        save_pt_file=True \
                        light_model.dataset.batch_size=$batch_size seed=2024 deterministic=True &
        wait
        echo "finish $START_EXP_NUM"
        START_EXP_NUM=$(( START_EXP_NUM + 1 ))
    done
done

