
#!/bin/bash
START_EXP_NUM=600
model_name=pits
data_name=ptb
patch_size_sets=6
shuffler=random
permute_freq=40
lr_rate=0.01
batch_size=32
light_model=light_pits_pretrain_permute
use_awl_loss=True
consistency_loss=True
margin_loss=True

d_model=512  
max_steps=4000

for patch_size_sets in 5 # 10 15 20
do
    for permute_freq in 5 10 20 40
    do
        echo "Start experiment $START_EXP_NUM, patch_size_sets=$patch_size_sets"
        python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=0\
                        gpu_id=2 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        shuffler.permute_freq=$permute_freq \
                        model.d_model=$d_model \
                        light_model.ssl_loss.use_awl_loss=True \
                        light_model.ssl_loss.use_consistency_loss=True  \
                        light_model.ssl_loss.use_margin_loss=True  \
                        light_model.callbacks.max_steps=$max_steps \
                        save_pt_file=False \
                        light_model.dataset.batch_size=$batch_size seed=2020 deterministic=True &
        sleep 3

        python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=1\
                        gpu_id=3 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        shuffler.permute_freq=$permute_freq \
                        model.d_model=$d_model \
                        light_model.ssl_loss.use_awl_loss=True \
                        light_model.ssl_loss.use_consistency_loss=True  \
                        light_model.ssl_loss.use_margin_loss=True  \
                        light_model.callbacks.max_steps=$max_steps \
                        save_pt_file=False \
                        light_model.dataset.batch_size=$batch_size seed=2021 deterministic=True &
        sleep 3

        python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=2\
                        gpu_id=2 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        shuffler.permute_freq=$permute_freq \
                        model.d_model=$d_model \
                        light_model.ssl_loss.use_awl_loss=True \
                        light_model.ssl_loss.use_consistency_loss=True  \
                        light_model.ssl_loss.use_margin_loss=True  \
                        light_model.callbacks.max_steps=$max_steps \
                        save_pt_file=False \
                        light_model.dataset.batch_size=$batch_size seed=2022 deterministic=True &
        sleep 3

        python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=3\
                        gpu_id=3 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        shuffler.permute_freq=$permute_freq \
                        model.d_model=$d_model \
                        light_model.ssl_loss.use_awl_loss=True \
                        light_model.ssl_loss.use_consistency_loss=True  \
                        light_model.ssl_loss.use_margin_loss=True  \
                        light_model.callbacks.max_steps=$max_steps \
                        save_pt_file=False \
                        light_model.dataset.batch_size=$batch_size seed=2023 deterministic=True &

        wait
        python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=4\
                        gpu_id=2 light_model=$light_model exp_num=$START_EXP_NUM \
                        model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                        light_model.optimizer.lr=$lr_rate \
                        shuffler.permute_freq=$permute_freq \
                        model.d_model=$d_model \
                        light_model.ssl_loss.use_awl_loss=True \
                        light_model.ssl_loss.use_consistency_loss=True  \
                        light_model.ssl_loss.use_margin_loss=True  \
                        light_model.callbacks.max_steps=$max_steps \
                        save_pt_file=False \
                        light_model.dataset.batch_size=$batch_size seed=2024 deterministic=True &
        wait
        echo "finish $START_EXP_NUM"
        START_EXP_NUM=$(( START_EXP_NUM + 1 ))
    done
done