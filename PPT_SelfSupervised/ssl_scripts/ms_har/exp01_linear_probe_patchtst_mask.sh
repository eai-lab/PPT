
#!/bin/bash
START_EXP_NUM=10
model_name=patchtst
data_name=ms_har
patch_size_sets=4
shuffler=random
permute_freq=40
lr_rate=0.001
batch_size=32
light_model=light_patchtst_pretrain_mask

d_model=256
max_steps=1000 
n_heads=4
d_ff=256


for max_steps in 1000 4000
do
    echo "Start experiment $START_EXP_NUM, patch_size_sets=$patch_size_sets"
    python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=0\
                    gpu_id=0 light_model=$light_model exp_num=$START_EXP_NUM \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    model.d_ff=$d_ff \
                    light_model.optimizer.lr=$lr_rate \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.ssl_loss.mask_ratio=0.4 \
                    light_model.callbacks.max_steps=$max_steps \
                    save_pt_file=True \
                    light_model.dataset.batch_size=$batch_size seed=2020 deterministic=True &
    sleep 3

    python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=1\
                    gpu_id=1 light_model=$light_model exp_num=$START_EXP_NUM \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    model.d_ff=$d_ff \
                    light_model.optimizer.lr=$lr_rate \
                    shuffler.permute_freq=$permute_freq \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.ssl_loss.mask_ratio=0.4 \
                    light_model.callbacks.max_steps=$max_steps \
                    save_pt_file=True \
                    light_model.dataset.batch_size=$batch_size seed=2021 deterministic=True &
    wait

    python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=2\
                    gpu_id=0 light_model=$light_model exp_num=$START_EXP_NUM \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    model.d_ff=$d_ff \
                    light_model.optimizer.lr=$lr_rate \
                    shuffler.permute_freq=$permute_freq \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.ssl_loss.mask_ratio=0.4 \
                    light_model.callbacks.max_steps=$max_steps \
                    save_pt_file=True \
                    light_model.dataset.batch_size=$batch_size seed=2022 deterministic=True &
    sleep 3

    python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=3\
                    gpu_id=1 light_model=$light_model exp_num=$START_EXP_NUM \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    model.d_ff=$d_ff \
                    light_model.optimizer.lr=$lr_rate \
                    shuffler.permute_freq=$permute_freq \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.ssl_loss.mask_ratio=0.4 \
                    light_model.callbacks.max_steps=$max_steps \
                    save_pt_file=True \
                    light_model.dataset.batch_size=$batch_size seed=2023 deterministic=True &

    wait
    python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=4\
                    gpu_id=1 light_model=$light_model exp_num=$START_EXP_NUM \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    model.d_ff=$d_ff \
                    light_model.optimizer.lr=$lr_rate \
                    shuffler.permute_freq=$permute_freq \
                    model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                    light_model.ssl_loss.mask_ratio=0.4 \
                    light_model.callbacks.max_steps=$max_steps \
                    save_pt_file=True \
                    light_model.dataset.batch_size=$batch_size seed=2024 deterministic=True &
    wait
    echo "finish $START_EXP_NUM"
    START_EXP_NUM=$(( START_EXP_NUM + 1 ))
done