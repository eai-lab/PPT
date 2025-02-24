
#!/bin/bash
START_EXP_NUM=202
model_name=pits
data_name=gilon_activity
patch_size_sets=10
shuffler=random
permute_freq=40
lr_rate=0.002
batch_size=32
light_model=light_pits_pretrain_mask
use_awl_loss=True
consistency_loss=True
margin_loss=True

d_model=128
max_steps=4000 

echo "Start experiment $START_EXP_NUM, patch_size_sets=$patch_size_sets"
python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=0\
                gpu_id=0 light_model=$light_model exp_num=$START_EXP_NUM \
                model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                light_model.optimizer.lr=$lr_rate \
                model.d_model=$d_model\
                light_model.ssl_loss.mask_ratio=0.4 \
                light_model.callbacks.max_steps=$max_steps \
                save_pt_file=True \
                light_model.dataset.batch_size=$batch_size seed=2020 deterministic=True &
sleep 3

python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=1\
                gpu_id=1 light_model=$light_model exp_num=$START_EXP_NUM \
                model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                light_model.optimizer.lr=$lr_rate \
                shuffler.permute_freq=$permute_freq \
                model.d_model=$d_model\
                light_model.ssl_loss.mask_ratio=0.4 \
                light_model.callbacks.max_steps=$max_steps \
                save_pt_file=True \
                light_model.dataset.batch_size=$batch_size seed=2021 deterministic=True &
wait

python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=2\
                gpu_id=0 light_model=$light_model exp_num=$START_EXP_NUM \
                model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                light_model.optimizer.lr=$lr_rate \
                shuffler.permute_freq=$permute_freq \
                model.d_model=$d_model\
                light_model.ssl_loss.mask_ratio=0.4 \
                light_model.callbacks.max_steps=$max_steps \
                save_pt_file=True \
                light_model.dataset.batch_size=$batch_size seed=2022 deterministic=True &
sleep 3

python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=3\
                gpu_id=1 light_model=$light_model exp_num=$START_EXP_NUM \
                model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                light_model.optimizer.lr=$lr_rate \
                shuffler.permute_freq=$permute_freq \
                model.d_model=$d_model\
                light_model.ssl_loss.mask_ratio=0.4 \
                light_model.callbacks.max_steps=$max_steps \
                save_pt_file=True \
                light_model.dataset.batch_size=$batch_size seed=2023 deterministic=True &

wait
python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=4\
                gpu_id=1 light_model=$light_model exp_num=$START_EXP_NUM \
                model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                light_model.optimizer.lr=$lr_rate \
                shuffler.permute_freq=$permute_freq \
                model.d_model=$d_model\
                light_model.ssl_loss.mask_ratio=0.4 \
                light_model.callbacks.max_steps=$max_steps \
                save_pt_file=True \
                light_model.dataset.batch_size=$batch_size seed=2024 deterministic=True &