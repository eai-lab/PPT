
#!/bin/bash
START_EXP_NUM=1001
model_name=pits
data_name=ms_har
patch_size_sets=4
shuffler=random
permute_freq=40
lr_rate=0.002
batch_size=32
light_model=light_pits_pretrain_mask
use_awl_loss=True
consistency_loss=True
margin_loss=True

d_model=512  
max_steps=1000 

for mask_ratio in 0.4 0.6
do
    echo "Start experiment $START_EXP_NUM, patch_size_sets=$patch_size_sets"
    python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=0\
                    gpu_id=0 light_model=$light_model exp_num=$START_EXP_NUM \
                    model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                    light_model.optimizer.lr=$lr_rate \
                    model.d_model=$d_model\
                    light_model.ssl_loss.mask_ratio=$mask_ratio \
                    light_model.callbacks.max_steps=$max_steps \
                    save_pt_file=False \
                    save_embedding=True \
                    light_model.dataset.batch_size=$batch_size seed=2020 deterministic=True &
    wait
    echo "finish $START_EXP_NUM"
    START_EXP_NUM=$(( START_EXP_NUM + 1 ))
done