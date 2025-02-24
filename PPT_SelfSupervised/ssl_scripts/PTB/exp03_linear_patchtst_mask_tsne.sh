
#!/bin/bash
START_EXP_NUM=501
model_name=patchtst
data_name=ptb
patch_size_sets=10
shuffler=random
lr_rate=0.002
batch_size=32
light_model=light_patchtst_pretrain_mask

d_model=64
max_steps=1000 
n_heads=4
d_ff=64


echo "Start experiment $START_EXP_NUM, patch_size_sets=$patch_size_sets"
python main_pretrain.py data=$data_name model=$model_name data.validation_cv_num=0\
                gpu_id=3 light_model=$light_model exp_num=$START_EXP_NUM \
                model.patch_len=$patch_size_sets model.stride=$patch_size_sets \
                model.d_ff=$d_ff \
                light_model.optimizer.lr=$lr_rate \
                model.pe=sincos model.d_model=$d_model model.n_heads=$n_heads model.n_layers=3 \
                light_model.ssl_loss.mask_ratio=0.2 \
                light_model.callbacks.max_steps=$max_steps \
                save_pt_file=True \
                save_embedding=True \
                light_model.dataset.batch_size=$batch_size seed=2020 deterministic=True &
sleep 3

