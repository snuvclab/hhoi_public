CUDA_VISIBLE_DEVICES=3 python runners/trainer_R_t_s.py \
--data_path data \
--log_dir RtsNet_canoe_paddle \
--agent_type score \
--sampler_mode ode \
--sampling_steps 500 \
--eval_freq 1000 \
--n_epochs 20000 \
--batch_size 500 \
--seed 0 \
--is_train \
--lr 5e-3 \
--lr_decay 0.99 \
#--normalize_data \
