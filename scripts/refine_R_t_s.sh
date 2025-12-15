CUDA_VISIBLE_DEVICES=0 python runners/refine_R_t_s.py \
--log_dir RtsNet_exp6 \
--score_model_path results/ckpts/RtsNet_exp6/ckpt_epoch20000.pth \
--input_path data/exp3/random_table_monitor.pkl \
--application_mode refinement \
--sampler_mode sdedit \
--seed 0 \
--T0 0.7 \
#--normalize_data \