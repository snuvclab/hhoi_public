CUDA_VISIBLE_DEVICES=7 python runners/inference_R_t_s.py \
--log_dir HOI \
--score_model_path results/ckpts/HOI/ckpt_epoch20000.pth \
--input_text_prompt "A person is sitting on a bench." \
--sampler_mode ode \
--batch_size 256 \
--seed 0 \
--human_pose_dim 10 \
--model_type hoi