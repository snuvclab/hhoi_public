CUDA_VISIBLE_DEVICES=0 python runners/inference_R_t_s.py \
--log_dir HHI_bench_demo \
--score_model_path results/ckpts/HHI_bench/ckpt_epoch18000.pth \
--sampler_mode ode \
--batch_size 256 \
--seed 0 \
--use_human_pose \
--human_pose_dim 10