import os
import pickle
import numpy as np
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import get_config
from networks.posenet_agent_R_t_s import PoseNet
from utils.metrics import get_rot_matrix

cfg = get_config()

score_agent = PoseNet(cfg)
score_agent.load_ckpt(model_dir=cfg.score_model_path, model_path=True, load_model_only=True) 

if cfg.normalize_data:
    mean_std_file_path = "data/mean_std_teapot_teacup.pkl"
    with open(mean_std_file_path, 'rb') as f:
        mean_std = pickle.load(f)
        mean = mean_std['mean']
        std = mean_std['std']
        
        #mean = mean.permute(0, 2, 1).reshape(mean.shape[0], -1)
        #std = std.permute(0, 2, 1).reshape(std.shape[0], -1)
        
        mean = mean.transpose(1, 0).reshape(-1)
        std = std.transpose(1, 0).reshape(-1)
        
        base_rot_mean = mean[0:6]
        base_rot_std = std[0:6]
        base_trans_mean = mean[9:12]
        base_trans_std = std[9:12]
        base_scale_mean = mean[12:15]
        base_scale_std = std[12:15]
        target_rot_mean = mean[15:21]
        target_rot_std = std[15:21]
        target_trans_mean = mean[24:27]
        target_trans_std  = std[24:27]
        target_scale_mean = mean[27:30]
        target_scale_std = std[27:30]
        
        mean = np.concatenate((base_rot_mean, base_trans_mean, base_scale_mean, target_rot_mean, target_trans_mean, target_scale_mean))
        std = np.concatenate((base_rot_std, base_trans_std, base_scale_std, target_rot_std, target_trans_std, target_scale_std))

with open(cfg.input_path, 'rb') as f:
    random_samples = pickle.load(f)
res_list, sampler_mode_list = score_agent.refinement_Rts(random_samples)

samples_dir = f"./results/refinement/{cfg.log_dir}"
os.makedirs(samples_dir, exist_ok=True)
pickle_name = cfg.input_path.split("/")[-1]
with open(f'{samples_dir}/{pickle_name}', 'wb') as f:
    pickle.dump(random_samples, f)

samples = {}
for i, sampler_mode in enumerate(sampler_mode_list):
    if cfg.normalize_data:
        res_normal = (res_list[i] * torch.from_numpy(std).to(res_list[i].device)) + torch.from_numpy(mean).to(res_list[i].device)
        
        base_rot_mat = get_rot_matrix(res_normal[:, :6], cfg.pose_mode).cpu().numpy()
        base_trans = res_normal[:, 6:9].cpu().numpy()
        base_scale = res_normal[:, 9:12].cpu().numpy()
        
        target_rot_mat = get_rot_matrix(res_normal[:, 12:18], cfg.pose_mode).cpu().numpy()
        target_trans = res_normal[:, 18:21].cpu().numpy()
        target_scale = res_normal[:, 21:].cpu().numpy()
    else:
        base_rot_mat = get_rot_matrix(res_list[i][:, :6], cfg.pose_mode).cpu().numpy()
        base_trans = res_list[i][:, 6:9].cpu().numpy()
        base_scale = res_list[i][:, 9:12].cpu().numpy()
        
        target_rot_mat = get_rot_matrix(res_list[i][:, 12:18], cfg.pose_mode).cpu().numpy()
        target_trans = res_list[i][:, 18:21].cpu().numpy()
        target_scale = res_list[i][:, 21:].cpu().numpy()
    
    samples[sampler_mode] = {"base_R": base_rot_mat, "base_t": base_trans, "base_s": base_scale, "target_R": target_rot_mat, "target_t": target_trans, "target_s": target_scale}


with open(f'{samples_dir}/refinement_T_{cfg.T0}.pkl', 'wb') as f:
    pickle.dump(samples, f)