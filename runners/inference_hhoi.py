import os
import pickle
import numpy as np
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import get_config
from networks.posenet_agent_R_t_s import PoseNet
from utils.metrics import get_rot_matrix
from pytorch3d.transforms import *
import cv2
from train_human_pose_enc_dec import EncoderDecoder
from collections import defaultdict, deque
from itertools import combinations
import pytorch3d

def topological_sort(human_list, hhi_pair_list):
	# Build graph
	graph = defaultdict(list)
	in_degree = {human: 0 for human in human_list}

	for parent, child in hhi_pair_list:
		graph[parent].append(child)
		in_degree[child] += 1

	# Prepare original index map
	index_map = {human: idx for idx, human in enumerate(human_list)}

	# Start from nodes with in-degree 0 (roots)
	queue = deque([node for node in human_list if in_degree[node] == 0])
	result = []

	while queue:
		node = queue.popleft()
		result.append((node, index_map[node]))  # Add original index here
		for neighbor in graph[node]:
			in_degree[neighbor] -= 1
			if in_degree[neighbor] == 0:
				queue.append(neighbor)

	# Check for cycles or disconnected nodes
	if len(result) != len(human_list):
		raise ValueError("Cycle detected or disconnected nodes exist!")

	return result

def find_non_adjacent_pairs(human_list, hhi_pair_list):
    # Build undirected adjacency set
    adjacency = defaultdict(set)
    for a, b in hhi_pair_list:
        adjacency[a].add(b)
        adjacency[b].add(a)

    # Collect all 2-combinations of humans
    non_adjacent = []
    for u, v in combinations(human_list, 2):
        if v not in adjacency[u]:
            non_adjacent.append((u, v))

    return non_adjacent

cfg = get_config()

score_agent = PoseNet(cfg)
score_agent.load_ckpt_for_hhoi(hoi_model_path=cfg.hoi_score_model_path, hhi_model_path=cfg.hhi_score_model_path)

model = EncoderDecoder().to(cfg.device)
checkpoint = torch.load('enc_dec_human_pose/encoder_decoder_latest.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with open(cfg.input_pickle_path, "rb") as f:
	input_pickle = pickle.load(f)
	# len(human_list) == len(hoi_text_list)
	# len(hhi_pair_list) == len(hhi_text_list)
	human_list = input_pickle['human_list']
	hhi_pair_list = input_pickle['hhi_pair_list']
	hoi_text_list = input_pickle['hoi_text_list']
	hhi_text_list = input_pickle['hhi_text_list']
	
	sorted_human_list = topological_sort(human_list, hhi_pair_list)  # sorted list of (human, original index)
	human_non_adjacent_info = find_non_adjacent_pairs(human_list, hhi_pair_list)
 
	human_pose_dim = cfg.human_pose_dim
	hoi_pose_dim = 10 + human_pose_dim # 10 = rot(6) + transl(3) + scale(1)
	hhi_pose_dim = 9 + 2 * human_pose_dim # 9 = rot(6) + transl(3)

	num_hoi = len(human_list)
	num_hhi = len(hhi_pair_list)

hoi_text_list *= cfg.batch_size
hhi_text_list *= cfg.batch_size

res_list, _ = score_agent.inference_hhoi(batch_size=cfg.batch_size, human_list=human_list, sorted_human_list=sorted_human_list, hhi_pair_list=hhi_pair_list, human_non_adjacent_info=human_non_adjacent_info, hoi_text_prompt=hoi_text_list, hhi_text_prompt=hhi_text_list)

samples_dir = f"./results/inference/{cfg.log_dir}/{cfg.input_pickle_path.split('/')[-1].split('.')[0]}"
os.makedirs(samples_dir, exist_ok=True)

x = res_list[0]
full_shape = x.shape
human_final_rts_dict = {}
human_final_pose_dict = {}
for human, orig_idx in sorted_human_list:
	human_final_rts_dict[human] = []
	human_final_pose_dict[human] = []

	human_rts_pose = x[:, hoi_pose_dim*orig_idx:hoi_pose_dim*(orig_idx+1)] # (B, hoi_pose_dim) : R(6), t(3), s(1), pose(human_pose_dim)
	human_final_rts_dict[human].append(human_rts_pose[:, :10])
	human_final_pose_dict[human].append(human_rts_pose[:, 10:])

	for hhi_pair_idx, (base_human, target_human) in enumerate(hhi_pair_list):
		if human == target_human:
			h2_rel_rt_pose = x[:, full_shape[1]-hhi_pose_dim*(num_hhi-hhi_pair_idx)+human_pose_dim:full_shape[1]-hhi_pose_dim*(num_hhi-1-hhi_pair_idx)] # (B, 9+human_pose_dim) : R(6), t(3), pose(human_pose_dim)
			R = get_rot_matrix(h2_rel_rt_pose[:, :6], cfg.pose_mode).permute(0, 2, 1)
			t = h2_rel_rt_pose[:, 6:9]
			pose = h2_rel_rt_pose[:, 9:]
			
			R1 = get_rot_matrix(human_final_rts_dict[base_human][:, :6], cfg.pose_mode).permute(0, 2, 1)
			R_ = pytorch3d.transforms.matrix_to_rotation_6d(torch.matmul(R1, R)) # (B, 6)
			t_ = human_final_rts_dict[base_human][:, 9:10] * (R1 @ t.unsqueeze(-1)).squeeze(-1) + human_final_rts_dict[base_human][:, 6:9] # (B, 3): s1R1@t + t1

			human_rts_ = torch.cat([R_, t_, human_rts_pose[:, 9:10]], dim=-1) # (B, 10)
			human_final_rts_dict[human].append(human_rts_)
			human_final_pose_dict[human].append(pose)
		elif human == base_human:
			pose = x[:, full_shape[1]-hhi_pose_dim*(num_hhi-hhi_pair_idx):full_shape[1]-hhi_pose_dim*(num_hhi-hhi_pair_idx)+human_pose_dim] # (B, human_pose_dim) : pose(human_pose_dim)
			human_final_pose_dict[human].append(pose)
		else:
			pass
			
	human_final_rts_dict[human] = torch.stack(human_final_rts_dict[human], dim=0).mean(dim=0) # (B, 10) : R2(6), t2(3), s2(1)
	human_final_pose_dict[human] = torch.stack(human_final_pose_dict[human], dim=0).mean(dim=0) # (B, human_pose_dim) : pose(human_pose_dim)

final = []
for human in human_list:
	human_rots = human_final_rts_dict[human][:, :6]
	human_transls = human_final_rts_dict[human][:, 6:9]
	human_scales = human_final_rts_dict[human][:, 9]
	human_pose = human_final_pose_dict[human] # (B, human_pose_dim) : pose(human_pose_dim)
	
	human_rot_mats = rotation_6d_to_matrix(human_rots.cpu()).numpy()			# (B, 3, 3)
	
	human_rot = np.array([cv2.Rodrigues(R)[0].flatten() for R in human_rot_mats])
	human_scale = human_scales.cpu().numpy().reshape((-1, 1))
	human_transl = human_transls.cpu().numpy()

	with torch.no_grad():
		human_pose_dec = model.decoder(human_pose.float())
		human_pose_list = []
		for i in range(21):
			nx_pose = human_pose_dec[:, 6 * i : 6 * i + 6]
			nx_mat = rotation_6d_to_matrix(nx_pose.cpu()).numpy()
			nx_mats = np.array([cv2.Rodrigues(R)[0].flatten() for R in nx_mat])
			human_pose_list.append(nx_mats)
		
		human_pose = np.concatenate(human_pose_list, axis=-1)
	final_human = np.concatenate([human_rot, human_transl, human_scale, human_pose], axis=-1)
	final.append(final_human)

final = np.stack(final, axis=1)
np.savez(f'{samples_dir}/inference.npz', transform=final, model_type=cfg.model_type)