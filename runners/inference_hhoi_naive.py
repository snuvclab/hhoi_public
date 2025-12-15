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

cfg = get_config()

score_agent = PoseNet(cfg)
score_agent.load_ckpt_for_hhoi(hoi_model_path=cfg.hoi_score_model_path, hhi_model_path=cfg.hhi_score_model_path)

model = EncoderDecoder().to(cfg.device)
checkpoint = torch.load('enc_dec_human_pose/encoder_decoder_latest.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


num_repetitions = cfg.batch_size
prompt_h1o = cfg.input_h1o_text_prompt
prompt_h1h2 = cfg.input_h1h2_text_prompt

h1o_text_input = []
h1h2_text_input = []
for i in range(num_repetitions):
    h1o_text_input.append(prompt_h1o)
    h1h2_text_input.append(prompt_h1h2)

hoi_x_list, hhi_x_list = score_agent.inference_hhoi_naive(batch_size=num_repetitions, h1o_text_prompt=h1o_text_input, h1h2_text_prompt=h1h2_text_input)

samples_dir = f"./results/inference/{cfg.log_dir}"
os.makedirs(samples_dir, exist_ok=True)
samples = {}

hoi_rots = hoi_x_list[0][:, :6]
hoi_transls = hoi_x_list[0][:, 6:9]
hoi_scales = hoi_x_list[0][:, 9]
hoi_poses = hoi_x_list[0][:, 10:]

hhi_h1_poses = hhi_x_list[0][:, :10]
hhi_h2_rots = hhi_x_list[0][:, 10:16]
hhi_h2_transl = hhi_x_list[0][:, 16:19]
hhi_h2_poses = hhi_x_list[0][:, 19:]

hoi_rot_mats = rotation_6d_to_matrix(hoi_rots.cpu()).numpy()			# (B, 3, 3)
hhi_h2_rot_mats = rotation_6d_to_matrix(hhi_h2_rots.cpu()).numpy()		# (B, 3, 3)


h2_rot_mats = hoi_rot_mats @ hhi_h2_rot_mats																			# (B, 3, 3)
h2_rot = np.array([cv2.Rodrigues(R)[0].flatten() for R in h2_rot_mats])													# (B, 3)
h2_scale = hoi_scales.cpu().numpy().reshape((-1, 1))																	# (B, 1)
h2_transl = h2_scale * np.einsum('bij,bj->bi', hoi_rot_mats, hhi_h2_transl.cpu().numpy()) + hoi_transls.cpu().numpy()	# (B, 3)
with torch.no_grad():
	h2_pose_dec = model.decoder(hhi_h2_poses.float())																	# (B, 126)
	h2_pose_list = []
	for i in range(21):
		nx_pose = h2_pose_dec[:, 6 * i : 6 * i + 6]
		nx_mat = rotation_6d_to_matrix(nx_pose.cpu()).numpy()
		nx_mats = np.array([cv2.Rodrigues(R)[0].flatten() for R in nx_mat])
		h2_pose_list.append(nx_mats)
	
	h2_pose = np.concatenate(h2_pose_list, axis=-1)																		# (B, 63)
final_h2 = np.concatenate([h2_rot, h2_transl, h2_scale, h2_pose], axis=-1)

h1_rot = np.array([cv2.Rodrigues(R)[0].flatten() for R in hoi_rot_mats])
h1_scale = hoi_scales.cpu().numpy().reshape((-1, 1))
h1_transl = hoi_transls.cpu().numpy()
with torch.no_grad():
	h1_pose_dec = model.decoder(hoi_poses.float())
	h1_pose_list = []
	for i in range(21):
		nx_pose = h1_pose_dec[:, 6 * i : 6 * i + 6]
		nx_mat = rotation_6d_to_matrix(nx_pose.cpu()).numpy()
		nx_mats = np.array([cv2.Rodrigues(R)[0].flatten() for R in nx_mat])
		h1_pose_list.append(nx_mats)
	
	h1_pose = np.concatenate(h1_pose_list, axis=-1)
final_h1 = np.concatenate([h1_rot, h1_transl, h1_scale, h1_pose], axis=-1)


final = np.stack([final_h1, final_h2], axis=1)
np.savez(f'{samples_dir}/inference.npz', transform=final, model_type=cfg.model_type)

'''
# HOI
rots = hoi_x_list[0][:, :6]
transls = hoi_x_list[0][:, 6:9]
scales = hoi_x_list[0][:, 9]
poses = hoi_x_list[0][:, 10:]

mats = rotation_6d_to_matrix(rots.cpu()).numpy()
mats = np.array([cv2.Rodrigues(R)[0].flatten() for R in mats])

with torch.no_grad():
	human_pose_dec_vec = model.decoder(poses.float())

final_pose_list = []
for i in range(21):
	nx_pose = human_pose_dec_vec[:, 6 * i : 6 * i + 6]
	nx_mat = rotation_6d_to_matrix(nx_pose.cpu()).numpy()
	nx_mats = np.array([cv2.Rodrigues(R)[0].flatten() for R in nx_mat])
	final_pose_list.append(nx_mats)
 
final_pose = np.concatenate(final_pose_list, axis=-1)
final_hoi = np.concatenate([mats, transls.cpu().numpy(), scales.cpu().numpy().reshape((-1, 1)), final_pose], axis=-1)

# HHI
p1_vecs = hhi_x_list[0][:, :10]
p2_rots = hhi_x_list[0][:, 10:16]
p2_transl = hhi_x_list[0][:, 16:19]
p2_vecs = hhi_x_list[0][:, 19:]

p2_mats = rotation_6d_to_matrix(p2_rots.cpu()).numpy()
p2_mats = np.array([cv2.Rodrigues(R)[0].flatten() for R in p2_mats])

# Decoding the two encoded vectors
with torch.no_grad():
	p1_dec = model.decoder(p1_vecs.float())
	p2_dec = model.decoder(p2_vecs.float())

# From 6-D to cv2.Rodrigues format (3-D)
final_pose_list_1 = []
final_pose_list_2 = []
for i in range(21):
	nx_pose_1 = p1_dec[:, 6 * i : 6 * i + 6]
	nx_mat_1 = rotation_6d_to_matrix(nx_pose_1.cpu()).numpy()
	nx_mats_1 = np.array([cv2.Rodrigues(R)[0].flatten() for R in nx_mat_1])
	final_pose_list_1.append(nx_mats_1)
	
	nx_pose_2 = p2_dec[:, 6 * i : 6 * i + 6]
	nx_mat_2 = rotation_6d_to_matrix(nx_pose_2.cpu()).numpy()
	nx_mats_2 = np.array([cv2.Rodrigues(R)[0].flatten() for R in nx_mat_2])
	final_pose_list_2.append(nx_mats_2)
	
final_pose_1 = np.concatenate(final_pose_list_1, axis=-1)
final_pose_2 = np.concatenate(final_pose_list_2, axis=-1)
final_hhi = np.concatenate([final_pose_1, p2_mats, p2_transl.cpu().detach().numpy(), final_pose_2], axis=-1)

np.savez(f'{samples_dir}/inference.npz', transform_hoi=final_hoi, transform_hhi=final_hhi, model_type=cfg.model_type)
'''
