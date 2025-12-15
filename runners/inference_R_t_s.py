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


def sixd_to_raw(sixd_rep):
    raw_rep = []
    for i in range(44):
        sixd_single = torch.tensor(sixd_rep[6*i : 6*i + 6]).unsqueeze(0)
        aa = matrix_to_axis_angle(rotation_6d_to_matrix(sixd_single)).squeeze().numpy()
        raw_rep.append(aa)
    raw_rep =  np.concatenate(raw_rep)
    final_rep = np.concatenate([raw_rep, sixd_rep[264:]])
    return final_rep  

score_agent = PoseNet(cfg)
score_agent.load_ckpt(model_dir=cfg.score_model_path, model_path=True, load_model_only=True) 

model = EncoderDecoder().to(cfg.device)
checkpoint = torch.load('enc_dec_human_pose/encoder_decoder_latest.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


num_repetitions = cfg.batch_size
prompt = cfg.input_text_prompt

text_input = []
for i in range(num_repetitions):
    text_input.append(prompt)

res_list, sampler_mode_list = score_agent.inference_score_func(batch_size=num_repetitions, text_prompt = text_input)

samples_dir = f"./results/inference/{cfg.log_dir}/{'_'.join(cfg.input_text_prompt.split(' '))}"
os.makedirs(samples_dir, exist_ok=True)
samples = {}

if cfg.model_type == 'hoi':
	rots = res_list[0][:, :6]
	transls = res_list[0][:, 6:9]
	scales = res_list[0][:, 9]
	poses = res_list[0][:, 10:]

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


	final = np.concatenate([mats, transls.cpu().numpy(), scales.cpu().numpy().reshape((-1, 1)), final_pose], axis=-1)
elif cfg.model_type == 'hhi':
	p1_vecs = res_list[0][:, :10]
	p2_rots = res_list[0][:, 10:16]
	p2_transl = res_list[0][:, 16:19]
	p2_vecs = res_list[0][:, 19:]

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

	final = np.concatenate([final_pose_1, p2_mats, p2_transl.cpu().detach().numpy(), final_pose_2], axis=-1)
else:
    raise NotImplementedError

np.savez(f'{samples_dir}/inference.npz', transform=final, model_type=cfg.model_type)
