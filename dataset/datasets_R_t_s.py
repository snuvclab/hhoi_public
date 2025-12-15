import sys
import os
import cv2
import random
import torch
import numpy as np
import _pickle as cPickle
import torch.utils.data as data
import copy
import pytorch3d
import pickle
sys.path.insert(0, '../')
import clip

from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from utils.data_augmentation import defor_2D, get_rotation
from utils.data_augmentation import data_augment
from utils.datasets_utils import aug_bbox_DZI, get_2d_coord_np, crop_resize_by_warp_affine
from utils.sgpa_utils import load_depth, get_bbox
from configs.config import get_config
from utils.misc import get_rot_matrix
from glob import glob
from pytorch3d.transforms import *

def raw_to_6d(raw_rep):
    six_d = []
    for i in range(44):
        axis_angle = torch.tensor(raw_rep[3*i : 3*i + 3]).unsqueeze(0)
        six_d_rep = matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle)).squeeze().numpy()
        six_d.append(six_d_rep)
    six_d = np.concatenate(six_d)
    final_rep = np.concatenate([six_d, raw_rep[132:]])
    return final_rep

def sixd_to_raw(sixd_rep):
    raw_rep = []
    for i in range(44):
        sixd_single = torch.tensor(sixd_rep[6*i : 6*i + 6]).unsqueeze(0)
        aa = matrix_to_axis_angle(rotation_6d_to_matrix(sixd_single)).squeeze().numpy()
        raw_rep.append(aa)
    raw_rep =  np.concatenate(raw_rep)
    final_rep = np.concatenate([raw_rep, sixd_rep[264:]])
    return final_rep  


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class RtsDataSet(data.Dataset):
    def __init__(self, data_dir=None, normalize_data=False):
        self.data_dir = data_dir

        data_list = []
        text_list = []
        for next_pkl in tqdm(sorted(glob(self.data_dir + "/*.pkl"))):
            with open(next_pkl, "rb") as f:
                pkl_content = pickle.load(f)
            orient = pkl_content['global_orient']
            transl = pkl_content['transl']
            scale_factor = pkl_content["scale_factor"]
            # transl = transl / scale_factor
            text_prompt = pkl_content['text_prompt']
            ### TEMP
            if "1 person sits on the chair" not in text_prompt:
                continue
            scale_factor = np.array(scale_factor).reshape((1, -1))
            data_list.append(np.concatenate([orient, transl, scale_factor], axis=-1))
            text_list.append(text_prompt)
        
        data_list = np.concatenate(data_list, axis=0)
        self.data_list = data_list
        print("total dataset: ", len(self.data_list))
        self.length = len(self.data_list)
        
        # Text Pool
        self.text_list = ["One person sits on the chair", 
                          "A person takes a seat on the chair", 
                          "Someone sits down on the chair", 
                          "One individual is seated on the chair", 
                          "A person rests on the chair", 
                          "Someone occupies the chair"]
        
        # Turning the text pool into text embedding
        print("Preparing CLIP text features")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda",jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.text_emb_list = []
        with torch.no_grad():
            for text_prompt in self.text_list:
                tokens = clip.tokenize(text_prompt, truncate = True).to("cuda")
                clip_feat = clip_model.encode_text(tokens).float()
                self.text_emb_list.append(clip_feat.detach().cpu().numpy())
        print("CLIP preprocessing Done!")
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        corresponding_data = self.data_list[index]
        
        rand_idx = np.random.randint(0, len(self.text_list))
        corresponding_clip_feat = self.text_emb_list[rand_idx]
        
        data_dict = {}
        data_dict['processed_data'] = torch.as_tensor(corresponding_data, dtype=torch.float32).contiguous()
        data_dict['clip_feat'] = torch.as_tensor(corresponding_clip_feat, dtype=torch.float32).contiguous()
        
        return data_dict
    
    
class HOIDataSet(data.Dataset):
    def __init__(self, data_dir=None, normalize_data=False):
        self.data_dir = data_dir
        
        self.index_map = []
        self.data_dict = {}
        text_dict = {}
        scenario_list = sorted(os.listdir(self.data_dir))
        for i, scenario in enumerate(scenario_list):
            data_list = []
            for j, next_pkl in enumerate(sorted(glob(self.data_dir + f"/{scenario}" + "/*.pkl"))):
                with open(next_pkl, "rb") as f:
                    pkl_content = pickle.load(f)
                orient = pkl_content['global_orient']
                transl = pkl_content['transl']
                scale_factor = pkl_content["scale_factor"]
                # transl = transl / scale_factor

                body_pose = pkl_content['body_pose']
                scale_factor = np.array(scale_factor).reshape((1, -1))
                data_list.append(np.concatenate([orient, transl, scale_factor, body_pose], axis=-1))
                
                self.index_map.append((i, j))
        
            self.data_dict[i] = np.concatenate(data_list, axis=0)
            
            # Text Pool
            text_file_path = os.path.join(self.data_dir, scenario, "text_prompt.txt")
            with open(text_file_path, "r", encoding="utf-8") as f:
                text_dict[i] = [line.strip() for line in f  if line.strip()]

        print("total dataset: ", len(self.index_map))
        self.length = len(self.index_map)

        
        # Turning the text pool into text embedding
        print("Preparing CLIP text features")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda",jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        
        self.text_emb_dict = {}
        with torch.no_grad():
            for key in tqdm(text_dict.keys()):
                text_list = text_dict[key]
                tokens = clip.tokenize(text_list, truncate = True).to("cuda") # (len(text_list), 77)
                clip_feat = clip_model.encode_text(tokens).float() # (len(text_list), 512)
                self.text_emb_dict[key] = clip_feat.detach().cpu().numpy()
        print("CLIP preprocessing Done!")
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        key, local_idx = self.index_map[index]
        corresponding_data = self.data_dict[key][local_idx]
        
        rand_idx = np.random.randint(0, len(self.text_emb_dict[key]))
        corresponding_clip_feat = self.text_emb_dict[key][rand_idx:rand_idx+1]
        
        data_dict = {}
        data_dict['processed_data'] = torch.as_tensor(corresponding_data, dtype=torch.float32).contiguous()
        data_dict['clip_feat'] = torch.as_tensor(corresponding_clip_feat, dtype=torch.float32).contiguous()
        
        return data_dict
    

class HHIDataSet(data.Dataset):
    def __init__(self, data_dir=None, normalize_data=False):
        self.data_dir = data_dir

        self.index_map = []
        self.data_dict = {}
        text_dict = {}
        scenario_list = sorted(os.listdir(self.data_dir))
        
        for i, scenario in enumerate(scenario_list):
            data_list = []
            for j, next_pkl in enumerate(sorted(glob(self.data_dir + f"/{scenario}" + "/*.pkl"))):
                with open(next_pkl, "rb") as f:
                    pkl_content = pickle.load(f)
                left_pose = pkl_content['left']['body_pose']
                right_rot = pkl_content['right']['global_orient']
                right_transl = pkl_content['right']['transl']
                right_pose = pkl_content['right']['body_pose']
                data_list.append(np.concatenate([left_pose, right_rot, right_transl, right_pose], axis=-1))

                self.index_map.append((i, j))
        
            self.data_dict[i] = np.concatenate(data_list, axis=0)
            
            # Text Pool
            text_file_path = os.path.join(self.data_dir, scenario, "text_prompt.txt")
            with open(text_file_path, "r", encoding="utf-8") as f:
                text_dict[i] = [line.strip() for line in f  if line.strip()]

        print("total dataset: ", len(self.index_map))
        self.length = len(self.index_map)
        
        
        # Turning the text pool into text embedding
        print("Preparing CLIP text features")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda",jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        
        self.text_emb_dict = {}
        with torch.no_grad():
            for key in tqdm(text_dict.keys()):
                text_list = text_dict[key]
                tokens = clip.tokenize(text_list, truncate = True).to("cuda") # (len(text_list), 77)
                clip_feat = clip_model.encode_text(tokens).float() # (len(text_list), 512)
                self.text_emb_dict[key] = clip_feat.detach().cpu().numpy()
        print("CLIP preprocessing Done!")
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        key, local_idx = self.index_map[index]
        corresponding_data = self.data_dict[key][local_idx]
        
        rand_idx = np.random.randint(0, len(self.text_emb_dict[key]))
        corresponding_clip_feat = self.text_emb_dict[key][rand_idx:rand_idx+1]
        
        data_dict = {}
        data_dict['processed_data'] = torch.as_tensor(corresponding_data, dtype=torch.float32).contiguous()
        data_dict['clip_feat'] = torch.as_tensor(corresponding_clip_feat, dtype=torch.float32).contiguous()
        
        return data_dict

def get_data_loaders(
    batch_size,
    seed,
    data_path=None,
    normalize_data=False,
    mode='train',
    model_type = 'hoi',
    num_workers=32,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if model_type == 'hoi':
        dataset = HOIDataSet(data_dir=data_path, normalize_data=normalize_data)
    elif model_type == 'hhi':
        dataset = HHIDataSet(data_dir=data_path, normalize_data=normalize_data)
    else:
        raise NotImplementedError

    
    if mode == 'train':
        shuffle = True
        num_workers = num_workers
    else:
        shuffle = False
        num_workers = 1
        
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=False,
        pin_memory=True,
    )

    return dataloader


def get_data_loaders_from_cfg(cfg):
    train_loader = get_data_loaders(
        batch_size=cfg.batch_size, 
        seed=cfg.seed,           
        data_path=cfg.data_path,
        normalize_data=cfg.normalize_data,
        mode='train',
        model_type = cfg.model_type,
        num_workers=cfg.num_workers,
    )
        
    return train_loader


def process_batch(batch_sample,
                  device,
                  model_type):
    

    processed_orig = batch_sample["processed_data"]
    
    
    if model_type == 'hoi':
        base_rot = processed_orig[..., :3]
        base_location = processed_orig[..., 3:6]
        base_scale = processed_orig[..., 6:7]
        base_theta = processed_orig[..., 7:]
        sixd_list = []
        
        for i in range(21):
            next_theta = base_theta[..., 3 * i : 3 * i + 3]
            next_rot_mat = torch.tensor(np.array([cv2.Rodrigues(row.reshape(3, 1))[0] for row in next_theta.numpy()]))
            next_6d = matrix_to_rotation_6d(next_rot_mat)
            sixd_list.append(next_6d)
        base_theta = torch.cat(sixd_list, dim=-1)
            
        base_rot = torch.tensor(np.array([cv2.Rodrigues(row.reshape(3, 1))[0] for row in base_rot.numpy()]))
        base_rot = matrix_to_rotation_6d(base_rot)

        processed_sample = {}
        processed_sample['gt_pose_scale'] = torch.cat([base_rot.float(), base_location.float(), base_scale.float(), base_theta.float()], dim=-1).to(device) # [bs, 6 + 3 + 1 + 126]
        processed_sample["clip_feat"] = batch_sample["clip_feat"].to(device)
    elif model_type == 'hhi':
        p1_pose = processed_orig[..., :63]
        p2_rot = processed_orig[..., 63:66]
        p2_transl = processed_orig[..., 66:69]
        p2_pose = processed_orig[..., 69:]
        
        p1_sixd_list = []
        p2_sixd_list = []
        for i in range(21):
            next_theta_1 = p1_pose[..., 3 * i : 3 * i + 3]
            next_rot_mat_1 = torch.tensor(np.array([cv2.Rodrigues(row.reshape(3, 1))[0] for row in next_theta_1.numpy()]))
            next_6d_1 = matrix_to_rotation_6d(next_rot_mat_1)
            p1_sixd_list.append(next_6d_1)
            
            next_theta_2 = p2_pose[..., 3 * i : 3 * i + 3]
            next_rot_mat_2 = torch.tensor(np.array([cv2.Rodrigues(row.reshape(3, 1))[0] for row in next_theta_2.numpy()]))
            next_6d_2 = matrix_to_rotation_6d(next_rot_mat_2)
            p2_sixd_list.append(next_6d_2)
        p1_theta = torch.cat(p1_sixd_list, dim=-1)
        p2_theta = torch.cat(p2_sixd_list, dim=-1)
            
        p2_rot = torch.tensor(np.array([cv2.Rodrigues(row.reshape(3, 1))[0] for row in p2_rot.numpy()]))
        p2_rot = matrix_to_rotation_6d(p2_rot)
        
        processed_sample = {}
        
        processed_sample['gt_pose_scale'] = torch.cat([p1_theta.float(), p2_rot.float(), p2_transl.float(), p2_theta.float()], dim=-1).to(device) # [bs, 126 + 6 + 3 + 126]
        processed_sample["clip_feat"] = batch_sample["clip_feat"].to(device)
    else:
        raise NotImplementedError

    return processed_sample 
    

if __name__ == '__main__':
    cfg = get_config()
    cfg.pose_mode = 'rot_matrix'
    data_loaders = get_data_loaders_from_cfg(cfg, data_type=['train', 'val', 'test'])
    train_loader = data_loaders['train_loader']
    val_loader = data_loaders['val_loader']
    test_loader = data_loaders['test_loader']
    for index, batch_sample in enumerate(tqdm(test_loader)):
        batch_sample = process_batch(
            batch_sample = batch_sample, 
            device=cfg.device, 
            pose_mode=cfg.pose_mode,
            PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS
        )
    for index, batch_sample in enumerate(tqdm(val_loader)):
        batch_sample = process_batch(
            batch_sample = batch_sample, 
            device=cfg.device, 
            pose_mode=cfg.pose_mode,
            PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS
        )
    for index, batch_sample in enumerate(tqdm(train_loader)):
        batch_sample = process_batch(
            batch_sample = batch_sample, 
            device=cfg.device, 
            pose_mode=cfg.pose_mode,
            PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS
        )
