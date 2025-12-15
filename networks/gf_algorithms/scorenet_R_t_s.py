import sys
import os
import torch
import numpy as np
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.genpose_utils import get_pose_dim
from networks.decoder_head.rot_head import RotHead
from networks.decoder_head.trans_head import TransHead
import clip

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.norm2(out + x)
        return self.act(out)

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    
class HHOIScoreNet(nn.Module):
    def __init__(self, marginal_prob_func, human_pose_dim=126):
        super(HHOIScoreNet, self).__init__()
        self.act = nn.ReLU(True)
        
        self.hoi_pose_dim = 10 + human_pose_dim
        self.hhi_pose_dim = human_pose_dim + 9 + human_pose_dim
        
        ''' encode pose '''
        self.hoi_pose_encoder = nn.Sequential(
            nn.Linear(self.hoi_pose_dim, 256),
            #nn.LayerNorm(256),
            self.act,
            ResidualBlock(256),
            #self.act,
            nn.Linear(256, 256),
            #nn.LayerNorm(256),
            self.act,
            ResidualBlock(256),
            #self.act,
            nn.Linear(256, 256),
            #nn.LayerNorm(256),
            self.act,
            ResidualBlock(256),
            #self.act,
        )
        
        self.hhi_pose_encoder = nn.Sequential(
            nn.Linear(self.hhi_pose_dim, 256),
            #nn.LayerNorm(256),
            self.act,
            ResidualBlock(256),
            nn.Linear(256, 256),
            #nn.LayerNorm(256),
            self.act,
            ResidualBlock(256),
            nn.Linear(256, 256),
            self.act,
            ResidualBlock(256),
        )
        
        ''' encode text '''
        text_dim = 512
        self.hoi_prompt_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            #nn.LayerNorm(256),
            self.act,
            ResidualBlock(256),
            #self.act,
            nn.Linear(256, 256),
            #nn.LayerNorm(256),
            self.act,
            ResidualBlock(256),
            #self.act,
            nn.Linear(256, 128),
            #nn.LayerNorm(128),
            self.act,
            ResidualBlock(128),
            #self.act,
        )
        
        self.hhi_prompt_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            self.act,
            ResidualBlock(256),
            nn.Linear(256, 256),
            self.act,
            ResidualBlock(256),
            nn.Linear(256, 128),
            self.act,
            ResidualBlock(128),
        )
        
        ''' encode t '''
        self.hoi_t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            # self.act, # M4D26 update
            ResidualBlock(128),
            nn.Linear(128, 128),
            self.act,
            ResidualBlock(128),
            #nn.LayerNorm(128),
            #self.act,
        )
        
        self.hhi_t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            ResidualBlock(128),
            # self.act, # M4D26 update
            nn.Linear(128, 128),
            self.act,
            ResidualBlock(128),
        )
        
        def make_head(out_dim):
            return nn.Sequential(
                ResidualBlock(128+256+128),
                nn.Linear(128+256+128, 256),
                self.act,
                ResidualBlock(256),
                zero_module(nn.Linear(256, out_dim))
            )
            
        # HOI
        ''' rotation regress head '''
        self.hoi_fusion_tail_base_rot_x = make_head(3)
        self.hoi_fusion_tail_base_rot_y = make_head(3)
        ''' translation regress head '''
        self.hoi_fusion_tail_base_trans = make_head(3)
        ''' scale regress head '''
        self.hoi_fusion_tail_base_scale = make_head(1)
        ''' theta regress head '''
        self.hoi_fusion_theta = make_head(human_pose_dim)
        
        # HHI
        ''' rotation regress head '''
        self.hhi_fusion_p2_rot_x = make_head(3)
        self.hhi_fusion_p2_rot_y = make_head(3)
        ''' translation regress head '''
        self.hhi_fusion_p2_trans = make_head(3)
        ''' theta regress head '''
        self.hhi_fusion_p1_theta = make_head(human_pose_dim)
        self.hhi_fusion_p2_theta = make_head(human_pose_dim)
            
        self.marginal_prob_func = marginal_prob_func
        
    def hoi(self, data):
        sampled_pose = data['hoi_sampled_pose']
        t = data['t']
        clip_feat = data["hoi_clip_feat"].squeeze()
        
        t_feat = self.hoi_t_encoder(t.squeeze(1))
        pose_feat = self.hoi_pose_encoder(sampled_pose)
        text_feat = self.hoi_prompt_encoder(clip_feat)
        
        total_feat = torch.cat([text_feat, t_feat, pose_feat], dim=-1)
        _, std = self.marginal_prob_func(total_feat, t)
        
        base_rot_x = self.hoi_fusion_tail_base_rot_x(total_feat)
        base_rot_y = self.hoi_fusion_tail_base_rot_y(total_feat)
        base_trans = self.hoi_fusion_tail_base_trans(total_feat)
        base_scale = self.hoi_fusion_tail_base_scale(total_feat)
        base_theta = self.hoi_fusion_theta(total_feat)
        out_score = torch.cat([base_rot_x, base_rot_y, base_trans, base_scale, base_theta], dim=-1) / (std+1e-7)

        return out_score
    
    def hhi(self, data):
        sampled_pose = data['hhi_sampled_pose']
        t = data['t']
        clip_feat = data["hhi_clip_feat"].squeeze()
        
        t_feat = self.hhi_t_encoder(t.squeeze(1))
        pose_feat = self.hhi_pose_encoder(sampled_pose)
        text_feat = self.hhi_prompt_encoder(clip_feat)
        
        total_feat = torch.cat([text_feat, t_feat, pose_feat], dim=-1)
        _, std = self.marginal_prob_func(total_feat, t)
        
        p2_rot_x = self.hhi_fusion_p2_rot_x(total_feat)
        p2_rot_y = self.hhi_fusion_p2_rot_y(total_feat)
        p2_trans = self.hhi_fusion_p2_trans(total_feat)
        p1_theta = self.hhi_fusion_p1_theta(total_feat)
        p2_theta = self.hhi_fusion_p2_theta(total_feat)
        out_score = torch.cat([p1_theta, p2_rot_x, p2_rot_y, p2_trans, p2_theta], dim=-1) / (std+1e-7) # normalisation
        
        return out_score
    
    def hhoi_exp(self, data):
        h1oi_sampled_pose = data['h1oi_sampled_pose']
        h2oi_sampled_pose = data['h2oi_sampled_pose']
        hhi_sampled_pose = data['hhi_sampled_pose']
        t = data['t']
        h1oi_clip_feat = data["h1oi_clip_feat"].squeeze()
        h2oi_clip_feat = data["h2oi_clip_feat"].squeeze()
        hhi_clip_feat = data["hhi_clip_feat"].squeeze()
        
        hoi_t_feat = self.hoi_t_encoder(t.squeeze(1))
        hhi_t_feat = self.hhi_t_encoder(t.squeeze(1))
        h1oi_pose_feat = self.hoi_pose_encoder(h1oi_sampled_pose)
        h2oi_pose_feat = self.hoi_pose_encoder(h2oi_sampled_pose)
        hhi_pose_feat = self.hhi_pose_encoder(hhi_sampled_pose)
        h1oi_text_feat = self.hoi_prompt_encoder(h1oi_clip_feat)
        h2oi_text_feat = self.hoi_prompt_encoder(h2oi_clip_feat)
        hhi_text_feat = self.hhi_prompt_encoder(hhi_clip_feat)
        
        # h1oi
        h1oi_total_feat = torch.cat([h1oi_text_feat, hoi_t_feat, h1oi_pose_feat], dim=-1)
        _, std = self.marginal_prob_func(h1oi_total_feat, t)
        
        h1oi_base_rot_x = self.hoi_fusion_tail_base_rot_x(h1oi_total_feat)
        h1oi_base_rot_y = self.hoi_fusion_tail_base_rot_y(h1oi_total_feat)
        h1oi_base_trans = self.hoi_fusion_tail_base_trans(h1oi_total_feat)
        h1oi_base_scale = self.hoi_fusion_tail_base_scale(h1oi_total_feat)
        h1oi_base_theta = self.hoi_fusion_theta(h1oi_total_feat)
        h1oi_score = torch.cat([h1oi_base_rot_x, h1oi_base_rot_y, h1oi_base_trans, h1oi_base_scale, h1oi_base_theta], dim=-1) / (std+1e-7)
        
        # h2oi
        h2oi_total_feat = torch.cat([h2oi_text_feat, hoi_t_feat, h2oi_pose_feat], dim=-1)
        _, std = self.marginal_prob_func(h2oi_total_feat, t)
        
        h2oi_base_rot_x = self.hoi_fusion_tail_base_rot_x(h2oi_total_feat)
        h2oi_base_rot_y = self.hoi_fusion_tail_base_rot_y(h2oi_total_feat)
        h2oi_base_trans = self.hoi_fusion_tail_base_trans(h2oi_total_feat)
        h2oi_base_scale = self.hoi_fusion_tail_base_scale(h2oi_total_feat)
        h2oi_base_theta = self.hoi_fusion_theta(h2oi_total_feat)
        h2oi_score = torch.cat([h2oi_base_rot_x, h2oi_base_rot_y, h2oi_base_trans, h2oi_base_scale, h2oi_base_theta], dim=-1) / (std+1e-7)
        
        # hhi
        hhi_total_feat = torch.cat([hhi_text_feat, hhi_t_feat, hhi_pose_feat], dim=-1)
        _, std = self.marginal_prob_func(hhi_total_feat, t)
        
        p2_rot_x = self.hhi_fusion_p2_rot_x(hhi_total_feat)
        p2_rot_y = self.hhi_fusion_p2_rot_y(hhi_total_feat)
        p2_trans = self.hhi_fusion_p2_trans(hhi_total_feat)
        p1_theta = self.hhi_fusion_p1_theta(hhi_total_feat)
        p2_theta = self.hhi_fusion_p2_theta(hhi_total_feat)
        hhi_score = torch.cat([p1_theta, p2_rot_x, p2_rot_y, p2_trans, p2_theta], dim=-1) / (std+1e-7) # normalisation
        
        return torch.cat([h1oi_score, h2oi_score, hhi_score], dim=-1)
    
    def hhoi(self, data):
        sampled_pose = data['sampled_pose']              # (B, hoi_pose_dim*num_hoi + hhi_pose_dim*num_hhi)
        t = data['t']
        hoi_clip_feat = data["hoi_clip_feat"].squeeze()  # (B*num_hoi, F)
        hhi_clip_feat = data["hhi_clip_feat"].squeeze()  # (B*num_hhi, F)
        
        B = sampled_pose.shape[0]
        full_pose_dim = sampled_pose.shape[1]
        num_hoi = data['num_hoi']
        num_hhi = data['num_hhi']
        
        hoi_clip_feat = hoi_clip_feat.contiguous().view(B, num_hoi, -1)  # (B, num_hoi, F)
        hhi_clip_feat = hhi_clip_feat.contiguous().view(B, num_hhi, -1)  # (B, num_hhi, F)
        
        hoi_t_feat = self.hoi_t_encoder(t.squeeze(1))
        hhi_t_feat = self.hhi_t_encoder(t.squeeze(1))
        
        score_list = []
        for i in range(num_hoi):
            hoi_sampled_pose = sampled_pose[:, self.hoi_pose_dim*i:self.hoi_pose_dim*(i+1)]
            hoi_pose_feat = self.hoi_pose_encoder(hoi_sampled_pose)
            hoi_text_feat = self.hoi_prompt_encoder(hoi_clip_feat[:, i, :])
            hoi_total_feat = torch.cat([hoi_text_feat, hoi_t_feat, hoi_pose_feat], dim=-1)
            _, std = self.marginal_prob_func(hoi_total_feat, t)
            
            hoi_base_rot_x = self.hoi_fusion_tail_base_rot_x(hoi_total_feat)
            hoi_base_rot_y = self.hoi_fusion_tail_base_rot_y(hoi_total_feat)
            hoi_base_trans = self.hoi_fusion_tail_base_trans(hoi_total_feat)
            hoi_base_scale = self.hoi_fusion_tail_base_scale(hoi_total_feat)
            hoi_base_theta = self.hoi_fusion_theta(hoi_total_feat)
            hoi_score = torch.cat([hoi_base_rot_x, hoi_base_rot_y, hoi_base_trans, hoi_base_scale, hoi_base_theta], dim=-1) / (std+1e-7)
            score_list.append(hoi_score)
        
        for i in range(num_hhi):
            hhi_sampled_pose = sampled_pose[:, full_pose_dim-self.hhi_pose_dim*(num_hhi-i):full_pose_dim-self.hhi_pose_dim*(num_hhi-1-i)]
            hhi_pose_feat = self.hhi_pose_encoder(hhi_sampled_pose)
            hhi_text_feat = self.hhi_prompt_encoder(hhi_clip_feat[:, i, :])
            hhi_total_feat = torch.cat([hhi_text_feat, hhi_t_feat, hhi_pose_feat], dim=-1)
            _, std = self.marginal_prob_func(hhi_total_feat, t)
            
            p2_rot_x = self.hhi_fusion_p2_rot_x(hhi_total_feat)
            p2_rot_y = self.hhi_fusion_p2_rot_y(hhi_total_feat)
            p2_trans = self.hhi_fusion_p2_trans(hhi_total_feat)
            p1_theta = self.hhi_fusion_p1_theta(hhi_total_feat)
            p2_theta = self.hhi_fusion_p2_theta(hhi_total_feat)
            hhi_score = torch.cat([p1_theta, p2_rot_x, p2_rot_y, p2_trans, p2_theta], dim=-1) / (std+1e-7) # normalization
            score_list.append(hhi_score)
        
        total_score = torch.cat(score_list, dim=-1)
        
        return total_score
        
    def forward(self, data):
        '''
        Args:
            data, dict {
                'pts_feat': [bs, c]
                'pose_sample': [bs, pose_dim]
                't': [bs, 1]
            }
        '''
        
        out_score = None

        return out_score    

class HHOIScoreNet_old(nn.Module):
    def __init__(self, marginal_prob_func, human_pose_dim=126):
        super(HHOIScoreNet_old, self).__init__()
        self.act = nn.ReLU(True)
        
        self.hoi_pose_dim = 10 + human_pose_dim
        self.hhi_pose_dim = human_pose_dim + 9 + human_pose_dim
        
        ''' encode pose '''
        self.hoi_pose_encoder = nn.Sequential(
            nn.Linear(self.hoi_pose_dim, 256),
            #nn.LayerNorm(256),
            self.act,
            nn.Linear(256, 256),
            #nn.LayerNorm(256),
            self.act,
            nn.Linear(256, 256),
            self.act,
        )
        
        self.hhi_pose_encoder = nn.Sequential(
            nn.Linear(self.hhi_pose_dim, 256),
            #nn.LayerNorm(256),
            self.act,
            nn.Linear(256, 256),
            #nn.LayerNorm(256),
            self.act,
            nn.Linear(256, 256),
            self.act,
        )
        
        ''' encode text '''
        text_dim = 512
        self.hoi_prompt_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
            nn.Linear(256, 128),
            self.act,
        )
        
        self.hhi_prompt_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
            nn.Linear(256, 128),
            self.act,
        )
        
        ''' encode t '''
        self.hoi_t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            # self.act, # M4D26 update
            nn.Linear(128, 128),
            self.act,
        )
        
        self.hhi_t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            # self.act, # M4D26 update
            nn.Linear(128, 128),
            self.act,
        )
        
        ''' rotation_x_axis regress head '''
        self.hoi_fusion_tail_base_rot_x = nn.Sequential(
			#nn.Linear(128+256+1024, 256),
			nn.Linear(128+256+128, 256),
			# nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, 3)),
		)
        
        ''' rotation_y_axis regress head '''
        self.hoi_fusion_tail_base_rot_y = nn.Sequential(
			#nn.Linear(128+256+1024, 256),
			nn.Linear(128+256+128, 256),
			# nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, 3)),
		)
        
        ''' translation regress head '''
        self.hoi_fusion_tail_base_trans = nn.Sequential(
			#nn.Linear(128+256+1024, 256),
			nn.Linear(128+256+128, 256),
			# nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, 3)),
		)
        
        ''' scale regress head '''
        self.hoi_fusion_tail_base_scale = nn.Sequential(
			#nn.Linear(128+256+1024, 256),
			nn.Linear(128+256+128, 256),
			# nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, 1)),
		)
        
        ''' human pose regress head '''
        self.hoi_fusion_theta = nn.Sequential(
            nn.Linear(128+256+128, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
            zero_module(nn.Linear(256, human_pose_dim))
        )
        
        ''' regress head for hhi '''
        self.hhi_fusion_p2_rot_x = nn.Sequential(
			#nn.Linear(128+256+1024, 256),
			nn.Linear(128+256+128, 256),
			# nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, 3)),
		)
        
        self.hhi_fusion_p2_rot_y = nn.Sequential(
			#nn.Linear(128+256+1024, 256),
			nn.Linear(128+256+128, 256),
			# nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, 3)),
		)
        
        self.hhi_fusion_p2_trans = nn.Sequential(
			#nn.Linear(128+256+1024, 256),
			nn.Linear(128+256+128, 256),
			# nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, 3)),
		)
        
        self.hhi_fusion_p1_theta = nn.Sequential(
			nn.Linear(128+256+128, 256),
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, human_pose_dim)),
		)
        
        self.hhi_fusion_p2_theta = nn.Sequential(
			nn.Linear(128+256+128, 256),
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, human_pose_dim)),
		)
            
        self.marginal_prob_func = marginal_prob_func
        
    def hoi(self, data):
        sampled_pose = data['hoi_sampled_pose']
        t = data['t']
        clip_feat = data["hoi_clip_feat"].squeeze()
        
        t_feat = self.hoi_t_encoder(t.squeeze(1))
        pose_feat = self.hoi_pose_encoder(sampled_pose)
        text_feat = self.hoi_prompt_encoder(clip_feat)
        
        total_feat = torch.cat([text_feat, t_feat, pose_feat], dim=-1)
        _, std = self.marginal_prob_func(total_feat, t)
        
        base_rot_x = self.hoi_fusion_tail_base_rot_x(total_feat)
        base_rot_y = self.hoi_fusion_tail_base_rot_y(total_feat)
        base_trans = self.hoi_fusion_tail_base_trans(total_feat)
        base_scale = self.hoi_fusion_tail_base_scale(total_feat)
        base_theta = self.hoi_fusion_theta(total_feat)
        out_score = torch.cat([base_rot_x, base_rot_y, base_trans, base_scale, base_theta], dim=-1) / (std+1e-7)

        return out_score
    
    def hhi(self, data):
        sampled_pose = data['hhi_sampled_pose']
        t = data['t']
        clip_feat = data["hhi_clip_feat"].squeeze()
        
        t_feat = self.hhi_t_encoder(t.squeeze(1))
        pose_feat = self.hhi_pose_encoder(sampled_pose)
        text_feat = self.hhi_prompt_encoder(clip_feat)
        
        total_feat = torch.cat([text_feat, t_feat, pose_feat], dim=-1)
        _, std = self.marginal_prob_func(total_feat, t)
        
        p2_rot_x = self.hhi_fusion_p2_rot_x(total_feat)
        p2_rot_y = self.hhi_fusion_p2_rot_y(total_feat)
        p2_trans = self.hhi_fusion_p2_trans(total_feat)
        p1_theta = self.hhi_fusion_p1_theta(total_feat)
        p2_theta = self.hhi_fusion_p2_theta(total_feat)
        out_score = torch.cat([p1_theta, p2_rot_x, p2_rot_y, p2_trans, p2_theta], dim=-1) / (std+1e-7) # normalisation
        
        return out_score
    
    def hhoi_exp(self, data):
        h1oi_sampled_pose = data['h1oi_sampled_pose']
        h2oi_sampled_pose = data['h2oi_sampled_pose']
        hhi_sampled_pose = data['hhi_sampled_pose']
        t = data['t']
        h1oi_clip_feat = data["h1oi_clip_feat"].squeeze()
        h2oi_clip_feat = data["h2oi_clip_feat"].squeeze()
        hhi_clip_feat = data["hhi_clip_feat"].squeeze()
        
        hoi_t_feat = self.hoi_t_encoder(t.squeeze(1))
        hhi_t_feat = self.hhi_t_encoder(t.squeeze(1))
        h1oi_pose_feat = self.hoi_pose_encoder(h1oi_sampled_pose)
        h2oi_pose_feat = self.hoi_pose_encoder(h2oi_sampled_pose)
        hhi_pose_feat = self.hhi_pose_encoder(hhi_sampled_pose)
        h1oi_text_feat = self.hoi_prompt_encoder(h1oi_clip_feat)
        h2oi_text_feat = self.hoi_prompt_encoder(h2oi_clip_feat)
        hhi_text_feat = self.hhi_prompt_encoder(hhi_clip_feat)
        
        # h1oi
        h1oi_total_feat = torch.cat([h1oi_text_feat, hoi_t_feat, h1oi_pose_feat], dim=-1)
        _, std = self.marginal_prob_func(h1oi_total_feat, t)
        
        h1oi_base_rot_x = self.hoi_fusion_tail_base_rot_x(h1oi_total_feat)
        h1oi_base_rot_y = self.hoi_fusion_tail_base_rot_y(h1oi_total_feat)
        h1oi_base_trans = self.hoi_fusion_tail_base_trans(h1oi_total_feat)
        h1oi_base_scale = self.hoi_fusion_tail_base_scale(h1oi_total_feat)
        h1oi_base_theta = self.hoi_fusion_theta(h1oi_total_feat)
        h1oi_score = torch.cat([h1oi_base_rot_x, h1oi_base_rot_y, h1oi_base_trans, h1oi_base_scale, h1oi_base_theta], dim=-1) / (std+1e-7)
        
        # h2oi
        h2oi_total_feat = torch.cat([h2oi_text_feat, hoi_t_feat, h2oi_pose_feat], dim=-1)
        _, std = self.marginal_prob_func(h2oi_total_feat, t)
        
        h2oi_base_rot_x = self.hoi_fusion_tail_base_rot_x(h2oi_total_feat)
        h2oi_base_rot_y = self.hoi_fusion_tail_base_rot_y(h2oi_total_feat)
        h2oi_base_trans = self.hoi_fusion_tail_base_trans(h2oi_total_feat)
        h2oi_base_scale = self.hoi_fusion_tail_base_scale(h2oi_total_feat)
        h2oi_base_theta = self.hoi_fusion_theta(h2oi_total_feat)
        h2oi_score = torch.cat([h2oi_base_rot_x, h2oi_base_rot_y, h2oi_base_trans, h2oi_base_scale, h2oi_base_theta], dim=-1) / (std+1e-7)
        
        # hhi
        hhi_total_feat = torch.cat([hhi_text_feat, hhi_t_feat, hhi_pose_feat], dim=-1)
        _, std = self.marginal_prob_func(hhi_total_feat, t)
        
        p2_rot_x = self.hhi_fusion_p2_rot_x(hhi_total_feat)
        p2_rot_y = self.hhi_fusion_p2_rot_y(hhi_total_feat)
        p2_trans = self.hhi_fusion_p2_trans(hhi_total_feat)
        p1_theta = self.hhi_fusion_p1_theta(hhi_total_feat)
        p2_theta = self.hhi_fusion_p2_theta(hhi_total_feat)
        hhi_score = torch.cat([p1_theta, p2_rot_x, p2_rot_y, p2_trans, p2_theta], dim=-1) / (std+1e-7) # normalisation
        
        return torch.cat([h1oi_score, h2oi_score, hhi_score], dim=-1)
    
    def hhoi(self, data):
        sampled_pose = data['sampled_pose']              # (B, hoi_pose_dim*num_hoi + hhi_pose_dim*num_hhi)
        t = data['t']
        hoi_clip_feat = data["hoi_clip_feat"].squeeze()  # (B*num_hoi, F)
        hhi_clip_feat = data["hhi_clip_feat"].squeeze()  # (B*num_hhi, F)
        
        B = sampled_pose.shape[0]
        full_pose_dim = sampled_pose.shape[1]
        num_hoi = data['num_hoi']
        num_hhi = data['num_hhi']
        
        hoi_clip_feat = hoi_clip_feat.contiguous().view(B, num_hoi, -1)  # (B, num_hoi, F)
        hhi_clip_feat = hhi_clip_feat.contiguous().view(B, num_hhi, -1)  # (B, num_hhi, F)
        
        hoi_t_feat = self.hoi_t_encoder(t.squeeze(1))
        hhi_t_feat = self.hhi_t_encoder(t.squeeze(1))
        
        score_list = []
        for i in range(num_hoi):
            hoi_sampled_pose = sampled_pose[:, self.hoi_pose_dim*i:self.hoi_pose_dim*(i+1)]
            hoi_pose_feat = self.hoi_pose_encoder(hoi_sampled_pose)
            hoi_text_feat = self.hoi_prompt_encoder(hoi_clip_feat[:, i, :])
            hoi_total_feat = torch.cat([hoi_text_feat, hoi_t_feat, hoi_pose_feat], dim=-1)
            _, std = self.marginal_prob_func(hoi_total_feat, t)
            
            hoi_base_rot_x = self.hoi_fusion_tail_base_rot_x(hoi_total_feat)
            hoi_base_rot_y = self.hoi_fusion_tail_base_rot_y(hoi_total_feat)
            hoi_base_trans = self.hoi_fusion_tail_base_trans(hoi_total_feat)
            hoi_base_scale = self.hoi_fusion_tail_base_scale(hoi_total_feat)
            hoi_base_theta = self.hoi_fusion_theta(hoi_total_feat)
            hoi_score = torch.cat([hoi_base_rot_x, hoi_base_rot_y, hoi_base_trans, hoi_base_scale, hoi_base_theta], dim=-1) / (std+1e-7)
            score_list.append(hoi_score)
        
        for i in range(num_hhi):
            hhi_sampled_pose = sampled_pose[:, full_pose_dim-self.hhi_pose_dim*(num_hhi-i):full_pose_dim-self.hhi_pose_dim*(num_hhi-1-i)]
            hhi_pose_feat = self.hhi_pose_encoder(hhi_sampled_pose)
            hhi_text_feat = self.hhi_prompt_encoder(hhi_clip_feat[:, i, :])
            hhi_total_feat = torch.cat([hhi_text_feat, hhi_t_feat, hhi_pose_feat], dim=-1)
            _, std = self.marginal_prob_func(hhi_total_feat, t)
            
            p2_rot_x = self.hhi_fusion_p2_rot_x(hhi_total_feat)
            p2_rot_y = self.hhi_fusion_p2_rot_y(hhi_total_feat)
            p2_trans = self.hhi_fusion_p2_trans(hhi_total_feat)
            p1_theta = self.hhi_fusion_p1_theta(hhi_total_feat)
            p2_theta = self.hhi_fusion_p2_theta(hhi_total_feat)
            hhi_score = torch.cat([p1_theta, p2_rot_x, p2_rot_y, p2_trans, p2_theta], dim=-1) / (std+1e-7) # normalization
            score_list.append(hhi_score)
        
        total_score = torch.cat(score_list, dim=-1)
        
        return total_score
        
    def forward(self, data):
        '''
        Args:
            data, dict {
                'pts_feat': [bs, c]
                'pose_sample': [bs, pose_dim]
                't': [bs, 1]
            }
        '''
        
        out_score = None

        return out_score
    
class HHIScoreNet(nn.Module):
    def __init__(self, marginal_prob_func, human_pose_dim=126):
        """_summary_

        Args:
            marginal_prob_func (func): marginal_prob_func of score network

        Raises:
            NotImplementedError: _description_
        """
        super(HHIScoreNet, self).__init__()
        self.act = nn.ReLU(True)
        
        pose_dim = human_pose_dim + 9 + human_pose_dim

        ''' encode pose '''
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 256),
            #nn.LayerNorm(256),
            self.act,
            ResidualBlock(256),
            nn.Linear(256, 256),
            #nn.LayerNorm(256),
            self.act,
            ResidualBlock(256),
            nn.Linear(256, 256),
            self.act,
            ResidualBlock(256),
        )
        
        # Text Encoding
        text_dim = 512
        self.prompt_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            self.act,
            ResidualBlock(256),
            nn.Linear(256, 256),
            self.act,
            ResidualBlock(256),
            nn.Linear(256, 128),
            self.act,
            ResidualBlock(128),
        )
        
        ''' encode t '''
        self.t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            ResidualBlock(128),
            # self.act, # M4D26 update
            nn.Linear(128, 128),
            self.act,
            ResidualBlock(128),
        )
        
        def make_head(out_dim):
            return nn.Sequential(
                ResidualBlock(128+256+128),
                nn.Linear(128+256+128, 256),
                self.act,
                ResidualBlock(256),
                zero_module(nn.Linear(256, out_dim))
            )
        
        ''' rotation regress head '''
        self.fusion_p2_rot_x = make_head(3)
        self.fusion_p2_rot_y = make_head(3)
        ''' translation regress head '''
        self.fusion_p2_trans = make_head(3)
        ''' theta regress head '''
        self.fusion_p1_theta = make_head(human_pose_dim)
        self.fusion_p2_theta = make_head(human_pose_dim)
            
        self.marginal_prob_func = marginal_prob_func


    def forward(self, data):
        '''
        Args:
            data, dict {
                'pts_feat': [bs, c]
                'pose_sample': [bs, pose_dim]
                't': [bs, 1]
            }
        '''
        
        #pts_feat = data['pts_feat']
        sampled_pose = data['sampled_pose']
        t = data['t']
        clip_feat = data["clip_feat"].squeeze()
        
        t_feat = self.t_encoder(t.squeeze(1))
        pose_feat = self.pose_encoder(sampled_pose)
        text_feat = self.prompt_encoder(clip_feat)
        
        total_feat = torch.cat([text_feat, t_feat, pose_feat], dim=-1)
        _, std = self.marginal_prob_func(total_feat, t)
        
        p2_rot_x = self.fusion_p2_rot_x(total_feat)
        p2_rot_y = self.fusion_p2_rot_y(total_feat)
        p2_trans = self.fusion_p2_trans(total_feat)
        p1_theta = self.fusion_p1_theta(total_feat)
        p2_theta = self.fusion_p2_theta(total_feat)
        out_score = torch.cat([p1_theta, p2_rot_x, p2_rot_y, p2_trans, p2_theta], dim=-1) / (std+1e-7) # normalisation
        
        return out_score
    
class HHIScoreNet_old(nn.Module):
    def __init__(self, marginal_prob_func, human_pose_dim=126):
        """_summary_

        Args:
            marginal_prob_func (func): marginal_prob_func of score network

        Raises:
            NotImplementedError: _description_
        """
        super(HHIScoreNet_old, self).__init__()
        self.act = nn.ReLU(True)
        
        pose_dim = human_pose_dim + 9 + human_pose_dim

        ''' encode pose '''
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 256),
            #nn.LayerNorm(256),
            self.act,
            nn.Linear(256, 256),
            #nn.LayerNorm(256),
            self.act,
            nn.Linear(256, 256),
            self.act,
        )
        
        # Text Encoding
        text_dim = 512
        self.prompt_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
            nn.Linear(256, 128),
            self.act,
        )
        
        ''' encode t '''
        self.t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            # self.act, # M4D26 update
            nn.Linear(128, 128),
            self.act,
        )
        
		# MLP heads for R, t, Theta
        self.fusion_p2_rot_x = nn.Sequential(
			#nn.Linear(128+256+1024, 256),
			nn.Linear(128+256+128, 256),
			# nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, 3)),
		)
        
        self.fusion_p2_rot_y = nn.Sequential(
			#nn.Linear(128+256+1024, 256),
			nn.Linear(128+256+128, 256),
			# nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, 3)),
		)
        
        self.fusion_p2_trans = nn.Sequential(
			#nn.Linear(128+256+1024, 256),
			nn.Linear(128+256+128, 256),
			# nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, 3)),
		)
        
        self.fusion_p1_theta = nn.Sequential(
			nn.Linear(128+256+128, 256),
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, human_pose_dim)),
		)
        
        self.fusion_p2_theta = nn.Sequential(
			nn.Linear(128+256+128, 256),
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, human_pose_dim)),
		)
            
        self.marginal_prob_func = marginal_prob_func


    def forward(self, data):
        '''
        Args:
            data, dict {
                'pts_feat': [bs, c]
                'pose_sample': [bs, pose_dim]
                't': [bs, 1]
            }
        '''
        
        #pts_feat = data['pts_feat']
        sampled_pose = data['sampled_pose']
        t = data['t']
        clip_feat = data["clip_feat"].squeeze()
        
        t_feat = self.t_encoder(t.squeeze(1))
        pose_feat = self.pose_encoder(sampled_pose)
        text_feat = self.prompt_encoder(clip_feat)
        
        total_feat = torch.cat([text_feat, t_feat, pose_feat], dim=-1)
        _, std = self.marginal_prob_func(total_feat, t)
        
        p2_rot_x = self.fusion_p2_rot_x(total_feat)
        p2_rot_y = self.fusion_p2_rot_y(total_feat)
        p2_trans = self.fusion_p2_trans(total_feat)
        p1_theta = self.fusion_p1_theta(total_feat)
        p2_theta = self.fusion_p2_theta(total_feat)
        out_score = torch.cat([p1_theta, p2_rot_x, p2_rot_y, p2_trans, p2_theta], dim=-1) / (std+1e-7) # normalisation
        
        return out_score


class HOIScoreNet(nn.Module):
    def __init__(self, marginal_prob_func, human_pose_dim=126):
        """_summary_

        Args:
            marginal_prob_func (func): marginal_prob_func of score network
            use_human_pose (bool): whether the model considers only the R, t, s of the person or its body pose as well

        Raises:
            NotImplementedError: _description_
        """
        super(HOIScoreNet, self).__init__()
        self.act = nn.ReLU(True)
        
        pose_dim = 10 + human_pose_dim


        ''' encode pose '''
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 256),
            #nn.LayerNorm(256),
            self.act,
            ResidualBlock(256),
            #self.act,
            nn.Linear(256, 256),
            #nn.LayerNorm(256),
            self.act,
            ResidualBlock(256),
            #self.act,
            nn.Linear(256, 256),
            #nn.LayerNorm(256),
            self.act,
            ResidualBlock(256),
            #self.act,
        )
        
        text_dim = 512
        self.prompt_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            #nn.LayerNorm(256),
            self.act,
            ResidualBlock(256),
            #self.act,
            nn.Linear(256, 256),
            #nn.LayerNorm(256),
            self.act,
            ResidualBlock(256),
            #self.act,
            nn.Linear(256, 128),
            #nn.LayerNorm(128),
            self.act,
            ResidualBlock(128),
            #self.act,
        )
        
        ''' encode t '''
        self.t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            # self.act, # M4D26 update
            ResidualBlock(128),
            nn.Linear(128, 128),
            self.act,
            ResidualBlock(128),
            #nn.LayerNorm(128),
            #self.act,
        )
        
        def make_head(out_dim):
            return nn.Sequential(
                ResidualBlock(128+256+128),
                nn.Linear(128+256+128, 256),
                self.act,
                ResidualBlock(256),
                zero_module(nn.Linear(256, out_dim))
            )
        
        ''' rotation regress head '''
        self.fusion_tail_base_rot_x = make_head(3)
        self.fusion_tail_base_rot_y = make_head(3)
        ''' translation regress head '''
        self.fusion_tail_base_trans = make_head(3)
        ''' scale regress head '''
        self.fusion_tail_base_scale = make_head(1)
        ''' theta regress head '''
        self.fusion_theta = make_head(human_pose_dim)
            
        self.marginal_prob_func = marginal_prob_func


    def forward(self, data):
        '''
        Args:
            data, dict {
                'pts_feat': [bs, c]
                'pose_sample': [bs, pose_dim]
                't': [bs, 1]
            }
        '''
        
        #pts_feat = data['pts_feat']
        sampled_pose = data['sampled_pose']
        t = data['t']
        clip_feat = data["clip_feat"].squeeze()
        
        t_feat = self.t_encoder(t.squeeze(1))
        pose_feat = self.pose_encoder(sampled_pose)
        text_feat = self.prompt_encoder(clip_feat)
        
        total_feat = torch.cat([text_feat, t_feat, pose_feat], dim=-1)
        _, std = self.marginal_prob_func(total_feat, t)
        
        base_rot_x = self.fusion_tail_base_rot_x(total_feat)
        base_rot_y = self.fusion_tail_base_rot_y(total_feat)
        base_trans = self.fusion_tail_base_trans(total_feat)
        base_scale = self.fusion_tail_base_scale(total_feat)
        base_theta = self.fusion_theta(total_feat)
        out_score = torch.cat([base_rot_x, base_rot_y, base_trans, base_scale, base_theta], dim=-1) / (std+1e-7)

        return out_score


class HOIScoreNet_old(nn.Module):
    def __init__(self, marginal_prob_func, human_pose_dim=126):
        """_summary_

        Args:
            marginal_prob_func (func): marginal_prob_func of score network
            use_human_pose (bool): whether the model considers only the R, t, s of the person or its body pose as well

        Raises:
            NotImplementedError: _description_
        """
        super(HOIScoreNet_old, self).__init__()
        self.act = nn.ReLU(True)
        
        pose_dim = 10 + human_pose_dim


        ''' encode pose '''
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 256),
            #nn.LayerNorm(256),
            self.act,
            nn.Linear(256, 256),
            #nn.LayerNorm(256),
            self.act,
            nn.Linear(256, 256),
            self.act,
        )
        
        text_dim = 512
        self.prompt_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
            nn.Linear(256, 128),
            self.act,
        )
        
        ''' encode t '''
        self.t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            # self.act, # M4D26 update
            nn.Linear(128, 128),
            self.act,
        )

        ''' rotation_x_axis regress head '''
        self.fusion_tail_base_rot_x = nn.Sequential(
			#nn.Linear(128+256+1024, 256),
			nn.Linear(128+256+128, 256),
			# nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, 3)),
		)
        self.fusion_tail_base_rot_y = nn.Sequential(
			#nn.Linear(128+256+1024, 256),
			nn.Linear(128+256+128, 256),
			# nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, 3)),
		)
        ''' translation regress head '''
        self.fusion_tail_base_trans = nn.Sequential(
			#nn.Linear(128+256+1024, 256),
			nn.Linear(128+256+128, 256),
			# nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, 3)),
		)
        self.fusion_tail_base_scale = nn.Sequential(
			#nn.Linear(128+256+1024, 256),
			nn.Linear(128+256+128, 256),
			# nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
			self.act,
			nn.Linear(256, 256),
			self.act,
			zero_module(nn.Linear(256, 1)),
		)
        self.fusion_theta = nn.Sequential(
            nn.Linear(128+256+128, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
            zero_module(nn.Linear(256, human_pose_dim))
        )
            
        self.marginal_prob_func = marginal_prob_func


    def forward(self, data):
        '''
        Args:
            data, dict {
                'pts_feat': [bs, c]
                'pose_sample': [bs, pose_dim]
                't': [bs, 1]
            }
        '''
        
        #pts_feat = data['pts_feat']
        sampled_pose = data['sampled_pose']
        t = data['t']
        clip_feat = data["clip_feat"].squeeze()
        
        t_feat = self.t_encoder(t.squeeze(1))
        pose_feat = self.pose_encoder(sampled_pose)
        text_feat = self.prompt_encoder(clip_feat)
        
        total_feat = torch.cat([text_feat, t_feat, pose_feat], dim=-1)
        _, std = self.marginal_prob_func(total_feat, t)
        
        base_rot_x = self.fusion_tail_base_rot_x(total_feat)
        base_rot_y = self.fusion_tail_base_rot_y(total_feat)
        base_trans = self.fusion_tail_base_trans(total_feat)
        base_scale = self.fusion_tail_base_scale(total_feat)
        base_theta = self.fusion_theta(total_feat)
        out_score = torch.cat([base_rot_x, base_rot_y, base_trans, base_scale, base_theta], dim=-1) / (std+1e-7)

        return out_score


class PoseDecoderNet(nn.Module):
    def __init__(self, marginal_prob_func, sigma_data=1.4148, pose_mode='quat_wxyz', regression_head='RT'):
        """_summary_

        Args:
            marginal_prob_func (func): marginal_prob_func of score network
            pose_mode (str, optional): the type of pose representation from {'quat_wxyz', 'quat_xyzw', 'rot_matrix', 'euler_xyz'}. Defaults to 'quat_wxyz'.
            regression_head (str, optional): _description_. Defaults to 'RT'.

        Raises:
            NotImplementedError: _description_
        """
        super(PoseDecoderNet, self).__init__()
        self.sigma_data = sigma_data
        self.regression_head = regression_head
        self.act = nn.ReLU(True)
        pose_dim = 270

        ''' encode pose '''
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
        )
        
        ''' encode sigma(t) '''
        self.sigma_encoder = nn.Sequential(
            PositionalEmbedding(num_channels=128),
            nn.Linear(128, 128),
            self.act,
        )

        ''' fusion tail '''
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0) # init the final output layer's weights to zeros

        if self.regression_head == 'RT':
            self.fusion_tail = nn.Sequential(
                nn.Linear(128+256+1024, 512),
                self.act,
                Linear(512, pose_dim, **init_zero),
            )

        elif self.regression_head == 'R_and_T':
            ''' rotation regress head '''
            self.fusion_tail_rot = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                Linear(256, pose_dim - 3, **init_zero),
            )
            
            ''' tranalation regress head '''
            self.fusion_tail_trans = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                Linear(256, 3, **init_zero),
            )
            
        elif self.regression_head == 'Rx_Ry_and_T':
            if pose_mode != 'rot_matrix':
                raise NotImplementedError
            ''' rotation_x_axis regress head '''
            self.fusion_tail_rot_x = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                Linear(256, 3, **init_zero),
            )
            self.fusion_tail_rot_y = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                Linear(256, 3, **init_zero),
            )
            
            ''' tranalation regress head '''
            self.fusion_tail_trans = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                Linear(256, 3, **init_zero),
            )    
        
        else:
            raise NotImplementedError
            
        self.marginal_prob_func = marginal_prob_func


    def forward(self, data):
        '''
        Args:
            data, dict {
                'pts_feat': [bs, c]
                'pose_sample': [bs, pose_dim]
                't': [bs, 1]
            }
        '''

        pts_feat = data['pts_feat']
        sampled_pose = data['sampled_pose']
        t = data['t']
        _, sigma_t = self.marginal_prob_func(None, t) # \sigma(t) = t in EDM
        
        # determine scaling functions
        # EDM
        # c_skip = self.sigma_data ** 2 / (sigma_t ** 2 + self.sigma_data ** 2)
        # c_out = self.sigma_data * t / torch.sqrt(sigma_t ** 2 + self.sigma_data ** 2)
        # c_in = 1 / torch.sqrt(sigma_t ** 2 + self.sigma_data ** 2)
        # c_noise = torch.log(sigma_t) / 4
        # VE
        c_skip = 1
        c_out = sigma_t
        c_in = 1 
        c_noise = torch.log(sigma_t / 2)

        # comp total feat 
        sampled_pose_rescale = sampled_pose * c_in
        pose_feat = self.pose_encoder(sampled_pose_rescale)
        sigma_feat = self.sigma_encoder(c_noise.squeeze(1))
        total_feat = torch.cat([pts_feat, sigma_feat, pose_feat], dim=-1)
        
        if self.regression_head == 'RT':
            nn_output = self.fusion_tail(total_feat)
        elif self.regression_head == 'R_and_T':
            rot = self.fusion_tail_rot(total_feat)
            trans = self.fusion_tail_trans(total_feat)
            nn_output = torch.cat([rot, trans], dim=-1)
        elif self.regression_head == 'Rx_Ry_and_T':
            rot_x = self.fusion_tail_rot_x(total_feat)
            rot_y = self.fusion_tail_rot_y(total_feat)
            trans = self.fusion_tail_trans(total_feat)
            nn_output = torch.cat([rot_x, rot_y, trans], dim=-1)
        else:
            raise NotImplementedError
    
        denoised_output = c_skip * sampled_pose + c_out * nn_output
        return denoised_output

