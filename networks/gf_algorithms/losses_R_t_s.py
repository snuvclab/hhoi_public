import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_fn_edm(
        model, 
        data,
        marginal_prob_func, 
        sde_fn, 
        eps=1e-5,
        likelihood_weighting=False,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=1.4148,
        sigma_min=0.002,
        sigma_max=80,
    ):
    pts = data['zero_mean_pts']
    y = data['zero_mean_gt_pose']
    bs = pts.shape[0]

    # get noise n
    z = torch.randn_like(y) # [bs, pose_dim]
    # log_sigma_t = torch.randn([bs, 1], device=device) # [bs, 1]
    # sigma_t = (P_std * log_sigma_t + P_mean).exp() # [bs, 1]
    log_sigma_t = torch.rand([bs, 1], device=device) # [bs, 1]
    sigma_t = (math.log(sigma_min) + log_sigma_t * (math.log(sigma_max) - math.log(sigma_min))).exp() # [bs, 1]

    n = z * sigma_t

    perturbed_x = y + n # [bs, pose_dim]
    data['sampled_pose'] = perturbed_x
    data['t'] = sigma_t # t and sigma is interchangable in EDM
    data, output = model(data)    # [bs, pose_dim]
    
    # set_trace()
    
    # same as VE
    loss_ = torch.mean(torch.sum(((output * sigma_t + z)**2).view(bs, -1), dim=-1))

    return loss_


def loss_fn(
        model, 
        data,
        marginal_prob_func, 
        sde_fn, 
        eps=1e-5, 
        likelihood_weighting=False,
    ):
    #pts = data['zero_mean_pts']
    #gt_pose = data['zero_mean_gt_pose']
    gt_pose_scale = data['gt_pose_scale']
    
    ''' get std '''
    bs = gt_pose_scale.shape[0]
    random_t = torch.rand(bs, device=device) * (1. - eps) + eps         # [bs, ]
    random_t = random_t.unsqueeze(-1)                                   # [bs, 1]
    mu, std = marginal_prob_func(gt_pose_scale, random_t)               # [bs, pose_dim], [bs]
    std = std.view(-1, 1).to(device)
    mu = mu.to(device)                                               # [bs, 1]

    ''' perturb data and get estimated score '''
    z = torch.randn_like(gt_pose_scale).to(device)                                 # [bs, pose_dim]
    perturbed_x = mu + z * std                                          # [bs, pose_dim]
    data['sampled_pose'] = perturbed_x
    data['t'] = random_t
    estimated_score = model(data)                                 # [bs, pose_dim]

    ''' get target score '''
    # theoretic estimation
    target_score = - z * std / (std ** 2)
        
    ''' loss weighting '''
    #loss_weighting = std ** 2
    #loss_ = torch.mean(torch.sum((loss_weighting * (estimated_score - target_score)**2).view(bs, -1), dim=-1))
    #loss_ = torch.mean(torch.mean((loss_weighting * (estimated_score - target_score)**2).view(bs, -1), dim=-1))
    
    
    err = (std ** 2) * (estimated_score - target_score)**2
    
    loss_ = torch.mean(err.mean(dim=-1))
    
    return loss_


