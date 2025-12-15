import sys
import os
import torch
import numpy as np
import pytorch3d
import time

from scipy import integrate
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.genpose_utils import get_pose_dim
from utils.misc import normalize_rotation
from utils.metrics import get_rot_matrix

def global_prior_likelihood(z, sigma_max):
    """The likelihood of a Gaussian distribution with mean zero and 
        standard deviation sigma."""
    # z: [bs, pose_dim]
    shape = z.shape
    N = np.prod(shape[1:]) # pose_dim
    return -N / 2. * torch.log(2*np.pi*sigma_max**2) - torch.sum(z**2, dim=-1) / (2 * sigma_max**2)


def cond_ode_likelihood(
        score_model,
        data,
        prior,
        sde_coeff,
        marginal_prob_fn,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        num_steps=None,
        pose_mode='quat_wxyz', 
        init_x=None,
    ):
    
    pose_dim = get_pose_dim(pose_mode)
    batch_size = data['pts'].shape[0]
    epsilon = prior((batch_size, pose_dim)).to(device)
    init_x = data['sampled_pose'].clone().cpu().numpy() if init_x is None else init_x
    shape = init_x.shape
    init_logp = np.zeros((shape[0],)) # [bs]
    init_inp = np.concatenate([init_x.reshape(-1), init_logp], axis=0)
    
    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
        return score.cpu().numpy().reshape((-1,))

    def divergence_eval(data, epsilon):      
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        # save ckpt of sampled_pose
        origin_sampled_pose = data['sampled_pose'].clone()
        with torch.enable_grad():
            # make sampled_pose differentiable
            data['sampled_pose'].requires_grad_(True)
            score = score_model(data)
            score_energy = torch.sum(score * epsilon) # [, ]
            grad_score_energy = torch.autograd.grad(score_energy, data['sampled_pose'])[0] # [bs, pose_dim]
        # reset sampled_pose
        data['sampled_pose'] = origin_sampled_pose
        return torch.sum(grad_score_energy * epsilon, dim=-1) # [bs, 1]
    
    def divergence_eval_wrapper(data):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        with torch.no_grad(): 
            # Compute likelihood.
            div = divergence_eval(data, epsilon) # [bs, 1]
        return div.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    def ode_func(t, inp):        
        """The ODE function for use by the ODE solver."""
        # split x, logp from inp
        x = inp[:-shape[0]]
        logp = inp[-shape[0]:] # haha, actually we do not need use logp here
        # calc x-grad
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        x_grad = drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
        # calc logp-grad
        logp_grad = drift - 0.5 * (diffusion**2) * divergence_eval_wrapper(data)
        # concat curr grad
        return  np.concatenate([x_grad, logp_grad], axis=0)
  
    # Run the black-box ODE solver, note the 
    res = integrate.solve_ivp(ode_func, (eps, 1.0), init_inp, rtol=rtol, atol=atol, method='RK45')
    zp = torch.tensor(res.y[:, -1], device=device) # [bs * (pose_dim + 1)]
    z = zp[:-shape[0]].reshape(shape) # [bs, pose_dim]
    delta_logp = zp[-shape[0]:].reshape(shape[0]) # [bs,] logp
    _, sigma_max = marginal_prob_fn(None, torch.tensor(1.).to(device)) # we assume T = 1 
    prior_logp = global_prior_likelihood(z, sigma_max)
    log_likelihoods = (prior_logp + delta_logp) / np.log(2) # negative log-likelihoods (nlls)
    return z, log_likelihoods


def cond_pc_sampler(
        score_model, 
        data,
        prior,
        sde_coeff,
        num_steps=500, 
        snr=0.16,                
        device='cuda',
        eps=1e-5,
        pose_mode='quat_wxyz',
        init_x=None,
    ):
    
    pose_dim = get_pose_dim(pose_mode)
    batch_size = data['pts'].shape[0]
    init_x = prior((batch_size, pose_dim)).to(device) if init_x is None else init_x
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    noise_norm = np.sqrt(pose_dim) 
    x = init_x
    poses = []
    with torch.no_grad():
        for time_step in time_steps:      
            batch_time_step = torch.ones(batch_size, device=device).unsqueeze(-1) * time_step
            # Corrector step (Langevin MCMC)
            data['sampled_pose'] = x
            data['t'] = batch_time_step
            grad = score_model(data)
            grad_norm = torch.norm(grad.reshape(batch_size, -1), dim=-1).mean()
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)  

            # normalisation
            if pose_mode == 'quat_wxyz' or pose_mode == 'quat_xyzw':
                # quat, should be normalised
                x[:, :4] /= torch.norm(x[:, :4], dim=-1, keepdim=True)   
            elif pose_mode == 'euler_xyz':
                pass
            else:
                # rotation(x axis, y axis), should be normalised
                x[:, :3] /= torch.norm(x[:, :3], dim=-1, keepdim=True)
                x[:, 3:6] /= torch.norm(x[:, 3:6], dim=-1, keepdim=True)

            # Predictor step (Euler-Maruyama)
            drift, diffusion = sde_coeff(batch_time_step)
            drift = drift - diffusion**2*grad # R-SDE
            mean_x = x + drift * step_size
            x = mean_x + diffusion * torch.sqrt(step_size) * torch.randn_like(x)
            
            # normalisation
            x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
            poses.append(x.unsqueeze(0))
    
    xs = torch.cat(poses, dim=0)
    xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    mean_x[:, -3:] += data['pts_center']
    mean_x[:, :-3] = normalize_rotation(mean_x[:, :-3], pose_mode)
    # The last step does not include any noise
    return xs.permute(1, 0, 2), mean_x 


def cond_ode_sampler(
        score_model,
        data,
        prior,
        sde_coeff,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        T=1.0,
        num_steps=None,
        pose_mode='quat_wxyz', 
        denoise=True,
        init_x=None,
    ):
    pose_dim = get_pose_dim(pose_mode)
    batch_size=data['pts'].shape[0]
    init_x = prior((batch_size, pose_dim), T=T).to(device) if init_x is None else init_x + prior((batch_size, pose_dim), T=T).to(device)
    shape = init_x.shape
    
    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
        return score.cpu().numpy().reshape((-1,))
    
    def ode_func(t, x):      
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        return drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
  
    # Run the black-box ODE solver, note the 
    t_eval = None
    if num_steps is not None:
        # num_steps, from T -> eps
        t_eval = np.linspace(T, eps, num_steps)

    res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, pose_dim) # [num_steps, bs, pose_dim]
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape) # [bs, pose_dim]
    # denoise, using the predictor step in P-C sampler
    if denoise:
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
        drift, diffusion = sde_coeff(vec_eps)
        data['sampled_pose'] = x.float()
        data['t'] = vec_eps
        grad = score_model(data)
        drift = drift - diffusion**2*grad       # R-SDE
        mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
        x = mean_x
    
    num_steps = xs.shape[0]
    xs = xs.reshape(batch_size*num_steps, -1)
    xs[:, :-3] = normalize_rotation(xs[:, :-3], pose_mode)
    xs = xs.reshape(num_steps, batch_size, -1)
    xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
    x[:, -3:] += data['pts_center']
    return xs.permute(1, 0, 2), x

def cond_ode_sampler_for_R_t_s(
        score_model,
        data,
        prior,
        sde_coeff,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        T=1.0,
        num_steps=None,
        pose_mode='quat_wxyz',
        regression_mode='Rx_Ry_T_S_both',
        denoise=True,
        init_x=None,
    ):
    if score_model.cfg.model_type == "hoi":
        pose_dim = 10 + score_model.cfg.human_pose_dim # 10 = rot(6) + transl(3) + scale(1)
    elif score_model.cfg.model_type == "hhi":
        pose_dim = 9 + 2 * score_model.cfg.human_pose_dim # 9 = rot(6) + transl(3)
    else:
        raise NotImplementedError

    batch_size=data['gt_pose_scale'].shape[0]
    init_x = prior((batch_size, pose_dim), T=T).to(device) if init_x is None else init_x + prior((batch_size, pose_dim), T=T).to(device)
    shape = init_x.shape
    
    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
        return score.cpu().numpy().reshape((-1,))
    
    def ode_func(t, x):      
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        return drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
  
    # Run the black-box ODE solver, note the 
    t_eval = None
    if num_steps is not None:
        # num_steps, from T -> eps
        t_eval = np.linspace(T, eps, num_steps)

    res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, pose_dim) # [num_steps, bs, pose_dim]
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape) # [bs, pose_dim]
    # denoise, using the predictor step in P-C sampler
    if denoise:
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
        drift, diffusion = sde_coeff(vec_eps)
        data['sampled_pose'] = x.float()
        data['t'] = vec_eps
        grad = score_model(data)
        drift = drift - diffusion**2*grad       # R-SDE
        mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
        x = mean_x
    
    #num_steps = xs.shape[0]
    #xs = xs.reshape(batch_size*num_steps, -1)
    #xs[:, :6] = normalize_rotation(xs[:, :6], pose_mode)
    #xs[:, 12:18] = normalize_rotation(xs[:, 12:18], pose_mode)
    #xs = xs.reshape(num_steps, batch_size, -1)
    #xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    #x[:, :6] = normalize_rotation(x[:, :6], pose_mode)
    #x[:, 12:18] = normalize_rotation(x[:, 12:18], pose_mode)
    #x[:, -3:] += data['pts_center']

    return xs.permute(1, 0, 2), x


def cond_ode_sampler_for_hhoi_naive(
        score_model,
        data,
        prior,
        sde_coeff,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        T=1.0,
        num_steps=None,
        pose_mode='quat_wxyz',
        regression_mode='Rx_Ry_T_S_both',
        denoise=True,
        init_x=None,
    ):

    human_pose_dim = score_model.cfg.human_pose_dim
    hoi_pose_dim = 10 + human_pose_dim # 10 = rot(6) + transl(3) + scale(1)
    hhi_pose_dim = 9 + 2 * human_pose_dim # 9 = rot(6) + transl(3)

    batch_size=data['gt_pose_scale'].shape[0]
    hoi_init_x = prior((batch_size, hoi_pose_dim), T=T).to(device)
    hhi_init_x = prior((batch_size, hhi_pose_dim), T=T).to(device)
    hoi_shape = hoi_init_x.shape
    hhi_shape = hhi_init_x.shape
    
    def hoi_ode_func(t, x):      
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, hoi_pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['hoi_sampled_pose'] = x
        data['t'] = time_steps
        with torch.no_grad():
            score = score_model(data, mode='score_hoi_for_hhoi')
        return drift - 0.5 * (diffusion**2) * score.cpu().numpy().reshape((-1,))
  
    # Run the black-box ODE solver, note the 
    t_eval = None
    if num_steps is not None:
        # num_steps, from T -> eps
        t_eval = np.linspace(T, eps, num_steps)

    res = integrate.solve_ivp(hoi_ode_func, (T, eps), hoi_init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    hoi_xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, hoi_pose_dim) # [num_steps, bs, hoi_pose_dim]
    hoi_x = torch.tensor(res.y[:, -1], device=device).reshape(hoi_shape) # [bs, hoi_pose_dim]
    # denoise, using the predictor step in P-C sampler
    if denoise:
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones((hoi_x.shape[0], 1), device=hoi_x.device) * eps
        drift, diffusion = sde_coeff(vec_eps)
        data['hoi_sampled_pose'] = hoi_x.float()
        data['t'] = vec_eps
        grad = score_model(data, mode='score_hoi_for_hhoi')
        drift = drift - diffusion**2*grad       # R-SDE
        mean_hoi_x = hoi_x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
        hoi_x = mean_hoi_x
    
    hoi_xs = hoi_xs.permute(1, 0, 2)

    
    hhi_init_x[:, :human_pose_dim] = hoi_x[:, -human_pose_dim:]
    
    def hhi_ode_func(t, x):      
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, hhi_pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['hhi_sampled_pose'] = x
        data['t'] = time_steps
        with torch.no_grad():
            score = score_model(data, mode='score_hhi_for_hhoi')

        dx = (drift - 0.5 * (diffusion**2) * score.cpu().numpy()).reshape(batch_size, hhi_pose_dim)
        dx[:, :human_pose_dim] = 0
        
        return drift - 0.5 * (diffusion**2) * score.cpu().numpy().reshape((-1,))
    
    res = integrate.solve_ivp(hhi_ode_func, (T, eps), hhi_init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    hhi_xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, hhi_pose_dim) # [num_steps, bs, hhi_pose_dim]
    hhi_x = torch.tensor(res.y[:, -1], device=device).reshape(hhi_shape) # [bs, hhi_pose_dim]
    if denoise:
        vec_eps = torch.ones((hhi_x.shape[0], 1), device=hhi_x.device) * eps
        drift, diffusion = sde_coeff(vec_eps)
        data['hhi_sampled_pose'] = hhi_x.float()
        data['t'] = vec_eps
        grad = score_model(data, mode='score_hhi_for_hhoi')
        drift = drift - diffusion**2*grad       # R-SDE
        mean_hhi_x = hhi_x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
        hhi_x = mean_hhi_x
        hhi_x[:, :human_pose_dim] = hoi_x[:, -human_pose_dim:]
        
    hhi_xs = hhi_xs.permute(1, 0, 2)

    return hoi_x, hhi_x

def cond_ode_sampler_for_hhoi_exp(
        score_model,
        data,
        prior,
        sde_coeff,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        T=1.0,
        num_steps=None,
        pose_mode='quat_wxyz',
        regression_mode='Rx_Ry_T_S_both',
        denoise=True,
        init_x=None,
    ):

    human_pose_dim = score_model.cfg.human_pose_dim
    hoi_pose_dim = 10 + human_pose_dim # 10 = rot(6) + transl(3) + scale(1)
    hhi_pose_dim = 9 + 2 * human_pose_dim # 9 = rot(6) + transl(3)

    num_human = 2
    batch_size=data['gt_pose_scale'].shape[0]
    h1oi_init_x = prior((batch_size, hoi_pose_dim), T=T).to(device)
    h2oi_init_x = prior((batch_size, hoi_pose_dim), T=T).to(device)
    hhi_init_x = prior((batch_size, hhi_pose_dim), T=T).to(device)
    full_init_x = torch.cat([h1oi_init_x, h2oi_init_x, hhi_init_x], dim=-1) # (B, hoi_pose_dim*2 + hhi_pose_dim)
    full_shape = full_init_x.shape
    
    def inconsistency_loss(x, time_step):
        inc_loss = 0
        
        h1oi_sampled_pose = x[:, :hoi_pose_dim]                     # (B, hoi_pose_dim) : R1(6), t1(3), s1(1), theta1(human_pose_dim)
        h2oi_sampled_pose = x[:, hoi_pose_dim:2*hoi_pose_dim]       # (B, hoi_pose_dim) : R2(6), t2(3), s2(1), theta2(human_pose_dim)
        hhi_sampled_pose = x[:, -hhi_pose_dim:]                     # (B, hhi_pose_dim) : theta'1(human_pose_dim), R(6), t(3), theta'2(human_pose_dim)
        
        # 1. theta1 = theta'1
        inc_loss += torch.mean((h1oi_sampled_pose[:, -human_pose_dim:] - hhi_sampled_pose[:, :human_pose_dim]) ** 2)
        # 2. theta2 = theta'2
        inc_loss += torch.mean((h2oi_sampled_pose[:, -human_pose_dim:] - hhi_sampled_pose[:, -human_pose_dim:]) ** 2)
        # 3. s1 = s2
        inc_loss += torch.mean((h1oi_sampled_pose[:, 9] - h2oi_sampled_pose[:, 9]) ** 2)
        # 4. s2R2 @ X2 + t2 = s1R1@R @ X2 + s1R1t + t1 ==> R2 = R1@R, t2 = s1R1@t + t1
        R1 = get_rot_matrix(h1oi_sampled_pose[:, :6], pose_mode).permute(0, 2, 1)
        R2 = get_rot_matrix(h2oi_sampled_pose[:, :6], pose_mode).permute(0, 2, 1)
        R = get_rot_matrix(hhi_sampled_pose[:, human_pose_dim:human_pose_dim+6], pose_mode).permute(0, 2, 1)
        
        inc_loss += torch.mean((h2oi_sampled_pose[:, :6] - pytorch3d.transforms.matrix_to_rotation_6d(torch.matmul(R1, R))) ** 2)
        transl_ = h1oi_sampled_pose[:, 9:10] * (R1 @ hhi_sampled_pose[:, human_pose_dim+6:human_pose_dim+6+3].unsqueeze(-1)).squeeze(-1) + h1oi_sampled_pose[:, 6:9]
        inc_loss += torch.mean((h2oi_sampled_pose[:, 6:9] - transl_) ** 2)
        
        weight_term = (1/time_step**2) if (1/time_step**2) <= 1000 else 1000
        return inc_loss * weight_term * 100
    
    def ode_func(t, x):      
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, full_shape[1]), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['h1oi_sampled_pose'] = x[:, :hoi_pose_dim]
        data['h2oi_sampled_pose'] = x[:, hoi_pose_dim:2*hoi_pose_dim]
        data['hhi_sampled_pose'] = x[:, -hhi_pose_dim:]
        data['t'] = time_steps
        
        with torch.no_grad():
            score = score_model(data, mode='score_hhoi_exp')
            
        with torch.enable_grad():
            x_clone = x.clone().detach().requires_grad_(True)
            total_loss = 0
            
            use_inconsistency_loss = True
            use_collision_loss = False
            time_thres = 0.5
            if use_inconsistency_loss and t <= time_thres:
                inc_loss = inconsistency_loss(x_clone, t)
                if inc_loss > 0.0:
                    total_loss += inc_loss
                    
            if total_loss > 0.0:
                grad_all = torch.autograd.grad(-total_loss, x_clone, retain_graph=False)[0]
            else:
                grad_all = torch.zeros_like(score).to(score.device)
            
            x_clone.requires_grad_(False)
            
        return drift - 0.5 * (diffusion**2) * score.cpu().numpy().reshape((-1,)) - grad_all.cpu().numpy().reshape((-1,))
  
    # Run the black-box ODE solver, note the 
    t_eval = None
    if num_steps is not None:
        # num_steps, from T -> eps
        t_eval = np.linspace(T, eps, num_steps)

    res = integrate.solve_ivp(ode_func, (T, eps), full_init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, full_shape[1]) # [num_steps, bs, hoi_pose_dim*2 + hhi_pose_dim]
    x = torch.tensor(res.y[:, -1], device=device).reshape(full_shape) # [bs, hoi_pose_dim*2 + hhi_pose_dim]
    # denoise, using the predictor step in P-C sampler
    if denoise:
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
        drift, diffusion = sde_coeff(vec_eps)
        data['h1oi_sampled_pose'] = x[:, :hoi_pose_dim].float()
        data['h2oi_sampled_pose'] = x[:, hoi_pose_dim:2*hoi_pose_dim].float()
        data['hhi_sampled_pose'] = x[:, -hhi_pose_dim:].float()
        data['t'] = vec_eps
        grad = score_model(data, mode='score_hhoi_exp')
        drift = drift - diffusion**2*grad       # R-SDE
        mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
        x = mean_x
    
    xs = xs.permute(1, 0, 2)

    return xs, x

def cond_ode_sampler_for_hhoi(
        score_model,
        data,
        prior,
        sde_coeff,
        atol=1e-5,
        rtol=1e-5,
        device='cuda',
        eps=1e-5,
        T=1.0,
        num_steps=None,
        pose_mode='quat_wxyz',
        regression_mode='Rx_Ry_T_S_both',
        denoise=True,
        init_x=None,
        use_inconsistency_loss=False,
        use_collision_loss=False,
        time_thres=0.5,
        use_exact_collision=False,
        solver_type='cpu',
        additional_loss_step=1,
        sampling_steps_for_gpu_solver=2,
        inconsistency_loss_weight=100,
        collision_loss_weight=1600
    ):

    human_pose_dim = score_model.cfg.human_pose_dim
    hoi_pose_dim = 10 + human_pose_dim # 10 = rot(6) + transl(3) + scale(1)
    hhi_pose_dim = 9 + 2 * human_pose_dim # 9 = rot(6) + transl(3)

    num_hoi = data['num_hoi']
    num_hhi = data['num_hhi']
    human_list = data["human_list"]
    sorted_human_list = data["sorted_human_list"]
    hhi_pair_list = data["hhi_pair_list"]
    human_non_adjacent_info = data["human_non_adjacent_info"]
    batch_size = data['gt_pose_scale'].shape[0]
    
    if use_inconsistency_loss and use_collision_loss:
        from train_human_pose_enc_dec import EncoderDecoder
        import smplx
        pose_model = EncoderDecoder().to(score_model.cfg.device)
        checkpoint = torch.load('enc_dec_human_pose/encoder_decoder_latest.pth')
        pose_model.load_state_dict(checkpoint['model_state_dict'])
        pose_model.eval()
        
        smplx_model = smplx.create(model_path=".", model_type='smplx', num_expression_coeffs = 100).to(score_model.cfg.device)
        
        if use_exact_collision:
            beta = torch.zeros((batch_size, 10)).to(score_model.cfg.device)
            expression      = torch.zeros(batch_size, 100, device=score_model.cfg.device)
            jaw_pose        = torch.zeros(batch_size, 3,  device=score_model.cfg.device)
            left_hand_pose  = torch.zeros(batch_size, 6, device=score_model.cfg.device)
            right_hand_pose = torch.zeros(batch_size, 6, device=score_model.cfg.device)
            transl          = torch.zeros(batch_size, 3,  device=score_model.cfg.device)
            leye_pose = torch.zeros(batch_size, 3, device=score_model.cfg.device)
            reye_pose = torch.zeros(batch_size, 3, device=score_model.cfg.device)
        else:
            J_rest = (smplx_model.J_regressor @ smplx_model.v_template)  # (J,3)
            parent = smplx_model.parents.long() # (J,)
            offsets = (J_rest - J_rest[parent]).to(device)    # (J,3) bone vectors in rest pose
            offsets[0] = J_rest[0] # root has no parent
            parent  = parent.to(device) # (J,)
            
            # joint bbox --> smplx mesh vertices bbox
            trans_correction = torch.tensor([0.0029090494,  0.1257648468, -0.0011374354]).to(device)
            scale_correction = torch.tensor([1.1856606007, 1.1229841709, 1.3754303455]).to(device)
            
            # constant for capsule
            child_idx = torch.arange(1, 22, device=device)
            radius_factor = torch.tensor([1.3448601,  1.2154156,  1.3578306,  0.2410832,  0.24921997, 1.1963989,
                                  0.1288751,  0.13837513, 2.560507,   0.24903344, 0.26309416, 0.81440395,
                                  1.5429087,  1.516054,   0.4715843,  0.64832884, 0.7963264,  0.17275932,
                                  0.18719819, 0.12624489, 0.14348947], device=device)
            radius_factor_for_head_hand = torch.tensor([0.5336204,  0.16777988, 0.17133683], device=device)
    
    full_init_x = []
    for i in range(num_hoi):
        full_init_x.append(prior((batch_size, hoi_pose_dim), T=T).to(device))
    for i in range(num_hhi):
        full_init_x.append(prior((batch_size, hhi_pose_dim), T=T).to(device))
        
    full_init_x = torch.cat(full_init_x, dim=-1) # (B, hoi_pose_dim*num_hoi + hhi_pose_dim*num_hhi)
    full_shape = full_init_x.shape
    
    def zero_variance_loss(x):
        # x: (B, N, F)
        mean_per_batch = x.mean(dim=1, keepdim=True)                  # (B, 1, F)
        variance_per_batch = ((x - mean_per_batch) ** 2).mean(dim=1)  # (B, F)
        loss = variance_per_batch.mean()                              # scalar
        return loss
    
    def inconsistency_loss(x, time_step):
        # x: (B, hoi_pose_dim*num_hoi + hhi_pose_dim*num_hhi)
        ######################## Simple Case(two hoi, one hhi) ########################
        # h1oi_sampled_pose    # (B, hoi_pose_dim) : R1(6), t1(3), s1(1), theta1(human_pose_dim)
        # h2oi_sampled_pose    # (B, hoi_pose_dim) : R2(6), t2(3), s2(1), theta2(human_pose_dim)
        # hhi_sampled_pose     # (B, hhi_pose_dim) : theta'1(human_pose_dim), R(6), t(3), theta'2(human_pose_dim)
        # 
        # 1. theta1 = theta'1
        # 2. theta2 = theta'2
        # 3. s1 = s2
        # 4. s2R2 @ X2 + t2 = s1R1@R @ X2 + s1R1t + t1 ==> R2 = R1@R, t2 = s1R1@t + t1
        ###############################################################################
        inc_loss = 0
        
        # 1. var(s) = 0
        indices = torch.arange(num_hoi) * hoi_pose_dim + 9
        s = x[:, indices] # (B, num_hoi)
        inc_loss += zero_variance_loss(s.unsqueeze(-1)) * num_hoi
        
        # 2. var(theta_i) = 0
        human_final_pose_dict = {}
        for idx, human in enumerate(human_list):
            theta_list = []
            for hhi_pair_idx, (base_human, target_human) in enumerate(hhi_pair_list):
                if human == base_human:
                    theta_i_prime = x[:, full_shape[1]-hhi_pose_dim*(num_hhi-hhi_pair_idx):full_shape[1]-hhi_pose_dim*(num_hhi-hhi_pair_idx)+human_pose_dim] # (B, human_pose_dim)
                    theta_list.append(theta_i_prime)
                elif human == target_human:
                    theta_i_prime = x[:, full_shape[1]-hhi_pose_dim*(num_hhi-1-hhi_pair_idx)-human_pose_dim:full_shape[1]-hhi_pose_dim*(num_hhi-1-hhi_pair_idx)] # (B, human_pose_dim)
                    theta_list.append(theta_i_prime)
                else:
                    pass
                
            if len(theta_list) == 0:
                human_final_pose_dict[human] = x[:, hoi_pose_dim*(idx+1)-human_pose_dim:hoi_pose_dim*(idx+1)]
                continue
            
            theta_i = x[:, hoi_pose_dim*(idx+1)-human_pose_dim:hoi_pose_dim*(idx+1)] # (B, human_pose_dim)
            theta_list.append(theta_i)
            
            theta_stacked = torch.stack(theta_list, dim=1) # (B, n, human_pose_dim)
            inc_loss += zero_variance_loss(theta_stacked) * theta_stacked.shape[1]
            
            human_final_pose_dict[human] = theta_stacked.mean(dim=1)
        
        # 3. s2R2 @ X2 + t2 = s1R1@R @ X2 + s1R1t + t1 ==> R2 = R1@R, t2 = s1R1@t + t1
        human_final_rts_dict = {}
        for human, orig_idx in sorted_human_list:
            human_final_rts_dict[human] = []
            human_rts = x[:, hoi_pose_dim*orig_idx:hoi_pose_dim*orig_idx+10] # (B, 10) : R2(6), t2(3), s2(1)
            human_final_rts_dict[human].append(human_rts)

            for hhi_pair_idx, (base_human, target_human) in enumerate(hhi_pair_list):
                if human == target_human:
                    h2_rel_rt = x[:, full_shape[1]-hhi_pose_dim*(num_hhi-hhi_pair_idx)+human_pose_dim:full_shape[1]-hhi_pose_dim*(num_hhi-hhi_pair_idx)+human_pose_dim+9] # (B, 9) : R(6), t(3)
                    R = get_rot_matrix(h2_rel_rt[:, :6], pose_mode).permute(0, 2, 1)
                    t = h2_rel_rt[:, 6:9]
                    
                    R1 = get_rot_matrix(human_final_rts_dict[base_human][:, :6], pose_mode).permute(0, 2, 1)
                    R_ = pytorch3d.transforms.matrix_to_rotation_6d(torch.matmul(R1, R)) # (B, 6)
                    t_ = human_final_rts_dict[base_human][:, 9:10] * (R1 @ t.unsqueeze(-1)).squeeze(-1) + human_final_rts_dict[base_human][:, 6:9] # (B, 3): s1R1@t + t1

                    human_rts_ = torch.cat([R_, t_, human_rts[:, 9:10]], dim=-1) # (B, 10)
                    human_final_rts_dict[human].append(human_rts_)
            
                    
            rts_stacked = torch.stack(human_final_rts_dict[human], dim=1) # (B, n, 10)
            if rts_stacked.shape[1] > 1:
                inc_loss += zero_variance_loss(rts_stacked[:, :, :6]) * rts_stacked.shape[1]   # For R
                inc_loss += zero_variance_loss(rts_stacked[:, :, 6:9]) * rts_stacked.shape[1]  # For t
                    
            human_final_rts_dict[human] = rts_stacked.mean(dim=1) # (B, 10) : R2(6), t2(3), s2(1)

        
        weight_term = min(1000, 1/time_step**2)
        return inc_loss * weight_term * inconsistency_loss_weight, human_final_rts_dict, human_final_pose_dict
    
    def collision_loss_with_capsule(rts_dict, pose_dict, time_step):
        def fk_joints(rot_mats, offsets):
            B, J = rot_mats.shape[:2]
            R_ws, Ps = [], []

            for j in range(J):
                if j == 0:
                    Rj = rot_mats[:, 0]
                    Pj = offsets[0].unsqueeze(0).expand(B, -1)
                else:
                    p  = parent[j]
                    Rj = R_ws[p] @ rot_mats[:, j]
                    Pj = Ps[p]  + (R_ws[p] @ offsets[j]).squeeze(-1)
                R_ws.append(Rj)
                Ps.append(Pj)

            pos = torch.stack(Ps,  dim=1)    # (B,J,3)
            return pos
        
        def get_capsules(joints, parent):
            """
            joints: torch.Tensor(B, 22, 3)
            parent: torch.LongTensor(J,)
            """

            p0 = joints[:, parent[child_idx], :]   # (B, 21, 3)
            p1 = joints[:, child_idx, :]           # (B, 21, 3)

            vec    = p1 - p0                        # (B, 21, 3)
            length = vec.norm(dim=-1)               # (B, 21)

            radius = length * radius_factor        # (B, 21)
            
            head_wrist_joints = [15, 20, 21] #  head, left_wrist, right_wrist
            head_wrist_parents = parent[head_wrist_joints]                    # (3,)
            head_wrist_c = joints[:, head_wrist_joints, :]              # (B, 3, 3)

            head_wrist_vec = joints[:, head_wrist_joints, :] - joints[:, head_wrist_parents, :]
            
            head_wrist_r = head_wrist_vec.norm(dim=-1) * radius_factor_for_head_hand     # (B, 3)
            
            return torch.cat([p0, head_wrist_c - head_wrist_vec / 2], dim=1), torch.cat([p1, head_wrist_c + head_wrist_vec / 2], dim=1), torch.cat([radius, head_wrist_r], dim=1)
        
        def capsule_batch_penetration(p0_a, p1_a, r_a, p0_b, p1_b, r_b, eps=1e-8):
            """
            p0_a, p1_a: torch.Tensor, (B, N1, 3)  
            r_a       : torch.Tensor, (B, N1)     
            p0_b, p1_b: torch.Tensor, (B, N2, 3)  
            r_b       : torch.Tensor, (B, N2)    
            """

            u = (p1_a - p0_a).unsqueeze(2)           # (B, N1, 1, 3)
            v = (p1_b - p0_b).unsqueeze(1)           # (B, 1, N2, 3)
            w0 = p0_a.unsqueeze(2) - p0_b.unsqueeze(1)  # (B, N1, N2, 3)

            # dot products
            a = torch.sum(u*u, dim=-1)               # (B, N1, N2)
            b = torch.sum(u*v, dim=-1)
            c = torch.sum(v*v, dim=-1)
            d = torch.sum(u*w0, dim=-1)
            e = torch.sum(v*w0, dim=-1)

            denom = a*c - b*b                         # (B, N1, N2)
            denom = denom.clamp(min=eps)

            # On infinite straight lines
            s = ( b*e - c*d ) / denom
            t = ( a*e - b*d ) / denom
            # clamp
            s = s.clamp(0.0, 1.0)
            t = t.clamp(0.0, 1.0)

            p_sc = p0_a.unsqueeze(2) + s.unsqueeze(-1) * u  # (B, N1, N2, 3)
            q_tc = p0_b.unsqueeze(1) + t.unsqueeze(-1) * v  # (B, N1, N2, 3)

            dist = (p_sc - q_tc).norm(dim=-1)               # (B, N1, N2)

            rad_sum = r_a.unsqueeze(2) + r_b.unsqueeze(1)  # (B, N1, N2)

            pen = rad_sum - dist
            pen = pen.clamp(min=0.0)

            return pen.mean()
        
        col_loss = 0.0

        joints_cache = {}
        for human in pose_dict:
            # (a) local joint rotations -----------------------------
            human_pose = pose_dict[human] # (B, human_pose_dim)
            human_pose_dec = pose_model.decoder(human_pose)
            pose_dict[human] = human_pose_dec

        # (b) split back per human & FK -----------------------------
        for human in pose_dict:
            local = pytorch3d.transforms.rotation_6d_to_matrix(pose_dict[human].view(-1,6)).view(-1,21,3,3) # (B,21,3,3)
            global_R = pytorch3d.transforms.rotation_6d_to_matrix(rts_dict[human][:,:6]) # (B,3,3)
            I_root = torch.eye(3, device=local.device).unsqueeze(0).repeat(local.shape[0], 1, 1).unsqueeze(1) # (B,1,3,3)
            
            joints_cano = fk_joints(torch.cat([I_root, local], dim=1), offsets[:22])      # (B,22,3)
            
            # smplx(R, \theta) = R(smplx(I, \theta) - J0) + J0
            joints = (global_R.unsqueeze(1) @ (joints_cano - offsets[0]).unsqueeze(-1)).squeeze(-1) + offsets[0] # (B, 22, 3)

            s = rts_dict[human][:,9:10]; t = rts_dict[human][:,6:9] # scale, transl
            joints = joints * s.unsqueeze(1) + t.unsqueeze(1)
            joints_cache[human] = joints

        # (c) IoU ----------------------------------------------------
        for h1, h2 in human_non_adjacent_info:
            j1, j2 = joints_cache[h1], joints_cache[h2]
            
            p0_a, p1_a, r_a = get_capsules(j1, parent)
            p0_b, p1_b, r_b = get_capsules(j2, parent)
            
            pen = capsule_batch_penetration(p0_a, p1_a, r_a, p0_b, p1_b, r_b)
            col_loss += pen

        weight_term = min(1000, 1/time_step**2)
        return col_loss * weight_term * collision_loss_weight
    
    def collision_loss(rts_dict, pose_dict, time_step):
        def fk_joints(rot_mats, offsets):
            B, J = rot_mats.shape[:2]
            R_ws, Ps = [], []

            for j in range(J):
                if j == 0:
                    Rj = rot_mats[:, 0]
                    Pj = offsets[0].unsqueeze(0).expand(B, -1)
                else:
                    p  = parent[j]
                    Rj = R_ws[p] @ rot_mats[:, j]
                    Pj = Ps[p]  + (R_ws[p] @ offsets[j]).squeeze(-1)
                R_ws.append(Rj)
                Ps.append(Pj)

            pos = torch.stack(Ps,  dim=1)    # (B,J,3)
            return pos
        
        col_loss = 0.0

        joints_cache = {}
        for human in pose_dict:
            # (a) local joint rotations -----------------------------
            human_pose = pose_dict[human] # (B, human_pose_dim)
            human_pose_dec = pose_model.decoder(human_pose)
            pose_dict[human] = human_pose_dec

        # (b) split back per human & FK -----------------------------
        for human in pose_dict:
            local = pytorch3d.transforms.rotation_6d_to_matrix(pose_dict[human].view(-1,6)).view(-1,21,3,3) # (B,21,3,3)
            global_R = pytorch3d.transforms.rotation_6d_to_matrix(rts_dict[human][:,:6]) # (B,3,3)
            I_root = torch.eye(3, device=local.device).unsqueeze(0).repeat(local.shape[0], 1, 1).unsqueeze(1) # (B,1,3,3)
            
            joints_cano = fk_joints(torch.cat([I_root, local], dim=1), offsets[:22])      # (B,22,3)
            joints_cano = joints_cano * scale_correction + trans_correction               # (B,22,3)
            
            # smplx(R, \theta) = R(smplx(I, \theta) - J0) + J0
            joints = (global_R.unsqueeze(1) @ (joints_cano - offsets[0]).unsqueeze(-1)).squeeze(-1) + offsets[0] # (B, 22, 3)

            s = rts_dict[human][:,9:10]; t = rts_dict[human][:,6:9]
            joints = joints * s.unsqueeze(1) + t.unsqueeze(1)              # scale, transl
            joints_cache[human] = joints

        # (c) IoU ----------------------------------------------------
        for h1, h2 in human_non_adjacent_info:
            j1, j2 = joints_cache[h1], joints_cache[h2]
            min1, max1 = j1.min(1).values, j1.max(1).values
            min2, max2 = j2.min(1).values, j2.max(1).values
            
            inter_min, inter_max = torch.max(min1,min2), torch.min(max1,max2)
            inter_dim = (inter_max - inter_min).clamp(min=0)
            inter_vol = inter_dim.prod(1)
            union_vol = (max1-min1).prod(1) + (max2-min2).prod(1) - inter_vol
            col_loss += (inter_vol/(union_vol+1e-8)).mean()

        weight_term = min(1000, 1/time_step**2)
        return col_loss * weight_term * 10
    
    def collision_loss_exact(rts_dict, pose_dict, time_step):
        col_loss = 0
        
        human_vert_dict = {}
        human_joint_dict = {}
        for human in pose_dict.keys():
            human_pose = pose_dict[human] # (B, human_pose_dim)
            human_pose_dec = pose_model.decoder(human_pose)
            human_pose_list = []
            for i in range(21):
                nx_pose = human_pose_dec[:, 6 * i : 6 * i + 6]
                
                nx_mat = pytorch3d.transforms.rotation_6d_to_matrix(nx_pose) # (B, 3, 3)
                nx_mat = pytorch3d.transforms.matrix_to_axis_angle(nx_mat) # (B, 3)
                human_pose_list.append(nx_mat)
            
            human_pose = torch.cat(human_pose_list, axis=-1) # (B, 63)
                    
            human_orient = rts_dict[human][:, :6]
            human_orient = pytorch3d.transforms.rotation_6d_to_matrix(human_orient) # (B, 3, 3)
            human_orient = pytorch3d.transforms.matrix_to_axis_angle(human_orient) # (B, 3)
            human_transl = rts_dict[human][:, 6:9] # (B, 3)
            human_scale = rts_dict[human][:, 9:10] # (B, 1)
            
            human_vert = smplx_model(global_orient=human_orient, body_pose=human_pose, betas=beta, expression=expression, jaw_pose=jaw_pose, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=transl, leye_pose=leye_pose, reye_pose=reye_pose).vertices
            #human_joint = smplx_model(global_orient=human_orient, body_pose=human_pose, betas=beta, expression=expression, jaw_pose=jaw_pose, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=transl, leye_pose=leye_pose, reye_pose=reye_pose).joints
            
            human_vert = human_vert *  human_scale[:, None] + human_transl[:, None, :]
            #human_joint = human_joint *  human_scale[:, None] + human_transl[:, None, :]
            human_vert_dict[human] = human_vert
            #human_joint_dict[human] = human_joint
            
        for h1, h2 in human_non_adjacent_info:
            verts1 = human_vert_dict[h1]
            verts2 = human_vert_dict[h2]
            #verts1 = human_joint_dict[h1]
            #verts2 = human_joint_dict[h2]
            
            min1 = verts1.min(dim=1).values    # (B,3)
            max1 = verts1.max(dim=1).values    # (B,3)
            min2   = verts2.min(dim=1).values  # (B,3)
            max2   = verts2.max(dim=1).values  # (B,3)
            
            inter_min = torch.max(min1, min2)  # (B,3)
            inter_max = torch.min(max1, max2)  # (B,3)
            inter_dim = (inter_max - inter_min).clamp(min=0)  # (B,3)
            inter_vol = inter_dim.prod(dim=1)                 # (B,)
            
            vol1 = ((max1 - min1).clamp(min=0)).prod(dim=1)  # (B,)
            vol2 = ((max2 - min2).clamp(min=0)).prod(dim=1)  # (B,)
            union_vol = vol1 + vol2 - inter_vol              # (B,)
            
            iou  = inter_vol / (union_vol + 1e-8)
            col_loss += iou.mean()
            
        weight_term = min(1000, 1/time_step**2)
        return col_loss * weight_term * 10
    
    step_info = {"i": 0, "k": additional_loss_step}
    def ode_func(t, x):      
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, full_shape[1]), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        
        with torch.no_grad():
            score = score_model(data, mode='score_hhoi')
            
        with torch.enable_grad():
            x_clone = x.clone().detach().requires_grad_(True)
            total_loss = 0
            
            if use_inconsistency_loss and t <= time_thres and step_info["i"] % step_info["k"] == 0:
                inc_loss, human_final_rts_dict, human_final_pose_dict = inconsistency_loss(x_clone, t)
                if inc_loss > 0.0:
                    total_loss += inc_loss

                if use_collision_loss:
                    if use_exact_collision:
                        col_loss = collision_loss_exact(human_final_rts_dict, human_final_pose_dict, t)
                    else:
                        col_loss = collision_loss(human_final_rts_dict, human_final_pose_dict, t)
                    if col_loss > 0.0:
                        total_loss += col_loss
                    
            if total_loss > 0.0:
                grad_all = torch.autograd.grad(-total_loss, x_clone, retain_graph=False)[0]
            else:
                grad_all = torch.zeros_like(score).to(score.device)
            
            x_clone.requires_grad_(False)
        
        step_info["i"] += 1
            
        return drift - 0.5 * (diffusion**2) * score.cpu().numpy().reshape((-1,)) - grad_all.cpu().numpy().reshape((-1,))
    
    def ode_func_torch(t, x):
        """The ODE function for use by the ODE solver."""
        drift, diffusion = sde_coeff(t.expand(x.shape[0]))
        data['sampled_pose'] = x
        data['t'] = t.expand(x.shape[0]).unsqueeze(-1)
        
        with torch.no_grad():
            score = score_model(data, mode='score_hhoi')
            
        with torch.enable_grad():
            x_clone = x.clone().detach().requires_grad_(True)
            total_loss = 0
            
            if use_inconsistency_loss and t.item() <= time_thres and step_info["i"] % step_info["k"] == 0:
                inc_loss, human_final_rts_dict, human_final_pose_dict = inconsistency_loss(x_clone, t.item())
                if inc_loss > 0.0:
                    total_loss += inc_loss

                if use_collision_loss:
                    if use_exact_collision:
                        col_loss = collision_loss_exact(human_final_rts_dict, human_final_pose_dict, t.item())
                    else:
                        #col_loss = collision_loss(human_final_rts_dict, human_final_pose_dict, t.item())
                        col_loss = collision_loss_with_capsule(human_final_rts_dict, human_final_pose_dict, t.item())
                        #print(f"time step { t.item()}:", f"     ##col_loss [{col_loss.item()}]##", f"     ##inc_loss [{inc_loss.item()}]##")
                    if col_loss > 0.0:
                        total_loss += col_loss  
                    
            if total_loss > 0.0:
                grad_all = torch.autograd.grad(-total_loss, x_clone, retain_graph=False)[0]
            else:
                grad_all = torch.zeros_like(score).to(score.device)
            
            x_clone.requires_grad_(False)
        
        step_info["i"] += 1
            
        return drift - 0.5 * (diffusion.unsqueeze(-1)**2) * score - grad_all
    

    if solver_type == 'cpu':
        ############ 1. cpu ode solver ############
        # Run the black-box ODE solver, note the 
        t_eval = None
        if num_steps is not None:
            # num_steps, from T -> eps
            t_eval = np.linspace(T, eps, num_steps)

        print("Start sampling!!")
        start = time.time()
        res = integrate.solve_ivp(ode_func, (T, eps), full_init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
        end = time.time()
        print("sampling time(s):", end - start)
        xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, full_shape[1]) # (num_steps, B, hoi_pose_dim*num_hoi + hhi_pose_dim*num_hhi)
        x = torch.tensor(res.y[:, -1], device=device).reshape(full_shape) # (B, hoi_pose_dim*num_hoi + hhi_pose_dim*num_hhi)
    else:
        ############ 2. gpu ode solver ############
        from torchdiffeq import odeint
        t_eval = torch.linspace(T, eps, sampling_steps_for_gpu_solver).to(full_init_x.device)
        print("Start sampling!!")
        start = time.time()
        xs = odeint(ode_func_torch, full_init_x, t_eval, rtol=rtol, atol=atol)
        end = time.time()
        print("sampling time(s):", end - start)
        
        x = xs[-1]
    
    # denoise, using the predictor step in P-C sampler
    #if denoise:
    #    # Reverse diffusion predictor for denoising
    #    vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
    #    drift, diffusion = sde_coeff(vec_eps)
    #    data['sampled_pose'] = x.float()
    #    data['t'] = vec_eps
    #    grad = score_model(data, mode='score_hhoi')
    #    drift = drift - diffusion**2*grad       # R-SDE
    #    mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
    #    x = mean_x
    
    xs = xs.permute(1, 0, 2) # (B, n_steps, F)

    return xs, x

def cond_sdedit_for_R_t_s(
        score_model,
        data,
        prior,
        sde_coeff,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        T=1.0,
        num_steps=None,
        pose_mode='quat_wxyz',
        regression_mode='Rx_Ry_T_S_both',
        denoise=True,
        init_x=None,
    ):
    pose_dim = get_pose_dim(pose_mode)
    if regression_mode == 'Rx_Ry_T_S_both':
        pose_dim = 24
    batch_size=data['gt_pose_scale'].shape[0]
    #init_x = prior((batch_size, pose_dim), T=T).to(device) if init_x is None else init_x + prior((batch_size, pose_dim), T=T).to(device)
    init_x = data['gt_pose_scale'] + prior((batch_size, pose_dim), T=T).to(device)
    shape = init_x.shape
    
    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
        return score.cpu().numpy().reshape((-1,))
    
    def ode_func(t, x):      
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        return drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
  
    # Run the black-box ODE solver, note the 
    t_eval = None
    if num_steps is not None:
        # num_steps, from T -> eps
        t_eval = np.linspace(T, eps, num_steps)

    res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, pose_dim) # [num_steps, bs, pose_dim]
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape) # [bs, pose_dim]
    # denoise, using the predictor step in P-C sampler
    if denoise:
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
        drift, diffusion = sde_coeff(vec_eps)
        data['sampled_pose'] = x.float()
        data['t'] = vec_eps
        grad = score_model(data)
        drift = drift - diffusion**2*grad       # R-SDE
        mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
        x = mean_x
    
    #num_steps = xs.shape[0]
    #xs = xs.reshape(batch_size*num_steps, -1)
    #xs[:, :6] = normalize_rotation(xs[:, :6], pose_mode)
    #xs[:, 12:18] = normalize_rotation(xs[:, 12:18], pose_mode)
    #xs = xs.reshape(num_steps, batch_size, -1)
    #xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    #x[:, :6] = normalize_rotation(x[:, :6], pose_mode)
    #x[:, 12:18] = normalize_rotation(x[:, 12:18], pose_mode)
    #x[:, -3:] += data['pts_center']

    return xs.permute(1, 0, 2), x

def cond_edm_sampler(
    decoder_model, data, prior_fn, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    pose_mode='quat_wxyz', device='cuda'
):
    pose_dim = get_pose_dim(pose_mode)
    batch_size = data['pts'].shape[0]
    latents = prior_fn((batch_size, pose_dim)).to(device)

    # Time step discretization. note that sigma and t is interchangable
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    def decoder_wrapper(decoder, data, x, t):
        # save temp
        x_, t_= data['sampled_pose'], data['t']
        # init data
        data['sampled_pose'], data['t'] = x, t
        # denoise
        data, denoised = decoder(data)
        # recover data
        data['sampled_pose'], data['t'] = x_, t_
        return denoised.to(torch.float64)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    xs = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = torch.as_tensor(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = decoder_wrapper(decoder_model, data, x_hat, t_hat)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = decoder_wrapper(decoder_model, data, x_next, t_next)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        xs.append(x_next.unsqueeze(0))

    xs = torch.stack(xs, dim=0) # [num_steps, bs, pose_dim]
    x = xs[-1] # [bs, pose_dim]

    # post-processing
    xs = xs.reshape(batch_size*num_steps, -1)
    xs[:, :-3] = normalize_rotation(xs[:, :-3], pose_mode)
    xs = xs.reshape(num_steps, batch_size, -1)
    xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
    x[:, -3:] += data['pts_center']

    return xs.permute(1, 0, 2), x


