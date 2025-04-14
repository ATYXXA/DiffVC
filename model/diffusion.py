# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import torch

from model.base import BaseModule
from model.modules import Mish, Upsample, Downsample, Rezero
from model.modules import Residual, SinusoidalPosEmb, RefBlock
from model.modules import Block, ResnetBlock, LinearAttention
# from model.modules import SeparableBlock as Block, SeparableResnetBlock as ResnetBlock, SeparableLinearAttention as  LinearAttention
from .dpm_solver import NoiseScheduleVP

def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init * t + 0.5 * (beta_term - beta_init) * (t**2)
    else:
        noise = beta_init + (beta_term - beta_init) * t
    return noise

class GradLogPEstimator(BaseModule):
    def __init__(self, dim_base, dim_cond, use_ref_t, dim_mults=(1, 2, 4)):
        super(GradLogPEstimator, self).__init__()
        self.use_ref_t = use_ref_t
        dims = [2 + dim_cond, *map(lambda m: dim_base * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim_base)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_base, dim_base * 4), 
                               Mish(), torch.nn.Linear(dim_base * 4, dim_base))
        ###
        cond_total = dim_base + 256
        # cond_total = 256
        ###
        if use_ref_t:
            self.ref_block = RefBlock(out_dim=dim_cond, time_emb_dim=dim_base)
            cond_total += dim_cond
        self.cond_block = torch.nn.Sequential(torch.nn.Linear(cond_total, 4 * dim_cond),
                                      Mish(), torch.nn.Linear(4 * dim_cond, dim_cond))

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim_base),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim_base),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim_base),  ### *2是skip connection
                     ResnetBlock(dim_in, dim_in, time_emb_dim=dim_base),
                     Residual(Rezero(LinearAttention(dim_in))),
                     Upsample(dim_in)]))
        self.final_block = Block(dim_base, dim_base)
        self.final_conv = torch.nn.Conv2d(dim_base, 1, 1)
        ###
        self.unit_proj = torch.nn.Conv1d(768, 80, 1)
        ###
    def forward(self, x, x_mask, mean, ref, ref_mask, c, t):  # mean/unit here  c.shape=torch.Size([16, 256]) c为spk_emb

        condition = self.time_pos_emb(t)
        t = self.mlp(condition)
        ###
        mean = self.unit_proj(mean)
        ###
        x = torch.stack([mean, x], 1) # 增加一个维度
        x_mask = x_mask.unsqueeze(1)
        ref_mask = ref_mask.unsqueeze(1)

        if self.use_ref_t:
            condition = torch.cat([condition, self.ref_block(ref, ref_mask, t)], 1)
        ###
        condition = torch.cat([condition, c], 1) # 维度不变
        # print(condition.shape) # torch.Size([16, 512])
        # condition = c
        ###

        condition = self.cond_block(condition).unsqueeze(-1).unsqueeze(-1)
        condition = torch.cat(x.shape[2]*[condition], 2) 
        condition = torch.cat(x.shape[3]*[condition], 3) #value -> img
        # print(condition.shape) # torch.Size([16, 128, 80, 128])
        x = torch.cat([x, condition], 1) # channel维拼接
        # print(x.shape) # torch.Size([16, 130, 80, 128])
        hiddens = []
        masks = [x_mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, x_mask)
        output = self.final_conv(x * x_mask)

        return (output * x_mask).squeeze(1)


class Diffusion(BaseModule):
    def __init__(self, n_feats, dim_unet, dim_spk, use_ref_t, beta_min, beta_max):
        super(Diffusion, self).__init__()
        self.estimator = GradLogPEstimator(dim_unet, dim_spk, use_ref_t)
        self.n_feats = n_feats
        self.dim_unet = dim_unet
        self.dim_spk = dim_spk
        self.use_ref_t = use_ref_t
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.dpm_solver_sch = NoiseScheduleVP()

    def get_beta(self, t):
        beta = self.beta_min + (self.beta_max - self.beta_min) * t
        return beta

    def get_gamma(self, s, t, p=1.0, use_torch=False):
        beta_integral = self.beta_min + 0.5*(self.beta_max - self.beta_min)*(t + s)
        beta_integral *= (t - s)
        if use_torch:
            gamma = torch.exp(-0.5*p*beta_integral).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = math.exp(-0.5*p*beta_integral)
        return gamma

    def get_mu(self, s, t):
        a = self.get_gamma(s, t)
        b = 1.0 - self.get_gamma(0, s, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_nu(self, s, t):
        a = self.get_gamma(0, s)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_sigma(self, s, t):
        a = 1.0 - self.get_gamma(0, s, p=2.0)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return math.sqrt(a * b / c)

    def compute_diffused_mean(self, x0, mask, t, use_torch=False): # mean won't be used
        x0_weight = self.get_gamma(0, t, use_torch=use_torch)
        mean_weight = 1.0 - x0_weight
        ###
        #xt_mean = x0 * x0_weight + mean * mean_weight
        mean_zero = torch.zeros_like(x0)
        xt_mean = x0 * x0_weight + mean_zero * mean_weight
        ###
        # print(x0.requires_grad, mean.requires_grad, xt_mean.requires_grad) # False*3
        return xt_mean * mask
    
    def forward_diffusion(self, x0, mask, t):
        xt_mean = self.compute_diffused_mean(x0, mask, t, use_torch=True)
        variance = 1.0 - self.get_gamma(0, t, p=2.0, use_torch=True)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = xt_mean + z * torch.sqrt(variance)
        # print(x0.requires_grad,mean.requires_grad,xt.requires_grad,z.requires_grad) # False*3
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mean, ref, ref_mask, c, 
                          n_timesteps, mode):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = 1.0 - i*h
            time = t * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            beta_t = self.get_beta(t)
            xt_ref = [self.compute_diffused_mean(ref, ref_mask, t)]
            '''
            for j in range(15):
                xt_ref += [self.compute_diffused_mean(ref, ref_mask, mean_ref, (j+0.5)/15.0)]
            '''
            xt_ref = torch.stack(xt_ref, 1)
            if mode == 'pf':
                dxt = 0.5 * (mean - xt - self.estimator(xt, mask, mean, xt_ref, ref_mask, c, time)) * (beta_t * h)
            else:
                if mode == 'ml':
                    kappa = self.get_gamma(0, t - h) * (1.0 - self.get_gamma(t - h, t, p=2.0))
                    kappa /= (self.get_gamma(0, t) * beta_t * h)
                    kappa -= 1.0
                    omega = self.get_nu(t - h, t) / self.get_gamma(0, t)
                    omega += self.get_mu(t - h, t)
                    omega -= (0.5 * beta_t * h + 1.0)
                    sigma = self.get_sigma(t - h, t)
                else:
                    kappa = 0.0
                    omega = 0.0
                    sigma = math.sqrt(beta_t * h)
                ###
                # dxt = (mean - xt) * (0.5 * beta_t * h + omega)
                mean_zero = torch.zeros_like(xt) 
                dxt = (mean_zero - xt) * (0.5 * beta_t * h + omega)
                ###
                dxt -= self.estimator(xt, mask, mean, xt_ref, ref_mask, c, time) * (1.0 + kappa) * (beta_t * h)
                dxt += torch.randn_like(z, device=z.device) * sigma
            xt = (xt - dxt) * mask
        return xt

    def reverse_diffusion_dpm_solver(self, z, mask, mean, ref, ref_mask, c, 
                          n_timesteps):

        xt = z * mask
        mean_zero = torch.zeros_like(xt) 
        yt = xt - mean_zero
        T = 1
        eps = 1e-3
        time = self.dpm_solver_sch.get_time_steps(T, eps, n_timesteps)
        for i in range(n_timesteps):
            s = torch.ones((xt.shape[0], )).to(xt.device) * time[i]
            t = torch.ones((xt.shape[0], )).to(xt.device) * time[i + 1]
            ns = self.dpm_solver_sch
            lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
            h = lambda_t - lambda_s
            log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(
                s), ns.marginal_log_mean_coeff(t)
            sigma_t = ns.marginal_std(t)
            phi_1 = torch.expm1(h)
            ###
            # noise_s = self.estimator(yt + mean, mask, mean, s, spk)
            # xt_ref = [self.compute_diffused_mean(ref, ref_mask, t)]
            # xt_ref = torch.stack(xt_ref, 1).
            xt_ref = None
            noise_s = self.estimator(xt, mask, mean, xt_ref, ref_mask, c, s)
            ###
            lt = 1 - torch.exp(
                -get_noise(s, self.beta_min, self.beta_max, cumulative=True))
            a = torch.exp(log_alpha_t - log_alpha_s)
            b = sigma_t * phi_1 * torch.sqrt(lt)
            yt = a * yt + (b * noise_s)
        xt = yt + mean_zero
        return xt


    @torch.no_grad()
    def forward(self, z, mask, mean, ref, ref_mask, c, 
                n_timesteps, mode, dpm=False):
        if dpm:
            return self.reverse_diffusion_dpm_solver(z, mask, mean, ref, ref_mask, c, 
                                      n_timesteps)
        if mode not in ['pf', 'em', 'ml']:
            print('Inference mode must be one of [pf, em, ml]!')
            return z
        return self.reverse_diffusion(z, mask, mean, ref, ref_mask, c, 
                                      n_timesteps, mode)

    def loss_t(self, x0, mask, mean, x_ref, c, t):
        xt, z = self.forward_diffusion(x0, mask, t) ###forward_diffusion也调用了compute_diffused_mean
        xt_ref = [self.compute_diffused_mean(x_ref, mask, t, use_torch=True)]
        '''
        for j in range(15):
            xt_ref += [self.compute_diffused_mean(x_ref, mask, mean_ref, (j+0.5)/15.0)]
        '''
        xt_ref = torch.stack(xt_ref, 1)
        z_estimation = self.estimator(xt, mask, mean, xt_ref, mask, c, t)
        z_estimation *= torch.sqrt(1.0 - self.get_gamma(0, t, p=2.0, use_torch=True))
        loss = torch.sum((z_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        return loss

    def compute_loss(self, x0, mask, mean, x_ref, c, offset=1e-5): # mean/unit here
        b = x0.shape[0]
        t = torch.rand(b, dtype=x0.dtype, device=x0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mean, x_ref, c, t)


class GradLogPEstimator_4recon(BaseModule):
    def __init__(self, dim_base, dim_cond, dim_mults=(1, 2, 4)):
        super(GradLogPEstimator_4recon, self).__init__()
        dims = [2 + dim_cond, *map(lambda m: dim_base * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim_base)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_base, dim_base * 4), 
                               Mish(), torch.nn.Linear(dim_base * 4, dim_base))

        # cond_total = dim_base + 256
        cond_total = dim_base
        self.cond_block = torch.nn.Sequential(torch.nn.Linear(cond_total, 4 * dim_cond),
                                      Mish(), torch.nn.Linear(4 * dim_cond, dim_cond))

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim_base),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim_base),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim_base),  ### *2是skip connection
                     ResnetBlock(dim_in, dim_in, time_emb_dim=dim_base),
                     Residual(Rezero(LinearAttention(dim_in))),
                     Upsample(dim_in)]))
        self.final_block = Block(dim_base, dim_base)
        self.final_conv = torch.nn.Conv2d(dim_base, 1, 1)
        ###
        # self.unit_proj = torch.nn.Conv1d(768, 80, 1)
        self.unit_proj = torch.nn.Conv1d(768, 80, 1)
        ###
    def forward(self, x, x_mask, mean, t):  # mean/unit here

        condition = self.time_pos_emb(t)
        t = self.mlp(condition)
        ###
        mean = self.unit_proj(mean)
        ###
        x = torch.stack([mean, x], 1) 
        x_mask = x_mask.unsqueeze(1)
      
        condition = self.cond_block(condition).unsqueeze(-1).unsqueeze(-1)
        condition = torch.cat(x.shape[2]*[condition], 2) 
        condition = torch.cat(x.shape[3]*[condition], 3) 

        x = torch.cat([x, condition], 1) 
        hiddens = []
        masks = [x_mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, x_mask)
        output = self.final_conv(x * x_mask)
 
        return (output * x_mask).squeeze(1)


class Diffusion_4recon(BaseModule):
    def __init__(self, n_feats, dim_unet, dim_spk, beta_min, beta_max):
        super(Diffusion_4recon, self).__init__()
        self.estimator = GradLogPEstimator_4recon(dim_unet, dim_spk)
        self.n_feats = n_feats
        self.dim_unet = dim_unet
        self.dim_spk = dim_spk
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_beta(self, t):
        beta = self.beta_min + (self.beta_max - self.beta_min) * t
        return beta

    def get_gamma(self, s, t, p=1.0, use_torch=False):
        beta_integral = self.beta_min + 0.5*(self.beta_max - self.beta_min)*(t + s)
        beta_integral *= (t - s)
        if use_torch:
            gamma = torch.exp(-0.5*p*beta_integral).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = math.exp(-0.5*p*beta_integral)
        return gamma

    def get_mu(self, s, t):
        a = self.get_gamma(s, t)
        b = 1.0 - self.get_gamma(0, s, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_nu(self, s, t):
        a = self.get_gamma(0, s)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_sigma(self, s, t):
        a = 1.0 - self.get_gamma(0, s, p=2.0)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return math.sqrt(a * b / c)

    def compute_diffused_mean(self, x0, mask, t, use_torch=False): 
        x0_weight = self.get_gamma(0, t, use_torch=use_torch)
        mean_weight = 1.0 - x0_weight
        mean_zero = torch.zeros_like(x0)
        xt_mean = x0 * x0_weight + mean_zero * mean_weight
        return xt_mean * mask
    
    def forward_diffusion(self, x0, mask, t):
        xt_mean = self.compute_diffused_mean(x0, mask, t, use_torch=True)
        variance = 1.0 - self.get_gamma(0, t, p=2.0, use_torch=True)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = xt_mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mean,
                          n_timesteps, mode):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = 1.0 - i*h
            time = t * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            beta_t = self.get_beta(t)

            if mode == 'pf':
                dxt = 0.5 * (mean - xt - self.estimator(xt, mask, mean, time)) * (beta_t * h)
            else:
                if mode == 'ml':
                    kappa = self.get_gamma(0, t - h) * (1.0 - self.get_gamma(t - h, t, p=2.0))
                    kappa /= (self.get_gamma(0, t) * beta_t * h)
                    kappa -= 1.0
                    omega = self.get_nu(t - h, t) / self.get_gamma(0, t)
                    omega += self.get_mu(t - h, t)
                    omega -= (0.5 * beta_t * h + 1.0)
                    sigma = self.get_sigma(t - h, t)
                else:
                    kappa = 0.0
                    omega = 0.0
                    sigma = math.sqrt(beta_t * h)
                ###
                # dxt = (mean - xt) * (0.5 * beta_t * h + omega)
                mean_zero = torch.zeros_like(xt) 
                dxt = (mean_zero - xt) * (0.5 * beta_t * h + omega)
                ###
                dxt -= self.estimator(xt, mask, mean, time) * (1.0 + kappa) * (beta_t * h)
                dxt += torch.randn_like(z, device=z.device) * sigma
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mean, n_timesteps, mode):
        if mode not in ['pf', 'em', 'ml']:
            print('Inference mode must be one of [pf, em, ml]!')
            return z
        return self.reverse_diffusion(z, mask, mean,
                                      n_timesteps, mode)

    def loss_t(self, x0, mask, mean, t):
        xt, z = self.forward_diffusion(x0, mask, t) 
        z_estimation = self.estimator(xt, mask, mean, t)
        z_estimation *= torch.sqrt(1.0 - self.get_gamma(0, t, p=2.0, use_torch=True))
        loss = torch.sum((z_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        return loss

    def compute_loss(self, x0, mask, mean, offset=1e-5): # mean/unit here
        b = x0.shape[0]
        t = torch.rand(b, dtype=x0.dtype, device=x0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mean, t)
    


from .utils4ctrl import *     
class Controller(BaseModule):
    def __init__(self, dim_base, dim_cond, dim_mults=(1, 2, 4)):
        super(Controller, self).__init__()
        dims = [2 + dim_cond, *map(lambda m: dim_base * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim_base)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_base, dim_base * 4), 
                               Mish(), torch.nn.Linear(dim_base * 4, dim_base))

        self.unit_proj = torch.nn.Conv1d(768, 80, 1)

        # cond_total = dim_base + 256
        cond_total = dim_base # dim_time
        self.cond_block = torch.nn.Sequential(torch.nn.Linear(cond_total, 4 * dim_cond),
                                      Mish(), torch.nn.Linear(4 * dim_cond, dim_cond))

        self.downs = torch.nn.ModuleList([])
        num_resolutions = len(in_out)
        ###
        hint_dim = 256
        self.hint_block = torch.nn.Sequential(torch.nn.Linear(hint_dim, 4 * dim_cond),
                                      Mish(), torch.nn.Linear(4 * dim_cond, dim_cond)
                                      ) # 1 mish or 2?
        self.hint_zero_conv = self.make_zero_conv(dim_cond, dim=1)
        self.zero_convs = torch.nn.ModuleList([])
        ###
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim_base),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim_base),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))
            self.zero_convs.append(self.make_zero_conv(dim_out))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        self.mid_zero_conv = self.make_zero_conv(mid_dim)


    def make_zero_conv(self, channels, dim=2):
        assert dim in [1,2]
        if dim==1:
            return zero_module(torch.nn.Conv1d(channels, channels, kernel_size=1, padding=0))
        else:
            return zero_module(torch.nn.Conv2d(channels, channels, kernel_size=1, padding=0))


    def forward(self, x, x_mask, mean, t, hint):  # mean/unit here

        condition = self.time_pos_emb(t)
        t = self.mlp(condition)
        mean = self.unit_proj(mean)

        x = torch.stack([mean, x], 1) 
        x_mask = x_mask.unsqueeze(1)
      
        condition = self.cond_block(condition).unsqueeze(-1).unsqueeze(-1)
        ###
        hint = self.hint_zero_conv(self.hint_block(hint).unsqueeze(-1)).unsqueeze(-1)
        # print(hint.shape) (bs, dim_cond)
        condition += hint
        ###
        condition = torch.cat(x.shape[2]*[condition], 2) 
        condition = torch.cat(x.shape[3]*[condition], 3) 

        x = torch.cat([x, condition], 1) 
        ###
        # hiddens = []
        outputs = []
        ###
        masks = [x_mask]

        for (resnet1, resnet2, attn, downsample), zero_conv in zip(self.downs, self.zero_convs):
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            ###
            # hiddens.append(x)
            outputs.append(zero_conv(x))
            ###
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)
        ###
        outputs.append(self.mid_zero_conv(x))
        ###
 
        return outputs

class ControlledEstimator(GradLogPEstimator_4recon):

    def forward(self, x, x_mask, mean, t, controls=None, only_mid_control=False):  

        condition = self.time_pos_emb(t)
        t = self.mlp(condition)
        mean = self.unit_proj(mean) # mean/unit here

        x = torch.stack([mean, x], 1) 
        x_mask = x_mask.unsqueeze(1)
      
        condition = self.cond_block(condition).unsqueeze(-1).unsqueeze(-1)
        condition = torch.cat(x.shape[2]*[condition], 2) 
        condition = torch.cat(x.shape[3]*[condition], 3) 

        x = torch.cat([x, condition], 1) 
        hiddens = []
        masks = [x_mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        if controls is not None:
            x += controls.pop()
        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            if only_mid_control or controls is None:     
                x = torch.cat((x, hiddens.pop()), dim=1)
            else:
                x = torch.cat((x, hiddens.pop() + controls.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, x_mask)
        output = self.final_conv(x * x_mask)
 
        return (output * x_mask).squeeze(1)



class ControlledDiffusion(Diffusion_4recon):
    def __init__(self, n_feats, dim_unet, dim_spk, beta_min, beta_max):
        super().__init__(n_feats, dim_unet, dim_spk, beta_min, beta_max)
        self.estimator = ControlledEstimator(dim_unet, dim_spk)
        self.controller = Controller(dim_unet, dim_spk)
   
    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mean,
                          n_timesteps, mode, hint):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = 1.0 - i*h
            time = t * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            beta_t = self.get_beta(t)

            if mode == 'pf':
                dxt = 0.5 * (mean - xt - self.estimator(xt, mask, mean, time)) * (beta_t * h)
            else:
                if mode == 'ml':
                    kappa = self.get_gamma(0, t - h) * (1.0 - self.get_gamma(t - h, t, p=2.0))
                    kappa /= (self.get_gamma(0, t) * beta_t * h)
                    kappa -= 1.0
                    omega = self.get_nu(t - h, t) / self.get_gamma(0, t)
                    omega += self.get_mu(t - h, t)
                    omega -= (0.5 * beta_t * h + 1.0)
                    sigma = self.get_sigma(t - h, t)
                else:
                    kappa = 0.0
                    omega = 0.0
                    sigma = math.sqrt(beta_t * h)
                ###
                # dxt = (mean - xt) * (0.5 * beta_t * h + omega)
                mean_zero = torch.zeros_like(xt) 
                dxt = (mean_zero - xt) * (0.5 * beta_t * h + omega)
                ###
                controls = self.controller(xt, mask, mean, time, hint)
                dxt -= self.estimator(xt, mask, mean, time, controls = controls) * (1.0 + kappa) * (beta_t * h)
                dxt += torch.randn_like(z, device=z.device) * sigma
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mean, n_timesteps, mode, hint):
        if mode not in ['pf', 'em', 'ml']:
            print('Inference mode must be one of [pf, em, ml]!')
            return z
        return self.reverse_diffusion(z, mask, mean,
                                      n_timesteps, mode, hint)
    
    def forward_uncond(self, z, mask, mean, n_timesteps):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = 1.0 - i*h
            time = t * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            beta_t = self.get_beta(t)

            kappa = self.get_gamma(0, t - h) * (1.0 - self.get_gamma(t - h, t, p=2.0))
            kappa /= (self.get_gamma(0, t) * beta_t * h)
            kappa -= 1.0
            omega = self.get_nu(t - h, t) / self.get_gamma(0, t)
            omega += self.get_mu(t - h, t)
            omega -= (0.5 * beta_t * h + 1.0)
            sigma = self.get_sigma(t - h, t)

            mean_zero = torch.zeros_like(xt) 
            dxt = (mean_zero - xt) * (0.5 * beta_t * h + omega)
            
            dxt -= self.estimator(xt, mask, mean, time) * (1.0 + kappa) * (beta_t * h)
            dxt += torch.randn_like(z, device=z.device) * sigma
            xt = (xt - dxt) * mask
        return xt


    def loss_t(self, x0, mask, mean, t, hint):
        xt, z = self.forward_diffusion(x0, mask, t) 
        controls = self.controller(xt, mask, mean, t, hint)
        z_estimation = self.estimator(xt, mask, mean, t, controls = controls)
        z_estimation *= torch.sqrt(1.0 - self.get_gamma(0, t, p=2.0, use_torch=True))
        loss = torch.sum((z_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        return loss

    def compute_loss(self, x0, mask, mean, offset=1e-5, hint = None): # mean/unit here
        b = x0.shape[0]
        t = torch.rand(b, dtype=x0.dtype, device=x0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        assert hint is not None
        return self.loss_t(x0, mask, mean, t, hint)
    

import numpy as np
def mask_unit0(x0, l_ratio, h_ratio):
    '''x.shape = (bs,f,t)'''
    x = x0.clone()
    bs = x.shape[0]
    t = x.shape[-1]
    for i in range(bs):
        n_mask = np.random.randint(low=int(t*l_ratio), high=int(t*h_ratio)+1)
        mask = list(np.random.choice(range(t),n_mask,replace=False))
        x[i, :, mask] = 0
    return x

def mask_unit(x0, step):
    '''x.shape = (bs,f,t)'''
    x = x0.clone()
    mask=torch.zeros_like(x).cuda()
    mask[:,:,::step]=1
    return x*mask


class Controller_emo(BaseModule):
    def __init__(self, dim_base, dim_cond, dim_mults=(1, 2, 4)):
        super(Controller_emo, self).__init__()
        dims = [2 + dim_cond, *map(lambda m: dim_base * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim_base)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_base, dim_base * 4), 
                               Mish(), torch.nn.Linear(dim_base * 4, dim_base))

        self.unit_proj = torch.nn.Conv1d(768, 80, 1)

        # cond_total = dim_base + 256
        cond_total = dim_base # dim_time
        self.cond_block = torch.nn.Sequential(torch.nn.Linear(cond_total, 4 * dim_cond),
                                      Mish(), torch.nn.Linear(4 * dim_cond, dim_cond))

        self.downs = torch.nn.ModuleList([])
        num_resolutions = len(in_out)
        ###
        # hint_dim = 256 + 5
        hint_dim = 5 #emo only
        hint_dim = 768 
        self.hint_block = torch.nn.Sequential(torch.nn.Linear(hint_dim, 256),
                                      Mish(), torch.nn.Linear(256, 4 * dim_cond),
                                      Mish(), torch.nn.Linear(4 * dim_cond, dim_cond),
                                      Mish()
                                      ) # 1 mish or 2?
        self.hint_zero_conv = self.make_zero_conv(dim_cond, dim=1)
        self.zero_convs = torch.nn.ModuleList([])
        ###
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim_base),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim_base),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))
            self.zero_convs.append(self.make_zero_conv(dim_out))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        self.mid_zero_conv = self.make_zero_conv(mid_dim)


    def make_zero_conv(self, channels, dim=2):
        assert dim in [1,2]
        if dim==1:
            return zero_module(torch.nn.Conv1d(channels, channels, kernel_size=1, padding=0))
        else:
            return zero_module(torch.nn.Conv2d(channels, channels, kernel_size=1, padding=0))


    def forward(self, x, x_mask, mean, t, hint, emo):  # mean/unit here

        condition = self.time_pos_emb(t)
        t = self.mlp(condition)
        ###
        # mean = mask_unit(mean, 0.5, 0.75)
        ###
        mean = self.unit_proj(mean)

        x = torch.stack([mean, x], 1) 
        x_mask = x_mask.unsqueeze(1)
      
        condition = self.cond_block(condition).unsqueeze(-1).unsqueeze(-1)
        ###
        # hint = torch.cat([hint, emo], 1)
        # hint = self.hint_zero_conv(self.hint_block(hint).unsqueeze(-1)).unsqueeze(-1)
        hint = self.hint_zero_conv(self.hint_block(emo).unsqueeze(-1)).unsqueeze(-1)
        # print(hint.shape) (bs, dim_cond)
        condition += hint
        ###
        condition = torch.cat(x.shape[2]*[condition], 2) 
        condition = torch.cat(x.shape[3]*[condition], 3) 

        x = torch.cat([x, condition], 1) 
        ###
        # hiddens = []
        outputs = []
        ###
        masks = [x_mask]

        for (resnet1, resnet2, attn, downsample), zero_conv in zip(self.downs, self.zero_convs):
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            ###
            # hiddens.append(x)
            outputs.append(zero_conv(x))
            ###
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)
        ###
        outputs.append(self.mid_zero_conv(x))
        ###
 
        return outputs


class ControlledDiffusion_emo(ControlledDiffusion):
    def __init__(self, n_feats, dim_unet, dim_spk, beta_min, beta_max):
        super().__init__(n_feats, dim_unet, dim_spk, beta_min, beta_max)
        self.estimator = ControlledEstimator(dim_unet, dim_spk)
        self.controller = Controller_emo(dim_unet, dim_spk)    

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mean,
                          n_timesteps, mode, hint, emo):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = 1.0 - i*h
            time = t * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            beta_t = self.get_beta(t)

            if mode == 'pf':
                dxt = 0.5 * (mean - xt - self.estimator(xt, mask, mean, time)) * (beta_t * h)
            else:
                if mode == 'ml':
                    kappa = self.get_gamma(0, t - h) * (1.0 - self.get_gamma(t - h, t, p=2.0))
                    kappa /= (self.get_gamma(0, t) * beta_t * h)
                    kappa -= 1.0
                    omega = self.get_nu(t - h, t) / self.get_gamma(0, t)
                    omega += self.get_mu(t - h, t)
                    omega -= (0.5 * beta_t * h + 1.0)
                    sigma = self.get_sigma(t - h, t)
                else:
                    kappa = 0.0
                    omega = 0.0
                    sigma = math.sqrt(beta_t * h)
                ###
                # dxt = (mean - xt) * (0.5 * beta_t * h + omega)
                mean_zero = torch.zeros_like(xt) 
                dxt = (mean_zero - xt) * (0.5 * beta_t * h + omega)
                ###
                controls = self.controller(xt, mask, mean, time, hint, emo)
                dxt -= self.estimator(xt, mask, mean, time, controls = controls) * (1.0 + kappa) * (beta_t * h)
                dxt += torch.randn_like(z, device=z.device) * sigma
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mean, n_timesteps, mode, hint, emo):
        if mode not in ['pf', 'em', 'ml']:
            print('Inference mode must be one of [pf, em, ml]!')
            return z
        ###
        # mean_bkp=mean.clone()
        # print(mean)
        mean = mask_unit(mean, 10)
        # print(mean)
        # print((mean==mean_bkp).all())
        ###
        return self.reverse_diffusion(z, mask, mean,
                                      n_timesteps, mode, hint, emo)

    def loss_t(self, x0, mask, mean, t, hint, emo):
        xt, z = self.forward_diffusion(x0, mask, t) 
        controls = self.controller(xt, mask, mean, t, hint, emo)
        z_estimation = self.estimator(xt, mask, mean, t, controls = controls)
        z_estimation *= torch.sqrt(1.0 - self.get_gamma(0, t, p=2.0, use_torch=True))
        loss = torch.sum((z_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        return loss

    def compute_loss(self, x0, mask, mean, offset=1e-5, hint = None, emo=None): # mean/unit here
        b = x0.shape[0]
        t = torch.rand(b, dtype=x0.dtype, device=x0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        assert hint is not None and emo is not None
        ###
        mean = mask_unit(mean, 10)
        ###
        return self.loss_t(x0, mask, mean, t, hint, emo)