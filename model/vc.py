# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import torch

from model.base import BaseModule
from model.encoder import MelEncoder, UnitEncoder, UFEncoder
from model.postnet import PostNet
from model.diffusion import Diffusion, Diffusion_4recon, ControlledDiffusion, ControlledDiffusion_emo
from model.utils import sequence_mask, fix_len_compatibility, mse_loss, mae_loss
from model.diffusion_light import Diffusion as Diffusion_light

# "average voice" encoder as the module parameterizing the diffusion prior
class FwdDiffusion(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, dim):
        super(FwdDiffusion, self).__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.dim = dim
        self.encoder = MelEncoder(n_feats, channels, filters, heads, layers, 
                                  kernel, dropout, window_size)
        self.postnet = PostNet(dim)

    @torch.no_grad()
    def forward(self, x, mask):
        x, mask = self.relocate_input([x, mask])
        z = self.encoder(x, mask)
        z_output = self.postnet(z, mask)
        return z_output

    def compute_loss(self, x, y, mask):
        x, y, mask = self.relocate_input([x, y, mask])
        z = self.encoder(x, mask)
        z_output = self.postnet(z, mask)
        loss = mse_loss(z_output, y, mask, self.n_feats)
        return loss
    
class U2E(BaseModule):
    def __init__(self, n_vocab, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size):
        super().__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.encoder = UnitEncoder(n_vocab, n_feats, channels, filters, heads, layers, 
                                  kernel, dropout, window_size)

    @torch.no_grad()
    def forward(self, x, mask):
        x, mask = self.relocate_input([x, mask])
        z = self.encoder(x, mask)
        return z

    def compute_loss(self, x, y, mask):
        x, y, mask = self.relocate_input([x, y, mask])
        z = self.encoder(x, mask)
        # loss = mse_loss(z, y, mask, self.n_feats)
        loss = mae_loss(z, y, mask, self.n_feats)
        return loss
    
###
class UF2E(BaseModule):
    def __init__(self, n_vocab, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size):
        super().__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.encoder = UFEncoder(n_vocab, n_feats, channels, filters, heads, layers, 
                                  kernel, dropout, window_size)

    @torch.no_grad()
    def forward(self, x, mask, f0):
        x, mask = self.relocate_input([x, mask])
        z = self.encoder(x, mask, f0)
        return z

    def compute_loss(self, x, y, mask, f0):
        x, y, mask = self.relocate_input([x, y, mask])
        z = self.encoder(x, mask, f0)
        loss = mse_loss(z, y, mask, self.n_feats)
        # loss = mae_loss(z, y, mask, self.n_feats)
        return loss
###

from model.encoder import TEEncoder
class TE2E(BaseModule):
    def __init__(self, n_enc, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size):
        super().__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.encoder = TEEncoder(n_enc, n_feats, channels, filters, heads, layers, 
                                  kernel, dropout, window_size)

    @torch.no_grad()
    def forward(self, x, mask):
        x, mask = self.relocate_input([x, mask])
        z = self.encoder(x, mask)
        return z

    def compute_loss(self, x, y, mask):
        x, y, mask = self.relocate_input([x, y, mask])
        z = self.encoder(x, mask)
        # loss = mse_loss(z, y, mask, self.n_feats)
        loss = mae_loss(z, y, mask, self.n_feats)
        return loss

# the whole voice conversion model consisting of the "average voice" encoder 
# and the diffusion-based speaker-conditional decoder
class DiffVC(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, enc_dim, spk_dim, use_ref_t, dec_dim, 
                 beta_min, beta_max):
        super(DiffVC, self).__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.enc_dim = enc_dim
        self.spk_dim = spk_dim
        self.use_ref_t = use_ref_t
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        # self.encoder = FwdDiffusion(n_feats, channels, filters, heads, layers,
        #                             kernel, dropout, window_size, enc_dim)
        self.decoder = Diffusion(n_feats, dec_dim, spk_dim, use_ref_t, 
                                 beta_min, beta_max)

    # def load_encoder(self, enc_path):
    #     enc_dict = torch.load(enc_path, map_location=lambda loc, storage: loc)
    #     self.encoder.load_state_dict(enc_dict, strict=False)

    @torch.no_grad()
    def forward(self, x, x_lengths, x_ref, x_ref_lengths, c, unit, n_timesteps, 
                mode='ml', dpm=False):
        # forward(mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target, n_timesteps=30, mode='ml')
        """
        Generates mel-spectrogram from source mel-spectrogram conditioned on
        target speaker embedding. Returns:
            1. 'average voice' encoder outputs
            2. decoder outputs
        
        Args:
            x (torch.Tensor): batch of source mel-spectrograms.
            x_lengths (torch.Tensor): numbers of frames in source mel-spectrograms.
            x_ref (torch.Tensor): batch of reference mel-spectrograms.
            x_ref_lengths (torch.Tensor): numbers of frames in reference mel-spectrograms.
            c (torch.Tensor): batch of reference speaker embeddings
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            mode (string, optional): sampling method. Can be one of:
              'pf' - probability flow sampling (Euler scheme for ODE)
              'em' - Euler-Maruyama SDE solver
              'ml' - Maximum Likelihood SDE solver
        """
        x, x_lengths = self.relocate_input([x, x_lengths])
        x_ref, x_ref_lengths, c = self.relocate_input([x_ref, x_ref_lengths, c])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        x_ref_mask = sequence_mask(x_ref_lengths).unsqueeze(1).to(x_ref.dtype)
        # mean = self.encoder(x, x_mask) ###内容表示
        mean = unit
        mean_x = self.decoder.compute_diffused_mean(x, x_mask, 1.0)  ###x和mean加权 t=1.0
        # mean_ref = self.encoder(x_ref, x_ref_mask)

        b = x.shape[0]
        max_length = int(x_lengths.max())
        max_length_new = fix_len_compatibility(max_length)
        x_mask_new = sequence_mask(x_lengths, max_length_new).unsqueeze(1).to(x.dtype)
        mean_new = torch.zeros((b, 768, max_length_new), dtype=x.dtype, 
                                device=x.device)
        mean_x_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                  device=x.device)
        for i in range(b):
            mean_new[i, :, :x_lengths[i]] = mean[i, :, :x_lengths[i]]
            mean_x_new[i, :, :x_lengths[i]] = mean_x[i, :, :x_lengths[i]]

        z = mean_x_new
        z += torch.randn_like(mean_x_new, device=mean_x_new.device)

        y = self.decoder(z, x_mask_new, mean_new, x_ref, x_ref_mask, c, 
                         n_timesteps, mode, dpm) ###去噪结果？ ✔
        return mean_x, y[:, :, :max_length]

    def compute_loss(self, x, x_lengths, x_ref, c, unit):
        """
        Computes diffusion (score matching) loss.
            
        Args:
            x (torch.Tensor): batch of source mel-spectrograms.
            x_lengths (torch.Tensor): numbers of frames in source mel-spectrograms.
            x_ref (torch.Tensor): batch of reference mel-spectrograms.
            c (torch.Tensor): batch of reference speaker embeddings
        """
        x, x_lengths, x_ref, c = self.relocate_input([x, x_lengths, x_ref, c])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        # mean = self.encoder(x, x_mask).detach()
        # mean_ref = self.encoder(x_ref, x_mask).detach()
        diff_loss = self.decoder.compute_loss(x, x_mask, unit, x_ref, c)
        return diff_loss


class DiffVC_4recon(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, enc_dim, spk_dim, use_ref_t, dec_dim, 
                 beta_min, beta_max):
        super(DiffVC_4recon, self).__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.enc_dim = enc_dim
        self.spk_dim = spk_dim
        self.use_ref_t = use_ref_t
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.decoder = Diffusion_4recon(n_feats, dec_dim, spk_dim,
                                 beta_min, beta_max)


    @torch.no_grad()
    def forward(self, x, x_lengths, unit, n_timesteps, 
                mode='ml'):

        x, x_lengths = self.relocate_input([x, x_lengths])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        mean = unit
        mean_x = self.decoder.compute_diffused_mean(x, x_mask, 1.0)  ###x和mean加权 t=1.0

        b = x.shape[0]
        max_length = int(x_lengths.max())
        max_length_new = fix_len_compatibility(max_length)
        x_mask_new = sequence_mask(x_lengths, max_length_new).unsqueeze(1).to(x.dtype)
        mean_new = torch.zeros((b, 768, max_length_new), dtype=x.dtype, 
                                device=x.device)
        mean_x_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                  device=x.device)
        for i in range(b):
            mean_new[i, :, :x_lengths[i]] = mean[i, :, :x_lengths[i]]
            mean_x_new[i, :, :x_lengths[i]] = mean_x[i, :, :x_lengths[i]]

        z = mean_x_new
        z += torch.randn_like(mean_x_new, device=mean_x_new.device)

        y = self.decoder(z, x_mask_new, mean_new, n_timesteps, mode) 

        return mean_x, y[:, :, :max_length]

    def compute_loss(self, x, x_lengths, unit):
        x, x_lengths = self.relocate_input([x, x_lengths])
        x_mask = sequence_mask(x_lengths, max_length=128).unsqueeze(1).to(x.dtype)
        # print(x.shape, x_lengths.shape, x_mask.shape)
        diff_loss = self.decoder.compute_loss(x, x_mask, unit)
        return diff_loss

class ControlledDiffVC(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, enc_dim, spk_dim, use_ref_t, dec_dim, 
                 beta_min, beta_max):
        super(ControlledDiffVC, self).__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.enc_dim = enc_dim
        self.spk_dim = spk_dim
        self.use_ref_t = use_ref_t
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.decoder = ControlledDiffusion(n_feats, dec_dim, spk_dim,
                                 beta_min, beta_max)


    @torch.no_grad()
    def forward(self, x, x_lengths, unit, n_timesteps, hint,
                mode='ml', uncond=False):

        x, x_lengths = self.relocate_input([x, x_lengths])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        mean = unit
        mean_x = self.decoder.compute_diffused_mean(x, x_mask, 1.0)  ###x和mean加权 t=1.0

        b = x.shape[0]
        max_length = int(x_lengths.max())
        max_length_new = fix_len_compatibility(max_length)
        x_mask_new = sequence_mask(x_lengths, max_length_new).unsqueeze(1).to(x.dtype)
        mean_new = torch.zeros((b, 768, max_length_new), dtype=x.dtype, 
                                device=x.device)
        mean_x_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                  device=x.device)
        for i in range(b):
            mean_new[i, :, :x_lengths[i]] = mean[i, :, :x_lengths[i]]
            mean_x_new[i, :, :x_lengths[i]] = mean_x[i, :, :x_lengths[i]]

        z = mean_x_new
        z += torch.randn_like(mean_x_new, device=mean_x_new.device)

        if not uncond:
            y = self.decoder(z, x_mask_new, mean_new, n_timesteps, mode, hint) 
        else:
            y = self.decoder.forward_uncond(z, x_mask_new, mean_new, n_timesteps)  

        return mean_x, y[:, :, :max_length]

    def compute_loss(self, x, x_lengths, unit, hint):
        x, x_lengths = self.relocate_input([x, x_lengths])
        x_mask = sequence_mask(x_lengths, max_length=128).unsqueeze(1).to(x.dtype)
        # print(x.shape, x_lengths.shape, x_mask.shape)
        diff_loss = self.decoder.compute_loss(x, x_mask, unit, hint = hint)
        return diff_loss
    

class ControlledDiffVC_emo(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, enc_dim, spk_dim, use_ref_t, dec_dim, 
                 beta_min, beta_max):
        super().__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.enc_dim = enc_dim
        self.spk_dim = spk_dim
        self.use_ref_t = use_ref_t
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.decoder = ControlledDiffusion_emo(n_feats, dec_dim, spk_dim,
                                 beta_min, beta_max)


    @torch.no_grad()
    def forward(self, x, x_lengths, unit, n_timesteps, hint, emo,
                mode='ml'):

        x, x_lengths = self.relocate_input([x, x_lengths])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        mean = unit
        mean_x = self.decoder.compute_diffused_mean(x, x_mask, 1.0)  ###x和mean加权 t=1.0

        b = x.shape[0]
        max_length = int(x_lengths.max())
        max_length_new = fix_len_compatibility(max_length)
        x_mask_new = sequence_mask(x_lengths, max_length_new).unsqueeze(1).to(x.dtype)
        mean_new = torch.zeros((b, 768, max_length_new), dtype=x.dtype, 
                                device=x.device)
        mean_x_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                  device=x.device)
        for i in range(b):
            mean_new[i, :, :x_lengths[i]] = mean[i, :, :x_lengths[i]]
            mean_x_new[i, :, :x_lengths[i]] = mean_x[i, :, :x_lengths[i]]

        z = mean_x_new
        z += torch.randn_like(mean_x_new, device=mean_x_new.device)

        y = self.decoder(z, x_mask_new, mean_new, n_timesteps, mode, hint, emo) 

        return mean_x, y[:, :, :max_length]

    def compute_loss(self, x, x_lengths, unit, hint, emo):
        x, x_lengths = self.relocate_input([x, x_lengths])
        x_mask = sequence_mask(x_lengths, max_length=128).unsqueeze(1).to(x.dtype)
        # print(x.shape, x_lengths.shape, x_mask.shape)
        diff_loss = self.decoder.compute_loss(x, x_mask, unit, hint = hint, emo=emo)
        return diff_loss
    

class DiffVC_light(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, enc_dim, spk_dim, use_ref_t, dec_dim, 
                 beta_min, beta_max):
        super(DiffVC_light, self).__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.enc_dim = enc_dim
        self.spk_dim = spk_dim
        self.use_ref_t = use_ref_t
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.decoder = Diffusion_light(n_feats, dec_dim, spk_dim, use_ref_t, 
                                 beta_min, beta_max)


    @torch.no_grad()
    def forward(self, x, x_lengths, x_ref, x_ref_lengths, c, unit, n_timesteps, 
                mode='ml', dpm=False):
        
        x, x_lengths = self.relocate_input([x, x_lengths])
        x_ref, x_ref_lengths, c = self.relocate_input([x_ref, x_ref_lengths, c])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        x_ref_mask = sequence_mask(x_ref_lengths).unsqueeze(1).to(x_ref.dtype)

        mean = unit
        mean_x = self.decoder.compute_diffused_mean(x, x_mask, 1.0) 

        b = x.shape[0]
        max_length = int(x_lengths.max())
        max_length_new = fix_len_compatibility(max_length)
        x_mask_new = sequence_mask(x_lengths, max_length_new).unsqueeze(1).to(x.dtype)
        mean_new = torch.zeros((b, 768, max_length_new), dtype=x.dtype, 
                                device=x.device)
        mean_x_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                  device=x.device)
        for i in range(b):
            mean_new[i, :, :x_lengths[i]] = mean[i, :, :x_lengths[i]]
            mean_x_new[i, :, :x_lengths[i]] = mean_x[i, :, :x_lengths[i]]

        z = mean_x_new
        z += torch.randn_like(mean_x_new, device=mean_x_new.device)

        y = self.decoder(z, x_mask_new, mean_new, x_ref, x_ref_mask, c, 
                         n_timesteps, mode, dpm) 
        return mean_x, y[:, :, :max_length]

    def compute_loss(self, x, x_lengths, x_ref, c, unit):
 
        x, x_lengths, x_ref, c = self.relocate_input([x, x_lengths, x_ref, c])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        diff_loss = self.decoder.compute_loss(x, x_mask, unit, x_ref, c)
        return diff_loss

    def forward_streaming(self, x, x_lengths, x_ref, x_ref_lengths, c, unit, n_timesteps, 
                mode='ml', dpm=False, out_size=None):
        
        assert x.shape[0] == 1  # streaming inference only support batch size 1
        
        x, x_lengths = self.relocate_input([x, x_lengths])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype) #bs=1时 mask全1
        x_ref, x_ref_lengths, c = self.relocate_input([x_ref, x_ref_lengths, c])
        x_ref_mask = sequence_mask(x_ref_lengths).unsqueeze(1).to(x_ref.dtype)

        mean = unit #content
        mean_x = self.decoder.compute_diffused_mean(x, x_mask, 1.0) #X_T

        max_length = int(x_lengths.max()) #即x_length[0]
        max_length_new = fix_len_compatibility_lg(max_length)
        
        x_mask_new = sequence_mask(x_lengths, max_length=max_length_new).unsqueeze(1).to(x.dtype)  
        mean_new = torch.zeros((1, 768, max_length_new), dtype=x.dtype, device=x.device)
        mean_x_new = torch.zeros((1, self.n_feats, max_length_new), dtype=x.dtype, device=x.device)

        mean_new[0, :, :x_lengths[0]] = mean[0, :, :x_lengths[0]] 
        mean_x_new[0, :, :x_lengths[0]] = mean_x[0, :, :x_lengths[0]] #padding

        out_size = fix_len_compatibility_lg(out_size)
        (num_chunks, chunk_lengths, 
         start_frames, end_frames, lpad, rpad) = generate_idxs(max_length, out_size)
        
        # z = mean_x_new
        # z += torch.randn_like(mean_x_new, device=mean_x_new.device)
        
        # y = self.decoder(z, x_mask_new, mean_new, x_ref, x_ref_mask, c, n_timesteps, mode, dpm) 
        # return mean_x, y[:, :, :max_length]
        
        seed = torch.randint(0, 1000000, (1,))
        # print(seed)
        for i in range(num_chunks):
            # print(seed)
            torch.manual_seed(seed.item())

            lp = lpad[i]
            rp = rpad[i]
            l = chunk_lengths[i]

            # start_idx should be divisible by downsampling factor
            start_idx = fix_len_compatibility_lg(start_frames[i] - lp, type='floor')
            # adjust left padding part according to start_idx
            lp += start_frames[i] - lp - start_idx 
            end_idx = min(max_length_new, fix_len_compatibility_lg(end_frames[i] + rp))
            
            # print(f"Iteration {i}: start_idx={start_idx}, end_idx={end_idx}, lp={lp}, l={l}")

            mean_new_cut = mean_new[:, :, start_idx:end_idx]
            mean_x_new_cut = mean_x_new[:, :, start_idx:end_idx]
            x_mask_new_cut = x_mask_new[:, :, start_idx:end_idx]

            z_cut = mean_x_new_cut + torch.randn_like(
                mean_x_new_cut, device=mean_x_new_cut.device) 
            # z_cut = z[:, :, start_idx:end_idx]
            
            decoder_output_cut = self.decoder(z_cut, x_mask_new_cut, mean_new_cut, x_ref, x_ref_mask, c, 
                         n_timesteps, mode, dpm) 
            yield (mean_x_new_cut[:, :, lp:lp + l], decoder_output_cut[:, :, lp:lp + l])

import math
def fix_len_compatibility_lg(length, num_downsamplings_in_unet=2, type='ceil'):
    factor = 2**num_downsamplings_in_unet
    if type == 'ceil':
        return int(math.ceil(length / factor) * factor)
    elif type == 'floor':
        return int(math.floor(length / factor) * factor)
    else:
        raise ValueError(f'Wrong type: {type}')

def generate_idxs(mel_length, chunk_frames, overlap = 0.05):

    durations = torch.ones(mel_length).int().cuda()
    cum_sum = durations.cumsum(dim=0).int()

    idx = torch.div(cum_sum, chunk_frames, rounding_mode='trunc').int()
    num_chunks = idx.max() + 1
    lengths = torch.zeros((num_chunks),
                          device=durations.device,
                          dtype=torch.int)
    lpad = torch.zeros_like(lengths, device=lengths.device)
    rpad = torch.zeros_like(lengths, device=lengths.device)
    for i in range(num_chunks):
        duration_chunk_mask = i == idx
        duration_chunk = durations * duration_chunk_mask
        duration_chunk_frames = duration_chunk.sum()
        if i > 0:
            lpad[i] = int(chunk_frames * overlap)
        if i < num_chunks - 1:
            rpad[i] = int(chunk_frames * overlap)
        lengths[i] = duration_chunk_frames
    end_frames = lengths.cumsum(dim=0)
    start_frames = end_frames - lengths
    return num_chunks, lengths, start_frames, end_frames, lpad, rpad
