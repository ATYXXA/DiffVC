# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import torch
#import torchaudio
import numpy as np
from librosa.filters import mel as librosa_mel_fn

from model.base import BaseModule


def mse_loss(x, y, mask, n_feats):
    loss = torch.sum(((x - y)**2) * mask)
    return loss / (torch.sum(mask) * n_feats)

def mae_loss(x, y, mask, n_feats):
    loss = torch.sum((abs(x - y)) * mask)
    return loss / (torch.sum(mask) * n_feats)


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1

def repeat_expand_2d(content, target_len, mode = 'nearest'):
    # content : [h, t]
    return repeat_expand_2d_left(content, target_len) if mode == 'left' else repeat_expand_2d_other(content, target_len, mode)

def repeat_expand_2d_left(content, target_len):
    # content : [h, t]

    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(content.device)
    temp = torch.arange(src_len+1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos+1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]

    return target

# mode : 'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'
def repeat_expand_2d_other(content, target_len, mode = 'nearest'):
    # content : [h, t]
    content = content[None,:,:]
    target = torch.nn.functional.interpolate(content,size=target_len,mode=mode)[0]
    return target

class PseudoInversion(BaseModule):
    def __init__(self, n_mels, sampling_rate, n_fft):
        super(PseudoInversion, self).__init__()
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=8000)
        mel_basis_inverse = np.linalg.pinv(mel_basis)
        mel_basis_inverse = torch.from_numpy(mel_basis_inverse).float()
        self.register_buffer("mel_basis_inverse", mel_basis_inverse)

    def forward(self, log_mel_spectrogram):
        mel_spectrogram = torch.exp(log_mel_spectrogram)
        stftm = torch.matmul(self.mel_basis_inverse, mel_spectrogram)
        return stftm


class InitialReconstruction(BaseModule):
    def __init__(self, n_fft, hop_size):
        super(InitialReconstruction, self).__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        window = torch.hann_window(n_fft).float()
        self.register_buffer("window", window)

    def forward(self, stftm):
        real_part = torch.ones_like(stftm, device=stftm.device)
        imag_part = torch.zeros_like(stftm, device=stftm.device)
        stft = torch.stack([real_part, imag_part], -1)*stftm.unsqueeze(-1)
        ###
        complex_stft = torch.view_as_complex(stft)
        ###
        istft = torch.istft(complex_stft, n_fft=self.n_fft,  
                           hop_length=self.hop_size, win_length=self.n_fft, 
                           window=self.window, center=True)
        # Changed in version 2.0: Real datatype inputs are no longer supported. 
        # Input must now have a complex datatype, as returned by stft(..., return_complex=True).
        # RuntimeError: sistft requires a complex-valued input tensor matching the output from stft with return_complex=True.
        return istft.unsqueeze(1)


# Fast Griffin-Lim algorithm as a PyTorch module
class FastGL(BaseModule):
    def __init__(self, n_mels, sampling_rate, n_fft, hop_size, momentum=0.99):
        super(FastGL, self).__init__()
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.momentum = momentum
        self.pi = PseudoInversion(n_mels, sampling_rate, n_fft)
        self.ir = InitialReconstruction(n_fft, hop_size)
        window = torch.hann_window(n_fft).float()
        self.register_buffer("window", window)

    @torch.no_grad()
    def forward(self, s, n_iters=32):
        c = self.pi(s)
        x = self.ir(c)
        x = x.squeeze(1)
        c = c.unsqueeze(-1)
        prev_angles = torch.zeros_like(c, device=c.device)
        for _ in range(n_iters):        
            s = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_size, 
                           win_length=self.n_fft, window=self.window, 
                           center=True, return_complex=True) ###return_complex=False会报warning建议true+view_as_real
            ###
            s = torch.view_as_real(s)
            ###
            real_part, imag_part = s.unbind(-1)
            stftm = torch.sqrt(torch.clamp(real_part**2 + imag_part**2, min=1e-8))
            angles = s / stftm.unsqueeze(-1)
            s = c * (angles + self.momentum * (angles - prev_angles))
            ###
            complex_s = torch.view_as_complex(s)
            ###
            x = torch.istft(complex_s, n_fft=self.n_fft, hop_length=self.hop_size, 
                                            win_length=self.n_fft, window=self.window, 
                                            center=True, return_complex=False) ###return_complex=False (Default: False)
            
            prev_angles = angles
        return x.unsqueeze(1)
    

