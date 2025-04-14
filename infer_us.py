# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import numpy as np
# from tqdm import tqdm

import torch
# from torch.utils.data import DataLoader

import params
from data import  VCTKDecDataset#, VCDecBatchCollate
from model.vc_0 import DiffVC
# from model.utils import FastGL
from utils import save_plot, save_audio

n_mels = params.n_mels
sampling_rate = params.sampling_rate
n_fft = params.n_fft
hop_size = params.hop_size

channels = params.channels
filters = params.filters
layers = params.layers
kernel = params.kernel
dropout = params.dropout
heads = params.heads
window_size = params.window_size
enc_dim = params.enc_dim

dec_dim = params.dec_dim
spk_dim = params.spk_dim
use_ref_t = params.use_ref_t
beta_min = params.beta_min
beta_max = params.beta_max

random_seed = params.seed
test_size = params.test_size


data_dir = './dataset/vctk'
val_file = 'filelists/valid.txt'
exc_file = 'filelists/exceptions_vctk.txt'

# log_dir = 'logs_dec'
# enc_dir = 'logs_enc'
# epochs = 100
# batch_size = 16
# learning_rate = 1e-4
# save_every = 1

out_dir = 'outputs_diffvc'
# vc_path = 'checkpts/vc/vc_miu_95.pt'
vc_path = 'checkpts/vc/vc_vctk_wodyn.pt'
def noise_median_smoothing(x, w=5):
    y = np.copy(x)
    x = np.pad(x, w, "edge")
    for i in range(y.shape[0]):
        med = np.median(x[i:i+2*w+1])
        y[i] = min(x[i+w+1], med)
    return y

def mel_spectral_subtraction(mel_synth, mel_source, spectral_floor=0.02, silence_window=5, smoothing_window=5):
    mel_len = mel_source.shape[-1]
    energy_min = 100000.0
    i_min = 0
    for i in range(mel_len - silence_window):
        energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i+silence_window]))
        if energy_cur < energy_min:
            i_min = i
            energy_min = energy_cur
    estimated_noise_energy = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min+silence_window]), axis=-1)
    if smoothing_window is not None:
        estimated_noise_energy = noise_median_smoothing(estimated_noise_energy, smoothing_window)
    mel_denoised = np.copy(mel_synth)
    for i in range(mel_len):
        signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
        estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
        mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
    return mel_denoised


if __name__ == "__main__":

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    os.makedirs(out_dir, exist_ok=True)

    print('Initializing data loaders...')
    dataset = VCTKDecDataset(data_dir)

    print('Initializing and loading models...')
    model = DiffVC(n_mels, channels, filters, heads, layers, kernel, 
                   dropout, window_size, enc_dim, spk_dim, use_ref_t, 
                   dec_dim, beta_min, beta_max).cuda()
    
    print(f'Loading ckpt from {vc_path}.')
    model.load_state_dict(torch.load(vc_path))

    print('Encoder - Number of parameters = %.2fm' % (model.encoder.nparams/1e6))
    print('Decoder - Number of parameters = %.2fm' % (model.decoder.nparams/1e6))
    torch.backends.cudnn.benchmark = True

    # loading HiFi-GAN vocoder
    import sys
    sys.path.append('hifi-gan/')
    from env import AttrDict
    from models import Generator as HiFiGAN
    
    import json
    hfg_path = 'checkpts/vocoder/' # HiFi-GAN path
    with open(hfg_path + 'config.json') as f:
        h = AttrDict(json.load(f))

    hifigan_universal = HiFiGAN(h).cuda()
    hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])
    hifigan_universal.eval()
    hifigan_universal.remove_weight_norm()

    # model.eval()
    
    with torch.no_grad():
        spk_dict, utt_dict=dataset.get_unseen_dataset()
        for us_spk in dataset.unseen_speakers:
            print(f"Processing spk {us_spk}")
            utts = spk_dict[us_spk]
            tgt_spks = [spk for spk in dataset.unseen_speakers if spk!=us_spk]
            for utt in utts:
                mel, _ = utt_dict[utt]
                mel = mel.unsqueeze(0).float().cuda()
                mel_lengths = torch.LongTensor([mel.shape[-1]]).cuda()
                for tgt_spk in tgt_spks:
                    tgt_mel, tgt_emb = utt_dict[spk_dict[tgt_spk][0]]
                    tgt_mel = tgt_mel.unsqueeze(0).float().cuda()
                    tgt_mel_lengths = torch.LongTensor([tgt_mel.shape[-1]]).cuda()
                    tgt_emb = tgt_emb.unsqueeze(0).float().cuda()

                    mel_avg, mel_rec = model(mel, mel_lengths, tgt_mel, tgt_mel_lengths, tgt_emb, 
                                n_timesteps=30)
                    
                    mel_synth_np = mel_rec.detach().cpu().squeeze().numpy()
                    mel_source_np = mel_rec.detach().cpu().squeeze().numpy()  # check一下mel_source 好像谱减法根本用不上这个变量
                    mel = torch.from_numpy(mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)
    
                    # audio = hifigan_universal.forward(mel.cuda()).cpu().squeeze().clamp(-1, 1) 
                    audio = hifigan_universal.forward(mel.cuda())
                    os.makedirs(f'{out_dir}/{us_spk}', exist_ok=True)
                    save_audio(f'{out_dir}/{us_spk}/{us_spk}_{tgt_spk}_{utt[len(us_spk)+1:]}.wav',
                               sampling_rate, audio)
                
            
    '''
    print('Inference...\n')
    with torch.no_grad():
        mels = dataset.get_unseen_dataset()
        for i, (mel, c) in enumerate(mels):
            if i >= test_size:
                break
            mel = mel.unsqueeze(0).float().cuda()
            c = c.unsqueeze(0).float().cuda()
            mel_lengths = torch.LongTensor([mel.shape[-1]]).cuda()
            mel_avg, mel_rec = model(mel, mel_lengths, mel, mel_lengths, c, 
                                        n_timesteps=100)
            # if epoch == save_every:
            if save_every:
                save_plot(mel.squeeze().cpu(), f'{log_dir}/original_{i}.png')
                audio = fgl(mel)
                save_audio(f'{log_dir}/original_{i}.wav', sampling_rate, audio)
            save_plot(mel_avg.squeeze().cpu(), f'{log_dir}/average_{i}.png')
            audio = fgl(mel_avg)
            save_audio(f'{log_dir}/average_{i}.wav', sampling_rate, audio)
            save_plot(mel_rec.squeeze().cpu(), f'{log_dir}/reconstructed_{i}.png')
            audio = fgl(mel_rec)
            save_audio(f'{log_dir}/reconstructed_{i}.wav', sampling_rate, audio)
        '''

