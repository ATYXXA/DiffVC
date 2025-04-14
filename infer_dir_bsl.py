import os
from tqdm import tqdm
import json
import numpy as np
import torch

import params
from model.vc_0 import DiffVC
from utils import save_audio
from model.utils import repeat_expand_2d

import librosa
from librosa.filters import mel as librosa_mel_fn


sampling_rate = 22050
# mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)
mel_basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

src_dir = 'subreal3'
out_dir = 'subreal_hweg'
vc_path = 'checkpts/vc/vc_vctk_wodyn.pt'
enc_dir = 'logs_enc'
# vc_path = 'checkpts/vc/vc_miu_95.pt'


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


import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
hifigan_universal = None


mel_dir = './dataset/vctk/mels'
emb_dir = './dataset/vctk/embeds'
spks = os.listdir(mel_dir)


def get_vc_data(spk, audio_id):
    mels = get_mels(audio_id, spk)
    embed = get_embed(audio_id, spk)
    return (mels, embed)

def get_mels(audio_id, spk):
    mel_path = os.path.join(mel_dir, spk, audio_id + '_mel.npy')
    mels = np.load(mel_path)
    mels = torch.from_numpy(mels).float()
    return mels

def get_embed(audio_id, spk):
    embed_path = os.path.join(emb_dir, spk, audio_id + '_embed.npy')
    embed = np.load(embed_path)
    embed = torch.from_numpy(embed).float()
    return embed


def get_mel(wav_path):
    wav, _ = librosa.load(wav_path, sr=22050)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram


def process_one(src_path):
    src_name=src_path.split('/')[-1].split('.')[0]
    mel_source = torch.from_numpy(get_mel(src_path)).float().unsqueeze(0)
    mel_source = mel_source.cuda()
    
    mel_source_lengths = torch.LongTensor([mel_source.shape[-1]])
    mel_source_lengths = mel_source_lengths.cuda()

    spk = np.random.choice(spks,1)[0]
    utts = os.listdir(f'{mel_dir}/{spk}')
    audio_id = np.random.choice(utts,1)[0][:-8]

    tgt_mel, tgt_emb = get_vc_data(spk,audio_id)
    tgt_mel = tgt_mel.unsqueeze(0).float().cuda()
    tgt_mel_lengths = torch.LongTensor([tgt_mel.shape[-1]]).cuda()
    tgt_emb = tgt_emb.unsqueeze(0).float().cuda()

    mel_avg, mel_rec = model(mel_source, mel_source_lengths, tgt_mel, tgt_mel_lengths, tgt_emb, n_timesteps=30)
    mel_synth_np = mel_rec.detach().cpu().squeeze().numpy()
    mel_source_np = mel_rec.detach().cpu().squeeze().numpy()  
    mel_output = torch.from_numpy(mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)   
    audio = hifigan_universal.forward(mel_output.cuda())

    os.makedirs(f'{out_dir}', exist_ok=True)
    out_name='3'+src_name[1:]
    save_audio(f'{out_dir}/{out_name}.wav', sampling_rate, audio, normalize=False)                        


def search_wavs(rootdir):
    paths=[]
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".wav"):
                paths.append(os.path.join(root,file))
    
    return paths

def process_dir(dir):
    print("Processing...")
    for wav in tqdm(search_wavs(dir)):
        process_one(wav)


if __name__ == "__main__":

    # random_seed=params.seed
    # torch.manual_seed(random_seed)
    # np.random.seed(random_seed)

    os.makedirs(out_dir, exist_ok=True)

    print('Initializing and loading models...')
    model = DiffVC(params.n_mels, params.channels, params.filters, params.heads, 
                   params.layers, params.kernel, params.dropout, params.window_size, 
                   params.enc_dim, params.spk_dim, True, params.dec_dim,
                   params.beta_min, params.beta_max)

    print(f'Loading ckpt from {vc_path}.')
    model = model.cuda()
    model.load_state_dict(torch.load(vc_path))
    model.load_encoder(os.path.join(enc_dir, 'enc.pt'))
    model.eval()

    print('Decoder - Number of parameters = %.2fm' % (model.decoder.nparams/1e6))
    torch.backends.cudnn.benchmark = True

    hfg_path = 'checkpts/vocoder/' 
    with open(hfg_path + 'config.json') as f:
        h = AttrDict(json.load(f))
    hifigan_universal = HiFiGAN(h).cuda()
    hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])
    hifigan_universal.eval()
    hifigan_universal.remove_weight_norm()    

    with torch.no_grad():
        process_dir(src_dir)

                    