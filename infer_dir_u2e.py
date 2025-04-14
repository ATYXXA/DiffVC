import os
from tqdm import tqdm
import json
import numpy as np
import torch

import params
from model.vc import DiffVC, UF2E
from model.utils import sequence_mask
from utils import save_audio
from model.utils import repeat_expand_2d

import librosa
from librosa.filters import mel as librosa_mel_fn

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
dim = params.enc_dim

sampling_rate = 22050
mel_basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

out_dir = 'es_u2e_10_20'
enc_path = 'logs_uf2e/enc_200.pt'
vc_path = 'logs_dec24/vc_250.pt'


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

def normalize_(arr):
    min_value = np.min(arr)
    max_value = np.max(arr)
    normalized_array = (arr - min_value) / (max_value - min_value)
    return normalized_array

import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
hifigan_universal = None


sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path
enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt') 
spk_encoder.load_model(enc_model_fpath, device="cuda")
def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed

print(f'Loading ckpt from {enc_path}.')
uf2e = UF2E(100, 768, channels, filters, heads, layers, kernel, dropout, window_size).cuda()
uf2e.load_state_dict(torch.load(enc_path))
uf2e.eval()

def process_one(line):
    src_name, tgt_path, pred = line.strip().split('|')
    pred = eval(pred)
    
    unit = np.array(pred['units'])
    unit = torch.from_numpy(unit)
    f0 = np.array(pred['f0'])
    f0 = normalize_(f0)
    f0 = torch.from_numpy(f0).unsqueeze(0)

    unit_x = unit.unsqueeze(0).long().cuda()
    f0 = f0.unsqueeze(0).float().cuda()
    unit_lengths = torch.LongTensor([unit_x.shape[-1]]).cuda()
    unit_mask = sequence_mask(unit_lengths).unsqueeze(1).to(unit_x.dtype)
    unit_source = uf2e(unit_x, unit_mask, f0)


    unit_source = unit_source.squeeze()
    target_len = int(unit_source.shape[-1] * 1.72265625) # 1.72265625 (22050/256)/(16000/320) 
    unit_source = repeat_expand_2d(unit_source,target_len=target_len, mode='linear').unsqueeze(0)

    mel_source = torch.zeros((1,80,target_len)).cuda()
    mel_source_lengths = torch.LongTensor([target_len]).cuda()

    tgt_emb = torch.from_numpy(get_embed(tgt_path)).float().unsqueeze(0)


    mel_avg, mel_rec = model(mel_source, mel_source_lengths, mel_source, mel_source_lengths, tgt_emb, unit_source, 
                             n_timesteps=50)
    mel_synth_np = mel_rec.detach().cpu().squeeze().numpy()
    mel_source_np = mel_rec.detach().cpu().squeeze().numpy()  
    mel_output = torch.from_numpy(mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)   
    audio = hifigan_universal.forward(mel_output.cuda())

    
    tgt = os.path.basename(tgt_path).split('_')[0]
    os.makedirs(f'{out_dir}/{tgt}', exist_ok=True)
    out_name = f'{src_name}'
    save_audio(f'{out_dir}/{tgt}/{out_name}.wav', sampling_rate, audio, normalize=True)                        


def search_wavs(rootdir):
    paths=[]
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".wav"):
                paths.append(os.path.join(root,file))
    
    return paths


def process_list(filepath):
    print(filepath)
    with open(filepath,'r') as ff:
        lines=ff.readlines()
    for line in tqdm(lines):
        process_one(line)


if __name__ == "__main__":

    # random_seed=params.seed
    # torch.manual_seed(random_seed)
    # np.random.seed(random_seed)

    os.makedirs(out_dir, exist_ok=True)

    print('Initializing and loading models...')
    model = DiffVC(params.n_mels, params.channels, params.filters, params.heads, 
                   params.layers, params.kernel, params.dropout, params.window_size, 
                   params.enc_dim, params.spk_dim, False, params.dec_dim,
                   params.beta_min, params.beta_max)

    print(f'Loading ckpt from {vc_path}.')
    model = model.cuda()
    model.load_state_dict(torch.load(vc_path))
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

    print(out_dir)
    with torch.no_grad():
        process_list('/home/huangf79/projects/DISSC/src-tgt-pred-10-20.txt')

                    