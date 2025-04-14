import os
import numpy as np
from tqdm import tqdm

import torch
import params
from model.vc_0 import DiffVC
from utils import  save_audio

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
enc_dim = params.enc_dim

dec_dim = params.dec_dim
spk_dim = params.spk_dim
use_ref_t = params.use_ref_t
beta_min = params.beta_min
beta_max = params.beta_max

random_seed = params.seed
test_size = params.test_size




out_dir = 'diffvc_test'
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


mel_basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)
def get_mel(wav_path):
    wav, _ = librosa.load(wav_path, sr=22050)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram

import sys
sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path
enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt') 
spk_encoder.load_model(enc_model_fpath, device="cuda")
def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed


def process_one(src_path, tgt_path):
    src_name = src_path.split('/')[-1].split('.')[0]
    mel_source = torch.from_numpy(get_mel(src_path)).float().unsqueeze(0)
    mel_source = mel_source.cuda()
    
    mel_source_lengths = torch.LongTensor([mel_source.shape[-1]])
    mel_source_lengths = mel_source_lengths.cuda()

    tgt_mel = torch.from_numpy(get_mel(tgt_path)).float().unsqueeze(0)
    tgt_mel = tgt_mel.cuda()
    tgt_mel_lengths = torch.LongTensor([tgt_mel.shape[-1]]).cuda()
    tgt_emb = torch.from_numpy(get_embed(tgt_path)).float().unsqueeze(0)

    mel_avg, mel_rec = model(mel_source, mel_source_lengths, tgt_mel, tgt_mel_lengths, tgt_emb, 
                n_timesteps=50)
    
    mel_synth_np = mel_rec.detach().cpu().squeeze().numpy()
    mel_source_np = mel_rec.detach().cpu().squeeze().numpy()  
    mel_output = torch.from_numpy(mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)   
    audio = hifigan_universal.forward(mel_output.cuda())

    
    tgt = os.path.basename(tgt_path).split('_')[0]
    os.makedirs(f'{out_dir}/{tgt}', exist_ok=True)
    out_name = f'{src_name}_to_{tgt}'
    save_audio(f'{out_dir}/{tgt}/{out_name}.wav', sampling_rate, audio, normalize=True)        

def process_list(filepath):
    with open(filepath,'r') as ff:
        lines=ff.readlines()
    for line in tqdm(lines):
        src, tgt = line.strip().split('|')
        process_one(src, tgt)



if __name__ == "__main__":

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    os.makedirs(out_dir, exist_ok=True)

    print('Initializing and loading models...')
    model = DiffVC(n_mels, channels, filters, heads, layers, kernel, 
                   dropout, window_size, enc_dim, spk_dim, use_ref_t, 
                   dec_dim, beta_min, beta_max).cuda()
    
    print(f'Loading ckpt from {vc_path}.')
    model.load_state_dict(torch.load(vc_path))
    model.eval()

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

    
    
    with torch.no_grad():
        # process_dir(src_dir)
        process_list('/home/huangf79/projects/UnitSpeech/notebooks/src-tgt.txt')

