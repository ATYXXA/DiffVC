import torch
import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
from tqdm import tqdm
import os
from utils import save_audio


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

# 
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


datadir = 'dataset/vctk_unseen'
outputdir = 'oracle'
if __name__ == "__main__":

    os.makedirs(outputdir, exist_ok=True)

    for spk in os.listdir(datadir):
        print(f'Processing spk {spk}')

        for utt in os.listdir(f'{datadir}/{spk}'):
            source_path = f'{datadir}/{spk}/{utt}'
            mel = torch.from_numpy(get_mel(source_path))
            audio = hifigan_universal.forward(mel.unsqueeze(0).cuda())
            output_path = f'{outputdir}/{spk}/{utt}'

            os.makedirs(f'{outputdir}/{spk}', exist_ok=True)
            save_audio(output_path, 22050, audio)
            
    print("Done!")
            
'''
path=os
mel=get_mel(path)
audio=vocoder(mel)
save_audio()

audio = hifigan_universal.forward(mel.cuda())
os.makedirs(f'{out_dir}/{us_spk}', exist_ok=True)
save_audio(f'{out_dir}/{us_spk}/{us_spk}_{tgt_spk}_{utt[len(us_spk)+1:]}.wav',
            sampling_rate, audio, normalize=False)
'''