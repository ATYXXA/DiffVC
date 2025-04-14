import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Pool
import multiprocessing
import librosa
import torch

from textless.data.f0_preprocess import get_f0, align_f0_to_durations, F0_FRAME_SPACE
f0_code_ratio = 4.0

# from textless.data.speech_encoder import SpeechEncoder
# cnt_encoder = SpeechEncoder.by_name(dense_model_name='hubert-base-ls960-layer-9', quantizer_model_name='kmeans',
#                                     vocab_size=500, deduplicate=False, need_f0=True).to('cuda')

# def get_f0_path0(wav_path):
#     wav, _ = librosa.load(wav_path, sr=16000)
#     wav16k = torch.from_numpy(wav).to(cnt_encoder.device)
#     f0 = cnt_encoder(wav16k)['f0']
#     f0 = f0.detach().cpu().numpy()
#     return f0

def get_f0_path(wav_path):
    wav, _ = librosa.load(wav_path, sr=16000)
    f0 = torch.from_numpy(get_f0(wav)).float()
    wavlen = wav.shape[0] 
    durlen = int((wavlen-80)/320)
    dur = torch.ones(durlen)
    f0 = align_f0_to_durations(f0, dur, f0_code_ratio, 5*f0_code_ratio)
    return f0.cpu().numpy()

def process_one(wav_path):
    f0 = get_f0_path(wav_path)

    file_name = os.path.splitext(os.path.basename(wav_path))[0]
    speaker_name = os.path.basename(os.path.dirname(wav_path))
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(wav_path))), 'f0s', f'{speaker_name}', f"{file_name}_f0.npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    np.save(save_path, f0)


def process_all(root_folder, num_processes):
    wav_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".wav"):
                wav_paths.append(os.path.join(root, file))

    with Pool(num_processes) as pool:
        list(tqdm(pool.imap(process_one, wav_paths), total=len(wav_paths)))
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    data_folder = "dataset/ESD"  
    num_processes = 4
    multiprocessing.set_start_method('spawn', force=True)
    process_all(data_folder, num_processes)

