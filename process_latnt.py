import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Pool
import multiprocessing
from pathlib import Path
import librosa
import torch
from model.encodec import EncodecWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encodec = EncodecWrapper()
encodec.to(device)

def get_emb_idx(wav_path):
    wav, sr = librosa.load(wav_path, sr=24000)
    wav = torch.from_numpy(wav).to(device)
    emb, idx, _ = encodec(wav, input_sample_hz=sr, return_encoded=True)  
    emb = emb.detach().cpu().numpy()
    idx = idx.detach().cpu().numpy()
    return emb, idx

def process_one(wav_path):
    emb, idx = get_emb_idx(wav_path)

    file_name = os.path.splitext(os.path.basename(wav_path))[0]
    speaker_name = os.path.basename(os.path.dirname(wav_path))
    save_path_1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(wav_path))), 'latnts', f'{speaker_name}', f"{file_name}_emb.npy")
    save_path_2 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(wav_path))), 'latnts', f'{speaker_name}', f"{file_name}_idx.npy")

    os.makedirs(os.path.dirname(save_path_1), exist_ok=True)
    
    # print(emb.shape, idx.shape) # (202, 128) (202, 8)
    np.save(save_path_1, emb)
    np.save(save_path_2, idx)


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
    data_folder = "dataset/vctk"  
    num_processes = 4  
    multiprocessing.set_start_method('spawn', force=True)
    process_all(data_folder, num_processes)

