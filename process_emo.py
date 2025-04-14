import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Pool
import multiprocessing
from pathlib import Path
import librosa
import torch


label_map={"中立":0,    "生气":1,   "快乐":2,  "伤心":3, "惊喜":4,
           "Neutral":0, "Angry":1, "Happy":2, "Sad":3,  "Surprise":4}

map={}

def file2map(path):
    with open(path,'r') as f:
        lines=f.readlines()
    global map
    for line in lines:
        info = line.strip().split()
        map[info[0]]=label_map[info[-1]]
    
def one_hot(label,num_class=5):
    ''' label starts from 0 '''
    assert label < num_class
    return torch.eye(num_class)[label]


def get_emo(wav_path):
    # print(wav_path.split('/')[-1][:-4])
    label = map[wav_path.split('/')[-1][:-4]]
    emo = one_hot(label) 
    emo = emo.numpy()
    return emo

def process_one(wav_path):
    emo = get_emo(wav_path)

    file_name = os.path.splitext(os.path.basename(wav_path))[0]
    speaker_name = os.path.basename(os.path.dirname(wav_path))
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(wav_path))), 'emos', f'{speaker_name}', f"{file_name}_emo.npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    np.save(save_path, emo)


def process_all(root_folder, num_processes):
    wav_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".wav"):
                wav_paths.append(os.path.join(root, file))
            if file.endswith(".txt"):
                print(f"adding {file} to map")
                file2map(os.path.join(root, file))

    with Pool(num_processes) as pool:
        list(tqdm(pool.imap(process_one, wav_paths), total=len(wav_paths)))
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    data_folder = "dataset/ESD"  
    num_processes = 4
    process_all(data_folder, num_processes)

