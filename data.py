# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import random
import numpy as np
import torch
import tgt

from params import seed as random_seed
from params import n_mels, train_frames
from model.utils import repeat_expand_2d

def get_test_speakers():
    test_speakers = ['1401', '2238', '3723', '4014', '5126', 
                     '5322', '587', '6415', '8057', '8534']
    return test_speakers


def get_vctk_unseen_speakers():
    unseen_speakers = ['p252', 'p261', 'p241', 'p238', 'p243',
                       'p294', 'p334', 'p343', 'p360', 'p362']
    return unseen_speakers


def get_vctk_unseen_sentences():
    unseen_sentences = ['001', '002', '003', '004', '005']
    return unseen_sentences


def get_esd_unseen_speakers():
    unseen_speakers = ['0001', '0011']
    # unseen_speakers = [       '0002','0003','0004','0005',
                    #    '0006','0007','0008','0009','0010',
                    #    '0011','0012','0013','0014','0015',
                    #    '0016','0017','0018','0019','0020']
    return unseen_speakers
def get_esd_unseen_sentences():
    unseen_sentences = ['000001', '000351', '000701', '001051', '001401',
                        '000002', '000352', '000702', '001052', '001402',
                        '000003', '000353', '000703', '001053', '001403' ]
    return unseen_sentences

# exclude utterances where MFA couldn't recognize some words
def exclude_spn(data_dir, spk, mel_ids):
    res = []
    for mel_id in mel_ids:
        textgrid = mel_id + '.TextGrid'
        t = tgt.io.read_textgrid(os.path.join(data_dir, 'textgrids', spk, textgrid))
        t = t.get_tier_by_name('phones')
        spn_found = False
        for i in range(len(t)):
            if t[i].text == 'spn':
                spn_found = True
                break
        if not spn_found:
            res.append(mel_id)
    return res


# LibriTTS dataset for training "average voice" encoder
class VCEncDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, exc_file, avg_type):
        self.mel_x_dir = os.path.join(data_dir, 'mels')
        self.mel_y_dir = os.path.join(data_dir, 'mels_%s' % avg_type)

        self.test_speakers = get_test_speakers()
        self.speakers = [spk for spk in os.listdir(self.mel_x_dir) 
                         if spk not in self.test_speakers]
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_mel.npy' for e in exceptions]
        self.test_info = []
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_x_dir, spk))
            mel_ids = [m[:-8] for m in mel_ids if m not in self.exceptions]
            mel_ids = exclude_spn(data_dir, spk, mel_ids)
            self.train_info += [(m, spk) for m in mel_ids]
        for spk in self.test_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_x_dir, spk))
            mel_ids = [m[:-8] for m in mel_ids]
            self.test_info += [(m, spk) for m in mel_ids]
        print("Total number of test wavs is %d." % len(self.test_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, mel_id, spk):
        mel_x_path = os.path.join(self.mel_x_dir, spk, mel_id + '_mel.npy')
        mel_y_path = os.path.join(self.mel_y_dir, spk, mel_id + '_avgmel.npy')
        mel_x = np.load(mel_x_path)
        mel_y = np.load(mel_y_path)
        mel_x = torch.from_numpy(mel_x).float()
        mel_y = torch.from_numpy(mel_y).float()
        return (mel_x, mel_y)

    def __getitem__(self, index):
        mel_id, spk = self.train_info[index]
        mel_x, mel_y = self.get_vc_data(mel_id, spk)
        item = {'x': mel_x, 'y': mel_y}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_test_dataset(self):
        pairs = []
        for i in range(len(self.test_info)):
            mel_id, spk = self.test_info[i]
            mel_x, mel_y = self.get_vc_data(mel_id, spk)
            pairs.append((mel_x, mel_y))
        return pairs


# VCTK dataset for training "average voice" encoder
class VCTKEncDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, exc_file, avg_type):
        self.mel_x_dir = os.path.join(data_dir, 'mels')
        self.mel_y_dir = os.path.join(data_dir, 'mels_%s' % avg_type)

        self.unseen_speakers = get_vctk_unseen_speakers()
        self.unseen_sentences = get_vctk_unseen_sentences()
        self.speakers = [spk for spk in os.listdir(self.mel_x_dir) 
                         if spk not in self.unseen_speakers]
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_mel.npy' for e in exceptions]
        self.test_info = []
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_x_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m[:-8] for m in mel_ids if m not in self.exceptions]
            mel_ids = exclude_spn(data_dir, spk, mel_ids)
            self.train_info += [(m, spk) for m in mel_ids]
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_x_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m[:-8] for m in mel_ids if m not in self.exceptions]
            self.test_info += [(m, spk) for m in mel_ids]
        print("Total number of test wavs is %d." % len(self.test_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, mel_id, spk):
        mel_x_path = os.path.join(self.mel_x_dir, spk, mel_id + '_mel.npy')
        mel_y_path = os.path.join(self.mel_y_dir, spk, mel_id + '_avgmel.npy')
        mel_x = np.load(mel_x_path)
        mel_y = np.load(mel_y_path)
        mel_x = torch.from_numpy(mel_x).float()
        mel_y = torch.from_numpy(mel_y).float()
        return (mel_x, mel_y)

    def __getitem__(self, index):
        mel_id, spk = self.train_info[index]
        mel_x, mel_y = self.get_vc_data(mel_id, spk)
        item = {'x': mel_x, 'y': mel_y}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_test_dataset(self):
        pairs = []
        for i in range(len(self.test_info)):
            mel_id, spk = self.test_info[i]
            mel_x, mel_y = self.get_vc_data(mel_id, spk)
            pairs.append((mel_x, mel_y))
        return pairs

class VCTKUnitEncDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, exc_file, cfg):
        self.hunit_x_dir = os.path.join(data_dir, f'huberts_{cfg}')
        self.unit_y_dir = os.path.join(data_dir, 'units')

        self.unseen_speakers = get_vctk_unseen_speakers()
        self.unseen_sentences = get_vctk_unseen_sentences()
        self.speakers = [spk for spk in os.listdir(self.hunit_x_dir) 
                         if spk not in self.unseen_speakers]
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_hunit.npy' for e in exceptions]
        self.test_info = []
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.hunit_x_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m[:-10] for m in mel_ids if m not in self.exceptions]
            # mel_ids = exclude_spn(data_dir, spk, mel_ids)
            self.train_info += [(m, spk) for m in mel_ids]
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.hunit_x_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m[:-10] for m in mel_ids if m not in self.exceptions]
            self.test_info += [(m, spk) for m in mel_ids]
        print("Total number of test wavs is %d." % len(self.test_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, mel_id, spk):
        mel_x_path = os.path.join(self.hunit_x_dir, spk, mel_id + '_hunit.npy')
        mel_y_path = os.path.join(self.unit_y_dir, spk, mel_id + '_unit.npy')
        mel_x = np.load(mel_x_path)
        mel_y = np.load(mel_y_path)
        mel_x = torch.from_numpy(mel_x).long()
        mel_y = torch.from_numpy(mel_y).float()
        return (mel_x, mel_y)

    def __getitem__(self, index):
        mel_id, spk = self.train_info[index]
        mel_x, mel_y = self.get_vc_data(mel_id, spk)
        item = {'x': mel_x, 'y': mel_y}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_test_dataset(self):
        pairs = []
        # for i in range(len(self.test_info)):
        for i in range(10):
            mel_id, spk = self.test_info[i]
            mel_x, mel_y = self.get_vc_data(mel_id, spk)
            pairs.append((mel_x, mel_y))
        return pairs

###
class VCTKUnitEncDataset_v1(torch.utils.data.Dataset):
    def __init__(self, data_dir, exc_file, cfg):
        self.hunit_x_dir = os.path.join(data_dir, f'huberts_{cfg}')
        self.unit_y_dir = os.path.join(data_dir, 'units')
        self.f0_dir = os.path.join(data_dir, 'f0s')
        ### for embspeech
        self.unseen_speakers = get_vctk_unseen_speakers()[:1]
        self.unseen_sentences = get_vctk_unseen_sentences()[:1]
        ###
        self.speakers = [spk for spk in os.listdir(self.hunit_x_dir) 
                         if spk not in self.unseen_speakers]
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_hunit.npy' for e in exceptions]
        self.test_info = []
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.hunit_x_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m[:-10] for m in mel_ids if m not in self.exceptions]
            # mel_ids = exclude_spn(data_dir, spk, mel_ids)
            self.train_info += [(m, spk) for m in mel_ids]
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.hunit_x_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m[:-10] for m in mel_ids if m not in self.exceptions]
            self.test_info += [(m, spk) for m in mel_ids]
        print("Total number of test wavs is %d." % len(self.test_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, mel_id, spk):
        mel_x_path = os.path.join(self.hunit_x_dir, spk, mel_id + '_hunit.npy')
        mel_y_path = os.path.join(self.unit_y_dir, spk, mel_id + '_unit.npy')
        f0_path = os.path.join(self.f0_dir, spk, mel_id + '_f0.npy')
        mel_x = np.load(mel_x_path)
        mel_y = np.load(mel_y_path)
        f0 = np.load(f0_path)
        f0 = self.normalize_(f0)
        mel_x = torch.from_numpy(mel_x).long()
        mel_y = torch.from_numpy(mel_y).float()
        f0 = torch.from_numpy(f0).float()
        return (mel_x, mel_y, f0)

    def normalize_(self, arr):
        min_value = np.min(arr)
        max_value = np.max(arr)
        normalized_array = (arr - min_value) / (max_value - min_value)
        return normalized_array

    def __getitem__(self, index):
        mel_id, spk = self.train_info[index]
        mel_x, mel_y, f0 = self.get_vc_data(mel_id, spk)
        item = {'x': mel_x, 'y': mel_y, 'f0':f0}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_test_dataset(self):
        pairs = []
        # for i in range(len(self.test_info)):
        for i in range(10):
            mel_id, spk = self.test_info[i]
            mel_x, mel_y, f0 = self.get_vc_data(mel_id, spk)
            pairs.append((mel_x, mel_y, f0))
        return pairs
###


class DurDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, exc_file, cfg='l9k500'):
        self.hunit_x_dir = os.path.join(data_dir, f'huberts_{cfg}')
        self.emo_dir = os.path.join(data_dir,f'emo2vec')

        self.unseen_speakers = get_vctk_unseen_speakers()
        self.unseen_sentences = get_vctk_unseen_sentences()
        self.speakers = [spk for spk in os.listdir(self.hunit_x_dir) 
                         if spk not in self.unseen_speakers]
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_hunit.npy' for e in exceptions]
        self.test_info = []
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.hunit_x_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m[:-10] for m in mel_ids if m not in self.exceptions]
            # mel_ids = exclude_spn(data_dir, spk, mel_ids)
            self.train_info += [(m, spk) for m in mel_ids]
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.hunit_x_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m[:-10] for m in mel_ids if m not in self.exceptions]
            self.test_info += [(m, spk) for m in mel_ids]
        print("Total number of test wavs is %d." % len(self.test_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, mel_id, spk):
        mel_x_path = os.path.join(self.hunit_x_dir, spk, mel_id + '_hunit.npy')
        emov_path = os.path.join(self.emo_dir, mel_id + '.npy')
        mel_x = np.load(mel_x_path)
        mel_x = torch.from_numpy(mel_x).long()
        uniseq, dur = torch.unique_consecutive(mel_x, return_counts=True)
        uniseq = uniseq.long()
        dur = dur.float()
        emov = np.load(emov_path)
        emov = torch.from_numpy(emov)
        return (uniseq, dur, emov)

    def __getitem__(self, index):
        mel_id, spk = self.train_info[index]
        uniseq, dur, emov = self.get_vc_data(mel_id, spk)
        item = {'x': uniseq, 'dur': dur, 'emov': emov}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_test_dataset(self):
        pairs = []
        # for i in range(len(self.test_info)):
        for i in range(10):
            mel_id, spk = self.test_info[i]
            uniseq, dur, emov = self.get_vc_data(mel_id, spk)
            pairs.append((uniseq, dur, emov))
        return pairs

class DurDataset_ESD(torch.utils.data.Dataset):
    def __init__(self, data_dir, exc_file, cfg='l6k100'):
        self.hunit_x_dir = os.path.join(data_dir, f'huberts_{cfg}')
        self.emo_dir = os.path.join(data_dir,f'emos')

        self.unseen_speakers = get_esd_unseen_speakers()
        self.unseen_sentences = get_esd_unseen_sentences()
        self.speakers = [spk for spk in os.listdir(self.hunit_x_dir) 
                         if spk not in self.unseen_speakers]
        self.test_info = []
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.hunit_x_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m[:-10] for m in mel_ids]
            self.train_info += [(m, spk) for m in mel_ids]
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.hunit_x_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m[:-10] for m in mel_ids]
            self.test_info += [(m, spk) for m in mel_ids]
        print("Total number of test wavs is %d." % len(self.test_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, mel_id, spk):
        mel_x_path = os.path.join(self.hunit_x_dir, spk, mel_id + '_hunit.npy')
        # emov_path = os.path.join(self.emo_dir, mel_id + '.npy')
        emov_path = os.path.join(self.emo_dir, spk, mel_id + '_emo.npy')
        mel_x = np.load(mel_x_path)
        mel_x = torch.from_numpy(mel_x).long()
        uniseq, dur = torch.unique_consecutive(mel_x, return_counts=True)
        uniseq = uniseq.long()
        dur = dur.float()
        emov = np.load(emov_path)
        emov = torch.from_numpy(emov)
        return (uniseq, dur, emov)

    def __getitem__(self, index):
        mel_id, spk = self.train_info[index]
        uniseq, dur, emov = self.get_vc_data(mel_id, spk)
        item = {'x': uniseq, 'dur': dur, 'emov': emov}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_test_dataset(self):
        pairs = []
        # for i in range(len(self.test_info)):
        for i in range(10):
            mel_id, spk = self.test_info[i]
            uniseq, dur, emov = self.get_vc_data(mel_id, spk)
            pairs.append((uniseq, dur, emov))
        return pairs
    
######
class VCEncBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        mels_x = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        mels_y = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        max_starts = [max(item['x'].shape[-1] - train_frames, 0) 
                      for item in batch]
        starts = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel_x = item['x']
            mel_y = item['y']
            if mel_x.shape[-1] < train_frames:
                mel_length = mel_x.shape[-1]
            else:
                mel_length = train_frames
            mels_x[i, :, :mel_length] = mel_x[:, starts[i]:starts[i] + mel_length]
            mels_y[i, :, :mel_length] = mel_y[:, starts[i]:starts[i] + mel_length]
            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        return {'x': mels_x, 'y': mels_y, 'lengths': mel_lengths}

class UnitBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        mels_x = torch.zeros((B, train_frames), dtype=torch.long)
        mels_y = torch.zeros((B, 768, train_frames), dtype=torch.float32)
        max_starts = [max(item['x'].shape[-1] - train_frames, 0) 
                      for item in batch]
        starts = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel_x = item['x']
            mel_y = item['y']
            if mel_x.shape[-1] < train_frames:
                mel_length = mel_x.shape[-1]
            else:
                mel_length = train_frames
            mels_x[i, :mel_length] = mel_x[ starts[i]:starts[i] + mel_length]
            mels_y[i, :, :mel_length] = mel_y[:, starts[i]:starts[i] + mel_length]
            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        return {'x': mels_x, 'y': mels_y, 'lengths': mel_lengths}
    
###
class UnitBatchCollate_v1(object):
    def __call__(self, batch):
        B = len(batch)
        mels_x = torch.zeros((B, train_frames), dtype=torch.long)
        mels_y = torch.zeros((B, 768, train_frames), dtype=torch.float32)
        f0s = torch.zeros((B, 1, train_frames), dtype=torch.float32)
        max_starts = [max(item['x'].shape[-1] - train_frames, 0) 
                      for item in batch]
        starts = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel_x = item['x']
            mel_y = item['y']
            f0 = item['f0']
            if mel_x.shape[-1] < train_frames:
                mel_length = mel_x.shape[-1]
            else:
                mel_length = train_frames
            mels_x[i, :mel_length] = mel_x[ starts[i]:starts[i] + mel_length]
            mels_y[i, :, :mel_length] = mel_y[:, starts[i]:starts[i] + mel_length]
            f0s[i, :, :mel_length] = f0[ starts[i]:starts[i] + mel_length]
            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        return {'x': mels_x, 'y': mels_y, 'lengths': mel_lengths, 'f0s': f0s}
###

class DurBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)

        uniseqs = []
        durs = []
        emovs = []
        for i, item in enumerate(batch):
            uniseq = item['x']
            dur = item['dur']
            emov = item['emov']
            uniseqs.append(uniseq)
            durs.append(dur)
            emovs.append(emov)
        # mel_lengths = torch.LongTensor(mel_lengths)
        x=torch.nn.utils.rnn.pad_sequence(uniseqs, batch_first=True, padding_value=100)
        y=torch.nn.utils.rnn.pad_sequence(durs, batch_first=True, padding_value=-1)
        e=torch.nn.utils.rnn.pad_sequence(emovs, batch_first=True, padding_value=torch.nan)
        assert not torch.isnan(e.max())
        return {'uniseq': x, 'dur': y, 'emovs': e}

# LibriTTS dataset for training speaker-conditional diffusion-based decoder
class VCDecDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, val_file, exc_file):
        self.mel_dir = os.path.join(data_dir, 'mels')
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.test_speakers = get_test_speakers()
        self.speakers = [spk for spk in os.listdir(self.mel_dir)
                         if spk not in self.test_speakers]
        self.speakers = [spk for spk in self.speakers
                         if len(os.listdir(os.path.join(self.mel_dir, spk))) >= 10]
        random.seed(random_seed)
        random.shuffle(self.speakers)
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_mel.npy' for e in exceptions]
        with open(val_file) as f:
            valid_ids = f.readlines()
        self.valid_ids = set([v.strip() + '_mel.npy' for v in valid_ids])
        self.exceptions += self.valid_ids

        self.valid_info = [(v[:-8], v.split('_')[0]) for v in self.valid_ids]
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m not in self.exceptions]
            self.train_info += [(i[:-8], spk) for i in mel_ids]
        print("Total number of validation wavs is %d." % len(self.valid_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        print("Total number of training speakers is %d." % len(self.speakers))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, audio_info):
        audio_id, spk = audio_info
        mels = self.get_mels(audio_id, spk)
        embed = self.get_embed(audio_id, spk)
        return (mels, embed)

    def get_mels(self, audio_id, spk):
        mel_path = os.path.join(self.mel_dir, spk, audio_id + '_mel.npy')
        mels = np.load(mel_path)
        mels = torch.from_numpy(mels).float()
        return mels

    def get_embed(self, audio_id, spk):
        embed_path = os.path.join(self.emb_dir, spk, audio_id + '_embed.npy')
        embed = np.load(embed_path)
        embed = torch.from_numpy(embed).float()
        return embed

    def __getitem__(self, index):
        mels, embed = self.get_vc_data(self.train_info[index])
        item = {'mel': mels, 'c': embed}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_valid_dataset(self):
        pairs = []
        for i in range(len(self.valid_info)):
            mels, embed = self.get_vc_data(self.valid_info[i])
            pairs.append((mels, embed))
        return pairs


# VCTK dataset for training speaker-conditional diffusion-based decoder
class VCTKDecDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.mel_dir = os.path.join(data_dir, 'mels')
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.unseen_speakers = get_vctk_unseen_speakers()
        self.unseen_sentences = get_vctk_unseen_sentences()
        self.speakers = [spk for spk in os.listdir(self.mel_dir)
                         if spk not in self.unseen_speakers]
        random.seed(random_seed)
        random.shuffle(self.speakers)
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            self.train_info += [(i[:-8], spk) for i in mel_ids]
        self.valid_info = []
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            self.valid_info += [(i[:-8], spk) for i in mel_ids]
        print("Total number of validation wavs is %d." % len(self.valid_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        print("Total number of training speakers is %d." % len(self.speakers))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, audio_info):
        audio_id, spk = audio_info
        mels = self.get_mels(audio_id, spk)
        embed = self.get_embed(audio_id, spk)
        return (mels, embed)

    def get_mels(self, audio_id, spk):
        mel_path = os.path.join(self.mel_dir, spk, audio_id + '_mel.npy')
        mels = np.load(mel_path)
        mels = torch.from_numpy(mels).float()
        return mels

    def get_embed(self, audio_id, spk):
        embed_path = os.path.join(self.emb_dir, spk, audio_id + '_embed.npy')
        embed = np.load(embed_path)
        embed = torch.from_numpy(embed).float()
        return embed

    def __getitem__(self, index):
        mels, embed = self.get_vc_data(self.train_info[index])
        item = {'mel': mels, 'c': embed}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_valid_dataset(self):
        pairs = []
        for i in range(len(self.valid_info)):
            mels, embed = self.get_vc_data(self.valid_info[i])
            pairs.append((mels, embed))
        return pairs
    
    def get_unseen_dataset(self):

        unseen_info = []
        spk_dict = {}
        utt_dict = {}
        for unseen_spk in self.unseen_speakers:
            spk_dict[unseen_spk] = []
        for unseen_spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, unseen_spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] in self.unseen_sentences]
            unseen_info += [(i[:-8], unseen_spk) for i in mel_ids]
            spk_dict[unseen_spk] += [i[:-8] for i in mel_ids]
        print("Total number of unseen wavs is %d." % len(unseen_info))
        print("Total number of unseen speakers is %d." % len(self.unseen_speakers))

        for i in range(len(unseen_info)):
            mels, embed = self.get_vc_data(unseen_info[i])
            utt_dict[unseen_info[i][0]]=(mels,embed)

        return spk_dict, utt_dict


class VCTKDecDataset_v1(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.mel_dir = os.path.join(data_dir, 'mels')
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.unit_dir = os.path.join(data_dir, 'units')
        ### for embspeech
        # self.unseen_speakers = get_vctk_unseen_speakers()
        # self.unseen_sentences = get_vctk_unseen_sentences()
        self.unseen_speakers = get_vctk_unseen_speakers()[:1]
        self.unseen_sentences = get_vctk_unseen_sentences()[:1]
        ###
        self.speakers = [spk for spk in os.listdir(self.mel_dir)
                         if spk not in self.unseen_speakers]
        random.seed(random_seed)
        random.shuffle(self.speakers)
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            self.train_info += [(i[:-8], spk) for i in mel_ids]
        self.valid_info = []
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            self.valid_info += [(i[:-8], spk) for i in mel_ids]
        print("Total number of validation wavs is %d." % len(self.valid_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        print("Total number of training speakers is %d." % len(self.speakers))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, audio_info):
        audio_id, spk = audio_info
        mels = self.get_mel(audio_id, spk)
        embed = self.get_embed(audio_id, spk)
        unit = self.get_unit(audio_id, spk, mels.shape[-1])
        return (mels, embed, unit)

    def get_mel(self, audio_id, spk):
        mel_path = os.path.join(self.mel_dir, spk, audio_id + '_mel.npy')
        mel = np.load(mel_path)
        mel = torch.from_numpy(mel).float()
        return mel

    def get_embed(self, audio_id, spk):
        embed_path = os.path.join(self.emb_dir, spk, audio_id + '_embed.npy')
        embed = np.load(embed_path)
        embed = torch.from_numpy(embed).float()
        return embed
    
    def get_unit(self, audio_id, spk, mel_len):
        unit_path = os.path.join(self.unit_dir, spk, audio_id + '_unit.npy')
        unit = np.load(unit_path)
        unit = torch.from_numpy(unit).float()
        unit = repeat_expand_2d(unit, target_len=mel_len, mode='nearest')
        return unit        


    def __getitem__(self, index):
        mels, embed, unit = self.get_vc_data(self.train_info[index])
        item = {'mel': mels, 'c': embed, 'unit': unit}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_valid_dataset(self):
        tripairs = []
        for i in range(len(self.valid_info)):
            mel, embed, unit= self.get_vc_data(self.valid_info[i])
            tripairs.append((mel, embed, unit))
        return tripairs
    
    def get_unseen_dataset(self):
        unseen_info = []
        spk_dict = {}
        utt_dict = {}
        for unseen_spk in self.unseen_speakers:
            spk_dict[unseen_spk] = []
        for unseen_spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, unseen_spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] in self.unseen_sentences]
            unseen_info += [(i[:-8], unseen_spk) for i in mel_ids]
            spk_dict[unseen_spk] += [i[:-8] for i in mel_ids]
        print("Total number of unseen wavs is %d." % len(unseen_info))
        print("Total number of unseen speakers is %d." % len(self.unseen_speakers))

        for i in range(len(unseen_info)):
            mels, embed, unit = self.get_vc_data(unseen_info[i])
            utt_dict[unseen_info[i][0]] = (mels, embed, unit)

        return spk_dict, utt_dict



class EmotionalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.mel_dir = os.path.join(data_dir, 'mels')
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.unit_dir = os.path.join(data_dir, 'units')
        self.emo_dir = os.path.join(data_dir, 'emos')
        self.emoemb_dir = os.path.join(data_dir, 'emo2vec')
        self.unseen_speakers = get_esd_unseen_speakers()
        self.unseen_sentences = get_esd_unseen_sentences()
        self.speakers = [spk for spk in os.listdir(self.mel_dir)
                         if spk not in self.unseen_speakers]
        random.seed(random_seed)
        random.shuffle(self.speakers)
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            self.train_info += [(i[:-8], spk) for i in mel_ids] # _mel.npy
        self.valid_info = []
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            self.valid_info += [(i[:-8], spk) for i in mel_ids]
        print("Total number of validation wavs is %d." % len(self.valid_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        print("Total number of training speakers is %d." % len(self.speakers))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, audio_info):
        audio_id, spk = audio_info
        mels = self.get_mel(audio_id, spk)
        embed = self.get_embed(audio_id, spk)
        unit = self.get_unit(audio_id, spk, mels.shape[-1])
        emo = self.get_emo(audio_id, spk)
        return (mels, embed, unit, emo)

    def get_mel(self, audio_id, spk):
        mel_path = os.path.join(self.mel_dir, spk, audio_id + '_mel.npy')
        mel = np.load(mel_path)
        mel = torch.from_numpy(mel).float()
        return mel

    def get_embed(self, audio_id, spk):
        embed_path = os.path.join(self.emb_dir, spk, audio_id + '_embed.npy')
        embed = np.load(embed_path)
        embed = torch.from_numpy(embed).float()
        return embed
    
    def get_unit(self, audio_id, spk, mel_len):
        unit_path = os.path.join(self.unit_dir, spk, audio_id + '_unit.npy')
        unit = np.load(unit_path)
        unit = torch.from_numpy(unit).float()
        unit = repeat_expand_2d(unit, target_len=mel_len, mode='nearest')
        return unit        

    def get_emo0(self, audio_id, spk):
        emo_path = os.path.join(self.emo_dir, spk, audio_id + '_emo.npy')
        # label = np.load(emo_path)
        # emo = one_hot(label)
        emo = np.load(emo_path)
        emo = torch.from_numpy(emo).float()
        return emo

    def get_emo(self, audio_id, spk):
        emo_path = os.path.join(self.emoemb_dir, audio_id + '.npy')
        emo = np.load(emo_path)
        emo = torch.from_numpy(emo).float()
        return emo

    def __getitem__(self, index):
        mels, embed, unit, emo = self.get_vc_data(self.train_info[index])
        item = {'mel': mels, 'c': embed, 'unit': unit, 'emo':emo}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_valid_dataset(self):
        quapairs = []
        # for i in range(len(self.valid_info)):
        for i in range(10):    
            mel, embed, unit, emo= self.get_vc_data(self.valid_info[i])
            quapairs.append((mel, embed, unit, emo))
        return quapairs

class VCTKDecDataset_v2(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.mel_dir = os.path.join(data_dir, 'latnts')
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.unit_dir = os.path.join(data_dir, 'units')
        self.unseen_speakers = get_vctk_unseen_speakers()
        self.unseen_sentences = get_vctk_unseen_sentences()
        self.speakers = [spk for spk in os.listdir(self.mel_dir)
                         if spk not in self.unseen_speakers]
        random.seed(random_seed)
        random.shuffle(self.speakers)
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            # self.train_info += [(i[:-8], spk) for i in mel_ids]
            self.train_info += list(set([(i[:-8], spk) for i in mel_ids]))
        self.valid_info = []
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            self.valid_info += [(i[:-8], spk) for i in mel_ids]
        print("Total number of validation wavs is %d." % len(self.valid_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        print("Total number of training speakers is %d." % len(self.speakers))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, audio_info):
        audio_id, spk = audio_info
        mels = self.get_mel(audio_id, spk)
        embed = self.get_embed(audio_id, spk)
        unit = self.get_unit(audio_id, spk, mels.shape[-1])
        return (mels, embed, unit)

    def get_mel(self, audio_id, spk):
        mel_path = os.path.join(self.mel_dir, spk, audio_id + '_emb.npy')
        mel = np.load(mel_path)
        mel = torch.from_numpy(mel).float().transpose(0,1)
        return mel

    def get_embed(self, audio_id, spk):
        embed_path = os.path.join(self.emb_dir, spk, audio_id + '_embed.npy')
        embed = np.load(embed_path)
        embed = torch.from_numpy(embed).float()
        return embed
    
    def get_unit(self, audio_id, spk, mel_len):
        unit_path = os.path.join(self.unit_dir, spk, audio_id + '_unit.npy')
        unit = np.load(unit_path)
        unit = torch.from_numpy(unit).float()
        unit = repeat_expand_2d(unit, target_len=mel_len, mode='nearest')
        return unit        


    def __getitem__(self, index):
        mels, embed, unit = self.get_vc_data(self.train_info[index])
        item = {'mel': mels, 'c': embed, 'unit': unit}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_valid_dataset(self):
        tripairs = []
        for i in range(len(self.valid_info)):
            mel, embed, unit= self.get_vc_data(self.valid_info[i])
            tripairs.append((mel, embed, unit))
        return tripairs

class VCDecBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        mels1 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        mels2 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        max_starts = [max(item['mel'].shape[-1] - train_frames, 0)
                      for item in batch]
        starts1 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        starts2 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel = item['mel']
            if mel.shape[-1] < train_frames:
                mel_length = mel.shape[-1]
            else:
                mel_length = train_frames
            mels1[i, :, :mel_length] = mel[:, starts1[i]:starts1[i] + mel_length]
            mels2[i, :, :mel_length] = mel[:, starts2[i]:starts2[i] + mel_length]
            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        embed = torch.stack([item['c'] for item in batch], 0)
        return {'mel1': mels1, 'mel2': mels2, 'mel_lengths': mel_lengths, 'c': embed}
    
class VCDecBatchCollate_v1(object):
    def __call__(self, batch):
        B = len(batch)
        mels1 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        mels1_unit = torch.zeros((B, 768, train_frames), dtype=torch.float32)
        mels2 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        max_starts = [max(item['mel'].shape[-1] - train_frames, 0)
                      for item in batch]
        starts1 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        starts2 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel = item['mel']
            unit = item['unit']
            assert mel.shape[-1] == unit.shape[-1], "unit has not been expanded to match mel"
            if mel.shape[-1] < train_frames:
                mel_length = mel.shape[-1]
            else:
                mel_length = train_frames
            mels1[i, :, :mel_length] = mel[:, starts1[i]:starts1[i] + mel_length]
            mels1_unit[i, :, :mel_length] = unit[:, starts1[i]:starts1[i] + mel_length]

            mels2[i, :, :mel_length] = mel[:, starts2[i]:starts2[i] + mel_length]
            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        embed = torch.stack([item['c'] for item in batch], 0)
        return {'mel1': mels1, 'mel2': mels2, 'mel_lengths': mel_lengths, 'c': embed, 'mel1_unit': mels1_unit}

class VCDecBatchCollate_v2(object):
    def __call__(self, batch):
        B = len(batch)
        mels1 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        mels1_unit = torch.zeros((B, 768, train_frames), dtype=torch.float32) 
        # mels2 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        max_starts = [max(item['mel'].shape[-1] - train_frames, 0)
                      for item in batch]
        starts1 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        # starts2 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel = item['mel']
            unit = item['unit']
            assert mel.shape[-1] == unit.shape[-1], "unit has not been expanded to match mel"
            if mel.shape[-1] < train_frames:
                mel_length = mel.shape[-1]
            else:
                mel_length = train_frames
            mels1[i, :, :mel_length] = mel[:, starts1[i]:starts1[i] + mel_length]
            mels1_unit[i, :, :mel_length] = unit[:, starts1[i]:starts1[i] + mel_length]

            # mels2[i, :, :mel_length] = mel[:, starts2[i]:starts2[i] + mel_length]
            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        embed = torch.stack([item['c'] for item in batch], 0)
        return {'mel1': mels1, 'mel_lengths': mel_lengths, 'c': embed, 'mel1_unit': mels1_unit}


class VCDecBatchCollate_v3(object):
    def __call__(self, batch):
        B = len(batch)
        mels1 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        mels1_unit = torch.zeros((B, 768, train_frames), dtype=torch.float32) 
        max_starts = [max(item['mel'].shape[-1] - train_frames, 0)
                      for item in batch]
        starts1 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel = item['mel']
            unit = item['unit']
            assert mel.shape[-1] == unit.shape[-1], "unit has not been expanded to match mel"
            if mel.shape[-1] < train_frames:
                mel_length = mel.shape[-1]
            else:
                mel_length = train_frames
            mels1[i, :, :mel_length] = mel[:, starts1[i]:starts1[i] + mel_length]
            mels1_unit[i, :, :mel_length] = unit[:, starts1[i]:starts1[i] + mel_length]

            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        embed = torch.stack([item['c'] for item in batch], 0)
        emo = torch.stack([item['emo'] for item in batch], 0)
        return {'mel1': mels1, 'mel_lengths': mel_lengths, 'c': embed, 'mel1_unit': mels1_unit,
                'emo': emo}


###
###
def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text

class TextEmb2Dataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path):
        self.filepaths_and_text = parse_filelist(filelist_path)
        random.seed(random_seed)
        random.shuffle(self.filepaths_and_text)

    def get_pair(self, filepath_and_text):
        filepath = filepath_and_text[0]
        temb = self.get_temb(filepath)
        emb = self.get_emb(filepath, temb.shape[-1])
        return (temb, emb)

    def get_temb(self, filepath):
        temb_path = filepath.replace("/wavs/","/tembs/").replace(".wav","_temb.npy")
        temb = np.load(temb_path)
        temb = torch.from_numpy(temb).float()
        return temb

    def get_emb(self, filepath, tgtlen):
        emb_path = filepath.replace("/wavs/","/units/").replace(".wav","_unit.npy")
        emb = np.load(emb_path)
        emb = torch.from_numpy(emb).float()
        # emb = repeat_expand_2d(emb, target_len=tgtlen, mode='nearest')
        emb = repeat_expand_2d(emb, target_len=tgtlen, mode='linear')
        return emb

    def __getitem__(self, index):
        temb, emb = self.get_pair(self.filepaths_and_text[index])
        item = {'x': temb, 'y': emb}
        return item

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        np.random.seed(random_seed)
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch

class TextEmb2BatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        mels_x = torch.zeros((B, 80, train_frames), dtype=torch.float32)
        mels_y = torch.zeros((B, 768, train_frames), dtype=torch.float32)
        max_starts = [max(item['x'].shape[-1] - train_frames, 0) 
                      for item in batch]
        starts = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel_x = item['x']
            mel_y = item['y']
            if mel_x.shape[-1] < train_frames:
                mel_length = mel_x.shape[-1]
            else:
                mel_length = train_frames
            mels_x[i, :, :mel_length] = mel_x[:, starts[i]:starts[i] + mel_length]
            mels_y[i, :, :mel_length] = mel_y[:, starts[i]:starts[i] + mel_length]
            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        return {'x': mels_x, 'y': mels_y, 'lengths': mel_lengths}
    


