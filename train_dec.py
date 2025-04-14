# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import params
from data import VCDecBatchCollate_v1, VCTKDecDataset_v1
# from model.vc import DiffVC_light as DiffVC
from model.vc import DiffVC
from model.utils import FastGL
from utils import save_plot, save_audio
# from model.encodec import EncodecWrapper


n_mels = params.n_mels
# n_mels = 128
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
# use_ref_t = params.use_ref_t
use_ref_t = False
beta_min = params.beta_min
beta_max = params.beta_max

random_seed = params.seed
test_size = params.test_size

# data_dir = '../data/LibriTTS'
# val_file = 'filelists/valid.txt'
# exc_file = 'filelists/exceptions_libritts.txt'

data_dir = './dataset/vctk'
val_file = 'filelists/valid.txt'
exc_file = 'filelists/exceptions_vctk.txt'

log_dir = 'logs_dec24'
# enc_dir = 'logs_enc'
epochs = 250
batch_size = 16
learning_rate = 1e-4
save_every = 1

# encodec=EncodecWrapper().to("cuda")

def clean_ckpts(dir,num_keep):
    ckpts = [i for i in os.listdir(dir) if i.startswith("vc_")]
    optims = [i for i in os.listdir(dir) if i.startswith("optim_")]

    def time_key(_f):
        return os.path.getmtime(os.path.join(dir, _f))
    
    todel1 = sorted(ckpts, key = time_key)[:-num_keep]
    todel2 = sorted(optims, key = time_key)[:-num_keep]
    for ckpt in todel1:
        os.remove(os.path.join(dir,ckpt))
        print(f"Free up space by deleting ckpt: {ckpt}")
    for optim in todel2:
        os.remove(os.path.join(dir,optim))
        print(f"Free up space by deleting optim: {optim}")


if __name__ == "__main__":
    print('use_ref_t=',use_ref_t)
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    os.makedirs(log_dir, exist_ok=True)

    print('Initializing data loaders...')
    # train_set = VCDecDataset(data_dir, val_file, exc_file)
    train_set = VCTKDecDataset_v1(data_dir)
    collate_fn = VCDecBatchCollate_v1()
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                              collate_fn=collate_fn, num_workers=4, drop_last=True)

    print('Initializing and loading models...')
    fgl = FastGL(n_mels, sampling_rate, n_fft, hop_size).cuda()
    model = DiffVC(n_mels, channels, filters, heads, layers, kernel, 
                   dropout, window_size, enc_dim, spk_dim, use_ref_t, 
                   dec_dim, beta_min, beta_max).cuda()
    # model.load_encoder(os.path.join(enc_dir, 'enc.pt'))

    # print('Encoder:')
    # print(model.encoder)
    # print('Number of parameters = %.2fm\n' % (model.encoder.nparams/1e6))
    print('Decoder:')
    # print(model.decoder)
    print('Number of parameters = %.2fm\n' % (model.decoder.nparams/1e6))

    print('Initializing optimizers...')
    optimizer = torch.optim.Adam(params=model.decoder.parameters(), lr=learning_rate)

    print('Start training.')
    torch.backends.cudnn.benchmark = True
    ###
    ckpt_ep = None
    # ckpt_ep = 10
    # print(f'Load ckpts of epoch {ckpt_ep}.')
    # model.load_state_dict(torch.load(f'logs_dec/vc_{ckpt_ep}.pt'))
    # optimizer.load_state_dict(torch.load(f'logs_dec/optim_{ckpt_ep}.pt'))
    ###
    iteration = 0
    for epoch in range(ckpt_ep+1 if ckpt_ep is not None else 1, epochs + 1):
        print(f'Epoch: {epoch} [iteration: {iteration}]')
        model.train()
        losses = []
        for batch in tqdm(train_loader, total=len(train_set)//batch_size):
            mel, mel_ref = batch['mel1'].cuda(), batch['mel2'].cuda()
            c, mel_lengths = batch['c'].cuda(), batch['mel_lengths'].cuda()
            unit = batch['mel1_unit'].cuda()
            model.zero_grad()
            # print("mel, mel_len, mel_ref, c shape: ",mel.shape,mel_lengths.shape,mel_ref.shape,c.shape)
            # torch.Size([16, 80, 128]) torch.Size([16]) torch.Size([16, 80, 128]) torch.Size([16, 256])
            loss = model.compute_loss(mel, mel_lengths, mel_ref, c, unit)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1)
            optimizer.step()
            losses.append(loss.item())
            iteration += 1
            # break

        losses = np.asarray(losses)
        msg = 'Epoch %d: loss = %.4f\n' % (epoch, np.mean(losses))
        print(msg)
        with open(f'{log_dir}/train_dec.log', 'a') as f:
            f.write(msg)
        losses = []

        if epoch % save_every > 0:
            continue

        model.eval()
        print('Inference...\n')
        with torch.no_grad():
            mels = train_set.get_valid_dataset()
            for i, (mel, c, unit) in enumerate(mels):
                if i >= test_size:
                    break
                mel = mel.unsqueeze(0).float().cuda()
                c = c.unsqueeze(0).float().cuda()
                ###
                unit = unit.unsqueeze(0).float().cuda()
                ###
                mel_lengths = torch.LongTensor([mel.shape[-1]]).cuda()
                mel_avg, mel_rec = model(mel, mel_lengths, mel, mel_lengths, c, unit,
                                         n_timesteps=100)
                if epoch == save_every:
                    save_plot(mel.squeeze().cpu(), f'{log_dir}/original_{i}.png')
                    audio = fgl(mel)
                    save_audio(f'{log_dir}/original_{i}.wav', sampling_rate, audio)
                    # audio = encodec.decode(mel.transpose(1,2)).squeeze().squeeze().cpu()
                    # save_audio(f'{log_dir}/original_{i}.wav', 24000, audio)
                save_plot(mel_avg.squeeze().cpu(), f'{log_dir}/average_{i}.png')
                audio = fgl(mel_avg)
                save_audio(f'{log_dir}/average_{i}.wav', sampling_rate, audio)
                save_plot(mel_rec.squeeze().cpu(), f'{log_dir}/reconstructed_{i}.png')
                audio = fgl(mel_rec)
                save_audio(f'{log_dir}/reconstructed_{i}.wav', sampling_rate, audio)
                # audio = encodec.decode(mel_rec.transpose(1,2)).squeeze().squeeze().cpu()
                # save_audio(f'{log_dir}/reconstructed_{i}.wav', 24000, audio)
                # torch.save(mel_rec,f=f'{log_dir}/reconstructed.pt')
        print('Saving model...\n')
        ckpt = model.state_dict()
        optim = optimizer.state_dict()
        torch.save(ckpt, f=f"{log_dir}/vc_{epoch}.pt")
        torch.save(optim, f=f"{log_dir}/optim_{epoch}.pt")
        clean_ckpts(log_dir, num_keep=5)
