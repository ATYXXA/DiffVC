import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import params
from data import VCTKUnitEncDataset, UnitBatchCollate
from model.vc import U2E
from model.utils import sequence_mask
from utils import save_plot

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

random_seed = params.seed
test_size = params.test_size

data_dir = './dataset/vctk'
exc_file = './filelists/exceptions_vctk.txt'


log_dir = 'logs_enc_l9'
epochs = 100
batch_size = 32
learning_rate = 5e-4
save_every = 1


if __name__ == "__main__":

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    os.makedirs(log_dir, exist_ok=True)

    print('Initializing data loaders...')
    hubertcfg = 'l9k500'
    train_set = VCTKUnitEncDataset(data_dir, exc_file, hubertcfg)
    collate_fn = UnitBatchCollate()
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                              collate_fn=collate_fn, num_workers=4,
                              drop_last=True)

    print('Initializing models...')
    n_vocab = int(hubertcfg.split('k')[-1])
    model = U2E(n_vocab, 768, channels, filters, heads, layers, kernel, 
                         dropout, window_size).cuda()

    print('Encoder:')
    # print(model)
    print('Number of parameters = %.2fm\n' % (model.nparams/1e6))

    print('Initializing optimizers...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print('Start training.')
    torch.backends.cudnn.benchmark = True
    iteration = 0
    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch} [iteration: {iteration}]')
        model.train()
        losses = []
        for batch in tqdm(train_loader, total=len(train_set)//batch_size):
            mel_x, mel_y = batch['x'].cuda(), batch['y'].cuda()
            # print(mel_x.shape,mel_y.shape) #torch.Size([32, 128]) torch.Size([32, 768, 128])
            mel_lengths = batch['lengths'].cuda()
            mel_mask = sequence_mask(mel_lengths).unsqueeze(1).to(mel_x.dtype)
            # print(mel_lengths.shape,mel_mask.shape) #torch.Size([32]) torch.Size([32, 1, 128])

            model.zero_grad()
            loss = model.compute_loss(mel_x, mel_y, mel_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            losses.append(loss.item())
            iteration += 1

        losses = np.asarray(losses)
        msg = 'Epoch %d: loss = %.4f\n' % (epoch, np.mean(losses))
        print(msg)
        with open(f'{log_dir}/train_enc.log', 'a') as f:
            f.write(msg)
        losses = []
 
        if epoch % save_every > 0:
            continue

        model.eval()
        print('Inference...\n')
        with torch.no_grad():
            mels = train_set.get_test_dataset()
            for i, (mel_x, mel_y) in enumerate(mels):
                if i >= test_size:
                    break
                mel_x = mel_x.unsqueeze(0).long().cuda()
                mel_y = mel_y.unsqueeze(0).float().cuda()
                mel_lengths = torch.LongTensor([mel_x.shape[-1]]).cuda()
                mel_mask = sequence_mask(mel_lengths).unsqueeze(1).to(mel_x.dtype)
                mel = model(mel_x, mel_mask)
                save_plot(mel.squeeze().cpu(), f'{log_dir}/generated_{i}.png')

                if epoch == save_every:
                    save_plot(mel_x.cpu(), f'{log_dir}/source_{i}.png')
                    save_plot(mel_y.squeeze().cpu(), f'{log_dir}/target_{i}.png')


        print('Saving model...\n')
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/enc.pt")
