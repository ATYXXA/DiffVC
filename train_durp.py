import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import params
from data import DurDataset, DurBatchCollate, DurDataset_ESD
from model.dur_predictor import LenPredictor
from utils import save_plot



# data_dir = './dataset/vctk'
data_dir = './dataset/ESD'
exc_file = './filelists/exceptions_vctk.txt'


log_dir = 'logs_durp'
epochs = 100
batch_size = 32
learning_rate = 3e-4
save_every = 1

import torch.nn.functional as F

class LenSumLoss(torch.nn.Module):
    def __init__(self, pad_idx=-1):
        super(LenSumLoss, self).__init__()
        self.pad_idx = pad_idx
        self.mse = torch.nn.MSELoss(reduction='none')

    def forward(self, preds, lens):
        # This is used to encourage nearby errors to cancel out, as to not cause a length bias
        diff4 = (F.avg_pool2d((preds - lens).unsqueeze(0), (1, 4)) * 4) ** 2
        diff_mask4 = ~F.max_pool2d((lens == self.pad_idx).unsqueeze(0).float(), (1, 4)).bool()
        diff_loss4 = (diff_mask4 * diff4).sum()

        mask = (lens != self.pad_idx)
        total_loss = self.mse(preds, lens)
        return (mask * total_loss).sum()  + 0.5 * diff_loss4


if __name__ == "__main__":

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    os.makedirs(log_dir, exist_ok=True)

    print('Initializing data loaders...')
    train_set = DurDataset_ESD(data_dir, exc_file)
    collate_fn = DurBatchCollate()
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                              collate_fn=collate_fn, num_workers=4,
                              drop_last=True)
    print('Initializing models...')
    model = LenPredictor(n_tokens=100).cuda()
    # print(model)
    # print('Number of parameters = %.2fm\n' % (model.nparams/1e6))  
     
    # print('Load ckpt...')
    # model.load_state_dict(torch.load(f'{log_dir}/durp.pt')) 
    
    print('Initializing optimizers...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    iteration = 0

    loss_s = LenSumLoss()
    # loss_s = torch.nn.MSELoss()
    # loss_s = torch.nn.L1Loss()
    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch} [iteration: {iteration}]')
        model.train()
        losses = []
        for batch in tqdm(train_loader, total=len(train_set)//batch_size):
            uniseq_x, dur_y = batch['uniseq'].cuda(), batch['dur'].cuda()
            emovs = batch['emovs'].cuda()
            # print(uniseq_x.shape,dur_y.shape,emovs.shape) # torch.Size([32, 194]) torch.Size([32, 194]) torch.Size([32, 768])
            # break
            model.zero_grad()
            preds = model(uniseq_x, emovs)
            loss = loss_s(preds, dur_y)            
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

            # break
        # break
        if epoch % save_every > 0:
            continue

        model.eval()
        print('Inference...\n')
        with torch.no_grad():
            mels = train_set.get_test_dataset()
            for i, (uniseq, dur, emov) in enumerate(mels):
                if i >= params.test_size:
                    break
                uniseq = uniseq.unsqueeze(0).long().cuda()
                dur = dur.unsqueeze(0).float().cuda()
                emov = emov.unsqueeze(0).float().cuda()
                pred = model(uniseq, emov)
                save_plot(dur.cpu(), f'{log_dir}/label_{i}.png')
                save_plot(pred.cpu(), f'{log_dir}/pred_{i}.png')
                print(dur.cpu()[0][:10])
                print(pred.cpu()[0][:10])


        print('Saving model...\n')
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/durp.pt")