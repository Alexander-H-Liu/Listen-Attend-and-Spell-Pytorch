import os
import csv
import yaml
import argparse
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

"""
DataLoader
"""

class PhonemeDataset(Dataset):
    def __init__(self, csv_file):
        self.max_idx = 0
        self.pho = self.read_csv(csv_file)

    def __len__(self):
        return len(self.pho)

    def __getitem__(self, idx):
        return self.pho[idx]
    
    def read_csv(self, csv_file):
        def str2int(l):
            tem = [int(x) for x in l]
            if self.max_idx < max(tem):
                self.max_idx = max(tem)
            return tem
        with open(csv_file) as f:
            rows = list(csv.reader(f))
        pho = list(zip(*rows))[3][1:]
        return [str2int(p.split('_')) for p in pho]

# def padding(l, max_length):
    # pad_len = max_length - len(l)
    # return l + [0 for _ in range(pad_len)]

def padding(l, fill_value=0):
    return list(itertools.zip_longest(*l, fillvalue=fill_value))

def collate_fn(batch):
    batch.sort(key=lambda d: len(d), reverse=True)
    lens = [len(d) for d in batch]
    # batch = [padding(d, lens[0]) for d in batch]
    batch = padding(batch)
    batch = torch.LongTensor(batch)
    target = batch[1:, :]
    return batch, lens, target

def get_loader(csv_file, batch_size, num_workers):
    dataset = PhonemeDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True,
                      shuffle=True, num_workers=num_workers), dataset.max_idx+1

"""
Language Model
"""
class RNN_LM(nn.Module):
    def __init__(self, emb_dim, h_dim, out_dim, layers=1, rnn='LSTM', dropout_rate=0.0):
        super().__init__()
        self.h_dim = h_dim
        self.emb = nn.Embedding(out_dim, emb_dim)
        self.rnn = getattr(nn, rnn.upper())(emb_dim, h_dim, num_layers=layers, dropout=dropout_rate)
        self.out = nn.Linear(h_dim, out_dim)

    def forward(self, x, lens, hidden=None):
        embedded = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lens)
        outputs, hidden = self.rnn(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = self.out(outputs)
        return hidden, outputs

"""
Training
"""
def train(paras, config):
    model_para = config['model_para']
    writer = SummaryWriter(os.path.join(paras.log_dir), 'RNN-LM')
    if not os.path.isdir(config['save_dir']):
        os.makedirs(config['save_dir'])
    dl, out_dim = get_loader(paras.csv_file, paras.batch_size, paras.num_workers)
    lm = RNN_LM(model_para['emb_dim'], model_para['h_dim'], out_dim, model_para['layers'], 
                model_para['rnn'], model_para['dropout_rate'])
    lm.to(device)
    opt = torch.optim.Adam(lm.parameters(), lr=config['learning_rate'])
    best_loss = 9e10
    for epoch in range(paras.n_epochs):
        print('===============  Epoch: {}  ==============='.format(epoch))
        print_loss = 0 
        for input_seq, lens, target in tqdm(dl):
            input_seq = input_seq.to(device)
            target = target.to(device)
            _, output = lm(input_seq, lens)
            loss = F.cross_entropy(output[:-1].permute(0, 2, 1), target, ignore_index=0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print_loss += loss
        print_loss = print_loss/len(dl.dataset)/paras.batch_size
        if print_loss < best_loss:
            best_loss = print_loss
            torch.save(lm.state_dict(), os.path.join(config['save_dir'], 'lm.pt'))
        print('training loss: {:.5f}'.format(print_loss.item()))
        print('perplexity:    {:.5f}'.format(torch.exp(print_loss).item()))
        writer.add_scalar('loss', print_loss, epoch)
        writer.add_scalar('perplexity', torch.exp(print_loss).item(), epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Path to experiment config.', required=True)
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='log')

    paras = parser.parse_args()
    config = yaml.load(open(paras.config,'r'))['asr_model']['rnn_lm']

    train(paras, config)
