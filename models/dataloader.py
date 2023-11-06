import torch
import torch.nn as nn
from torch.nn import functional as F

class dataloader:
    def __init__(self, file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            text = f.read()        
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        stoi = { ch:i for i,ch in enumerate(self.chars) }
        itos = { i:ch for i,ch in enumerate(self.chars) }
        # functions to convert string to list of integer token and vice-versa
        self.encode = lambda s: [stoi[c] for c in s] 
        self.decode = lambda l: ''.join([itos[i] for i in l])
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9*len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
    
    def get_vocab_size(self):
        return self.vocab_size

    def get_sample_batch(self,split = 'train', batch_size = 4, block_size = 8 , device = 'cpu'):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y







