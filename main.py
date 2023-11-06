import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from models import dataloader
from models import bigram
from models import transformer
print(torch.cuda.is_available())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# torch.set_default_device('cuda:0')
data = dataloader('input.txt')
vocab_size = data.get_vocab_size()

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data.get_sample_batch('train', batch_size, block_size)
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(model, name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    PATH = 'checkpoints/' +name + '.pt'
    if (os.path.exists(PATH)):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
    else:
        for iter in range(max_iters):
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(model)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            xb, yb = data.get_sample_batch('train', batch_size, block_size)
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step() 

        torch.save({
                    'epoch': max_iters,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, PATH)

    print("Generated after training model - " + name, data.decode(
        model.generate(idx = torch.zeros((1,1), dtype=torch.long, device=device), max_new_token=2000)[0].tolist()))
    
# Training the model -- Bigram model
max_iters     = 40000
eval_interval = 300
eval_iters    = 50
block_size    = 256 
batch_size    = 64 
# model = bigram(vocab_size)
# xb, yb = data.get_sample_batch()
# logits, loss = model(xb, yb)
# print("Initial without training loss [ref: ", 
#       -torch.log(torch.tensor([1/vocab_size])).item(),"] =", loss)
# print("Generated before training model", data.decode(
#     model.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_token=100)[0].tolist()))
# train(model, 'bigram')


# training the transformer model
model = transformer(vocab_size, block_size).to(device)
train(model, 'transformer')



