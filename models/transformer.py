import torch
import torch.nn as nn
from torch.nn import functional as F

n_embd  = 32
dropout = 0.2
device = 'cuda:0'
class Head(nn.Module):
    def __init__(self, head_size, block_size):
        super().__init__()
        self.head_size  = head_size
        self.block_size = block_size
        self.key        = nn.Linear(n_embd,self.head_size, bias=False)
        self.query      = nn.Linear(n_embd,self.head_size, bias=False)
        self.value      = nn.Linear(n_embd,self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size ,self.block_size)))
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2,-1) * self.head_size**(-0.5)   # to keep variance near 1
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # this line makes it a decoder block without it would be encoder
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size,block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)
        self.droupout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return self.droupout(out)

class FeedforwardLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_head,block_size):
        super().__init__()
        headsize  = n_embd//n_head
        self.sa   = MultiHeadAttention(n_head, headsize, block_size)
        self.ffwd = FeedforwardLayer()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class transformer(nn.Module):
    def __init__(self, vocab_size, block_size):
        super().__init__()
        self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(4, block_size),
            Block(4, block_size),
            Block(4, block_size),
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)  

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, Embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits   = self.lm_head(x)          # (B, T, Vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  
            targets = targets.view(B*T)  
            loss   = F.cross_entropy(logits, targets)  
        return logits, loss

    def generate(self, idx, max_new_token):
        for _ in range(max_new_token):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx


