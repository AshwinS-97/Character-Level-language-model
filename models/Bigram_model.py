import torch
import torch.nn as nn
from torch.nn import functional as F

class bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        torch.manual_seed(1337)
        self.vocab = vocab_size
        self.token_embedding_table = nn.Embedding(num_embeddings=self.vocab, embedding_dim=self.vocab)
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
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
            logits, _ = self(idx)
            logits = logits[:,-1]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx

