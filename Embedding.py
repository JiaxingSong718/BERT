import torch
from torch import nn
from config import *
from dataset import vocab_size

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding:(maxlen, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        #self.norm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x, seg):
        #x.shape = (batch_size, maxlen)
        #seg.shape = (batch_size, maxlen)
        seq_len = x.size(1)#seq_len = maxlen
        pos = torch.arange(seq_len, dtype=torch.long)#(0, 1, 2, ..., maxlen-1)
        #pos.shape = (maxlen, )---> (1, maxlen)--->(batch_size, maxlen)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]

        #(batch_size, maxlen, d_model)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)