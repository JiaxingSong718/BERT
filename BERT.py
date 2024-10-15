import torch
from torch import nn
from dataset import vocab_size
from Embedding import Embedding
from Encoder import EncoderLayer
from gelu import gelu,get_attn_pad_mask
from config import *


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])


        ######################################
        #(batch_size, seq_len, d_model)
        #(batch_size, d_model)
        self.fc = nn.Sequential(
            #(batch_size, d_model)
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
            #(batch_size, d_model)
        )
        self.classifier = nn.Linear(d_model, 2)
        #(batch_size, 2)
        ######################################


        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight#weight.shape = (vocab_size, d_model)
        self.fc2 = nn.Linear(d_model, vocab_size)#weight.shape = (vocab_size, d_model)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        #word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']
        #[CLS] , sentence1, [SEP], sentence2, [SEP]

        #input_ids.shape = (batch_size, 1 + len1 + 1 + len2 + 1 )-->(batch_size, maxlen)
        #segment_ids.shape = (bat_size, 1 + len1 + 1 + len2 + 1)--->(batch_size, maxlen)
        #masked_pos.shape = (batch_size, n_pred)-->(batch_size, max_pred)

        output = self.embedding(input_ids, segment_ids) # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids) # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)
        # it will be decided by first token(CLS)
        #output.shape = (batch_size, seq_len, d_model)
        h_pooled = self.fc(output[:, 0]) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2] predict isNext

        # self.linear = nn.Linear(d_model, d_model)
        # self.activ2 = gelu
        # # fc2 is shared with embedding layer
        # embed_weight = self.embedding.tok_embed.weight#weight.shape = (vocab_size, d_model)
        # self.fc2 = nn.Linear(d_model, vocab_size, bias=False)#weight.shape = (vocab_size, d_model)
        # self.fc2.weight = embed_weight

        #masked_pos.shape = (batch_size, max_pred)-->(batch_size, max_pred, 1)-->(batch_size, max_pred, d_model)
        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model) # [batch_size, max_pred, d_model]

        #output.shape = (batch_size, seq_len, d_model)
        h_masked = torch.gather(output, 1, masked_pos)
        #h_masked.shape = (batch_size, max_pred, d_model)
        h_masked = self.activ2(self.linear(h_masked)) 
        #h_masked.shape = [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked) # [batch_size, max_pred, vocab_size]
        return logits_lm, logits_clsf
