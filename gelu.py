import torch
import math

def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def get_attn_pad_mask(seq_q, seq_k):
    #(batch_size, maxlen)
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    #(batch_size, 1, maxlen)
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]

    #pad_attn_mask.shape = (batch_size, 1, maxlen)
    #(batch_size, seq_len, seq_len):seq_len = maxlen
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]