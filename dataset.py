import re
import torch
from torch.utils.data import Dataset
from random import *
from config import *


text = (
    'Hello, how are you? I am Romeo.\n' # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n' # J
    'Nice meet you too. How are you today?\n' # R
    'Great. My baseball team won the competition.\n' # J
    'Oh Congratulations, Juliet\n' # R
    'Thank you Romeo\n' # J
    'Where are you going today?\n' # R
    'I am going shopping. What about you?\n' # J
    'I am going to visit my grandmother. she is not very well' # R
)
# text = (
#     'abc+'
#     '!#$$'
# )
# print(text);exit()

sentences = re.sub("[.,!?\\-]", '',text.lower()).split('\n') # filter '.', ',', '?', '!'
# print(sentences);exit()

sentences_join = " ".join(sentences)
word_list = list(set(sentences_join.split()))
# print(word_list);exit()

word2idx = {'[PAD]' :0, '[CLS]' :1, '[SEP]' :2, '[MASK]' :3}
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
idx2word = {index:word for word, index in word2idx.items()}
vocab_size = len(word2idx)

token_list = []
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    # print(sentence)
    # print(sentence.split())
    # print(arr)
    # exit()
    token_list.append(arr)

# sample IsNext and NotNext to be same in small batch size
def make_data():
    #batch_size = 6, max_pred = 5
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        #random.randrange(start, end, step=1):从[start, end)区间中以step为步长随机选取一个数for i in range(start, end, step)
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences)) # sample random index in sentences
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        #n_pred:被mask的单词数量
        n_pred =  min(max_pred, max(1, int(len(input_ids) * 0.15))) # 15 % of tokens in one sentence
        #记录可以被mask的单词的下标
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']
                        ] # candidate masked position
        
        #random.shuffle()
        shuffle(cand_maked_pos)#打乱顺序，方面后续随机选取数据mask
        masked_tokens, masked_pos = [], []
        #选前n_pred个单词进行mask
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)#记录input_ids的下标（被替换的）
            masked_tokens.append(input_ids[pos])#记录被替换之前的单词是哪一个
            if random() < 0.8:  # 80%
                #[0, 0.8)使用[MASK]
                input_ids[pos] = word2idx['[MASK]'] # make mask

                #[0.8, 0.9)不替换
            elif random() > 0.9:  # 10%
                #[0.9, 1.0]随机替换：但是不可以使用'CLS', 'SEP', 'PAD'
                #random.randint(a, b)生成[a, b]的整数
                index = randint(0, vocab_size - 1) # random index in vocabulary
                while index < 4: # can't involve 'CLS', 'SEP', 'PAD'
                  index = randint(0, vocab_size - 1)
                input_ids[pos] = index # replace


        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)#PAD
            masked_pos.extend([0] * n_pad)#CLS
        #pred = (batch_size, max_pred, d_model)-->(batch_size, max_pred, vocab_size)-->(batch_size*max_pred, vocab_size)
        #masked_tokens.shape = (batch_size, max_pred)--->(batch_size*max_pred, )
        #nn.CrossEntropyLoss(pred, masked_tokens)


        #input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        #segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
        #(batch_size, maxlen, d_model)--->(batch_size, max_pred, d_model):torch.gather
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            negative += 1
    return batch

#batch=[
# [input_ids, segment_ids, masked_tokens, masked_pos, Bool],
# [...]
# ]
batch = make_data()
# print(batch[0]);exit()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)#解数据操作
input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
    torch.LongTensor(input_ids),  torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens),\
    torch.LongTensor(masked_pos), torch.LongTensor(isNext)

class MyDataSet(Dataset):
  def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
    self.input_ids = input_ids
    self.segment_ids = segment_ids
    self.masked_tokens = masked_tokens
    self.masked_pos = masked_pos
    self.isNext = isNext
  
  def __len__(self):
    return len(self.input_ids)
  
  def __getitem__(self, idx):
    return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[idx]

