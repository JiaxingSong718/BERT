import torch
from torch import nn, optim
from BERT import BERT
from dataset import *
from torch.utils.data import DataLoader

model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)
dataset = MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext)
loader = DataLoader(dataset, batch_size, True)

for epoch in range(50):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
        #logits_lm.shape = (batch_size, max_pred, vocab_size)
        #logits_clsf.shape = (batch_size, 2)
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        #loss_lm.type = torch.Tensor


        #logits_lm.view(-1, vocab_size)  .shape = (batch_size*max_pred, vocab_size)
        #masked_tokens.shape = (batch_size, max_pred)--->(batch_size*max_pred,)
        loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
        # loss_lm = (loss_lm.float()).mean()

        #logits_clsf.shape = (batch_size, 2)
        #isNext.shape = (batch_size, )
        loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
        loss = loss_lm + loss_clsf
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Predict mask tokens ans isNext
input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[1]
print(text)
print('================================')
print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])

logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), \
                 torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list : ',[pos for pos in masked_tokens if pos != 0])
print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext : ',True if logits_clsf else False)