#!/usr/bin/env python
# coding: utf-8

# #### **BERTのfine-tuningによる予測**

# In[1]:


import random
import numpy as np
from tqdm import tqdm_notebook as tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from sklearn.metrics import accuracy_score, f1_score
from transformers import *
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import pickle


# #### **BERTの入力に合わせたENCODE処理**

# In[2]:


# BERTの入力仕様に合わせたtext変換
def bert_encode(text, max_len=512,output=False):
    
    if output:
        print(">> raw text:")
        print(text)
        print()
    
    text = tokenizer.tokenize(text)
    text = text[:max_len-2] # CLSとSEPいれて最大max_lenになるように削る
    
    if output:
        print(">> after tokenize")
        print(text)
        print()
    
    
    input_sequence = ["[CLS]"] + text + ["[SEP]"]
    if output:
        print(">> joint CLS and SEP")
        print(input_sequence)
        print()
    
    # 単語のまま渡さず, 辞書に登録されている単語のIDに変換してからbertに渡す
    tokens = tokenizer.convert_tokens_to_ids(input_sequence)
    if output:
        print(">> after tokenization")
        print(tokens)
        print()
    
    segment_id = [0] * (len(text)+2) + [0] * (max_len - len(tokens))
    
    # 長さをmax_lenにそろえる. 足りない場合は後ろに0をくっつける
    tokens += [0] * (max_len - len(input_sequence))
    if output:
        print(">> after 0-padding")
        print(tokens)
        print()
    
    # ちょっと何やってるかわかんない
    pad_masks = [1] * len(input_sequence) + [0] * (max_len - len(input_sequence))
    
    if output:
        print(">> mask?")
        print(pad_masks)
        print()
        
    
    return tokens, pad_masks, segment_id


# #### **Dataset, DataLoaderの作成**

# In[3]:


# NNのデータ読み込みを楽にしてくれるDataLoaderの定義
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, train_tokens,train_pad_masks, train_segment_ids, targets):
        
        super(Dataset, self).__init__() # <= これ必要か?
        self.train_tokens = train_tokens
        self.train_pad_masks = train_pad_masks
        self.segment_ids = train_segment_ids
        self.targets = targets
    
    def __getitem__(self, index):
        tokens = self.train_tokens[index]
        masks = self.train_pad_masks[index]
        segment_ids = self.segment_ids[index]
        target = self.targets[index]
        
        # 出力は2要素, input, output
        return (tokens, masks, segment_ids), target
    
    def __len__(self,):
        return len(self.train_tokens)

def make_train(texts,targets):
    train_text = texts

    # すべてのtextをtokenizeしていく
    train_tokens = [] # 1st input
    train_pad_masks = [] # 2nd input
    train_segment_ids = [] # 3rd input

    for text in (train_text):
        tokens, masks, segment_id = bert_encode(text)

        train_tokens.append(tokens)
        train_pad_masks.append(masks)
        train_segment_ids.append(segment_id)

    train_tokens = np.array(train_tokens)
    train_tokens = torch.tensor(train_tokens,dtype=torch.long)

    train_pad_masks = np.array(train_pad_masks)
    train_pad_masks = torch.tensor(train_pad_masks,dtype=torch.long)

    train_segment_ids = np.array(train_segment_ids)
    train_segment_ids = torch.tensor(train_segment_ids,dtype=torch.long)

    targets = train["target"].values
    targets = torch.tensor(targets,dtype=torch.long)

    train_dataset = Dataset(train_tokens=train_tokens,
                            train_pad_masks=train_pad_masks,
                            train_segment_ids=train_segment_ids,
                            targets=targets)
    
    return train_dataset


# #### **BERTモデル定義**

# In[4]:


class BERT_clf(nn.Module):
    
    def __init__(self, hidden_size=768, num_class=2):
        super(BERT_clf, self).__init__()
        
        # BERTのベース層
        self.bert = BertModel.from_pretrained('bert-base-uncased',  
                                        output_hidden_states=True,
                                        output_attentions=True)
        
        # BERTのベースパラメータをTuning可能にする(fine-tuning)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # 出力層のために作ったのか?
        self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropouts = nn.ModuleList([ nn.Dropout(0.5) for _ in range(5) ])
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, input_ids, input_mask, segment_ids):
        all_hidden_states, all_attentions = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[-2:]
        
        batch_size = input_ids.shape[0]
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(13, batch_size, 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
        
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.fc(dropout(feature))
            else:
                h += self.fc(dropout(feature))
        
        h = h / len(self.dropouts)
        
        return h


# #### **データ読み込み**

# In[5]:


#train = pd.read_csv('rawdata/train.csv')
#test = pd.read_csv('rawdata/test.csv')
#submit = pd.read_csv('rawdata/sample_submission.csv')

train = pd.read_csv('data/cleaned_train.csv')
test = pd.read_csv('data/cleaned_test.csv')
submit = pd.read_csv('rawdata/sample_submission.csv')

print('Train size:', train.shape)
print('Test size:', test.shape)


# In[9]:


train_text = train["text"].values
targets = train["target"].values


# 学習データセット作成
train_dataset = make_train(train_text[:6000],targets[:6000])
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=4, 
                                               shuffle=True)

# Validationデータセット作成
valid_dataset = make_train(train_text[6000:],targets[6000:])
valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, 
                                               batch_size=1, 
                                               shuffle=True)


# In[10]:


# モデルの定義
model = BERT_clf()
model = model.cuda()

# ロス関数の定義
loss_fn = torch.nn.CrossEntropyLoss()

# 勾配法の定義
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-6)


# In[8]:


# fine-tuning開始


result = pd.DataFrame(columns=["train acc","test acc"])
prev_acc = -np.inf
for epoch in (range(50)):
    
    print("epoch:",epoch)
    model.train()
    acc = 0
    total = 0
    for batch in tqdm(train_dataloader): 

        # 入力データ取り出し, GPUのメモリに乗せる
        inputs = batch[0]
        inputs = tuple(t.cuda() for t in inputs)

        # 正解ラベル取り出し, GPUのメモリに乗せる
        label = batch[1]
        label = label.cuda()

        # 推定結果算出
        output = model(inputs[0], # tokenizeしたtext
                       inputs[1], # マスク
                       inputs[2]) # segment id

        # ロス計算 & Gradient Decent
        loss = loss_fn(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 一番確率の高いラベルが予測
        pred_label = torch.argmax(output,axis=1)
        
        # 精度記録
        total += len(label)
        acc += pred_label.eq(label).cpu().sum().item()
    
    result.loc[epoch,"train loss"] = acc/total
    
    model.eval()
    acc = 0
    total = 0
    for batch in tqdm(valid_dataloader): 

        # 入力データ取り出し, GPUのメモリに乗せる
        inputs = batch[0]
        inputs = tuple(t.cuda() for t in inputs)

        # 正解ラベル取り出し, GPUのメモリに乗せる
        label = batch[1]
        label = label.cuda()

        # 推定結果算出
        output = model(inputs[0], # tokenizeしたtext
                       inputs[1], # マスク
                       inputs[2]) # segment id

        # 一番確率の高いラベルが予測
        pred_label = torch.argmax(output,axis=1)
        
        # 精度記録
        total += len(label)
        acc += pred_label.eq(label).cpu().sum().item()
    
    result.loc[epoch,"test loss"] = acc/total
    result.to_csv("BERT_loss.csv")
    
    if prev_acc < acc/total:
        with open("BERT.mdl","wb") as f:
            pickle.dump(model,f)
        
        prev_acc = acc/total




