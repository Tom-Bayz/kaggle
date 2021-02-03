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
    
    def __init__(self, test_tokens,test_pad_masks, test_segment_ids):
        
        super(Dataset, self).__init__() # <= これ必要か?
        self.test_tokens = test_tokens
        self.test_pad_masks = test_pad_masks
        self.segment_ids = test_segment_ids
    
    def __getitem__(self, index):
        tokens = self.test_tokens[index]
        masks = self.test_pad_masks[index]
        segment_ids = self.segment_ids[index]
        
        # 出力は2要素, input, output
        return (tokens, masks, segment_ids)
    
    def __len__(self,):
        return len(self.test_tokens)

def make_test(texts):
    test_text = texts

    # すべてのtextをtokenizeしていく
    test_tokens = [] # 1st input
    test_pad_masks = [] # 2nd input
    test_segment_ids = [] # 3rd input

    for text in (test_text):
        tokens, masks, segment_id = bert_encode(text)

        test_tokens.append(tokens)
        test_pad_masks.append(masks)
        test_segment_ids.append(segment_id)

    test_tokens = np.array(test_tokens)
    test_tokens = torch.tensor(test_tokens,dtype=torch.long)

    test_pad_masks = np.array(test_pad_masks)
    test_pad_masks = torch.tensor(test_pad_masks,dtype=torch.long)

    test_segment_ids = np.array(test_segment_ids)
    test_segment_ids = torch.tensor(test_segment_ids,dtype=torch.long)

    test_dataset = Dataset(test_tokens=test_tokens,
                            test_pad_masks=test_pad_masks,
                            test_segment_ids=test_segment_ids)
    
    return test_dataset



train = pd.read_csv('data/cleaned_train.csv',index_col = 0,encoding="utf-8")
test = pd.read_csv('data/cleaned_test.csv',index_col = 0,encoding="utf-8")
submit = pd.read_csv('rawdata/sample_submission.csv')

print('Train size:', train.shape)
print('Test size:', test.shape)


# In[9]:

test = pd.concat([train,test],sort=False)

test_text = test["text"].values
targets = test["target"].values


# テストデータセット作成
test_dataset = make_test(test_text)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=4,
                                              shuffle=True)
# In[10]:

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



# モデルの定義
with open("BERT.mdl","rb") as f:
    model = pickle.load(f)
    
model = model.cuda()

# In[8]:


# fine-tuning開始
prediction = []
for epoch in (range(1)):
    
    print("epoch:",epoch)
    model.eval()
    acc = 0
    total = 0
    for batch in tqdm(test_dataloader): 

        # 入力データ取り出し, GPUのメモリに乗せる
        inputs = batch
        inputs = tuple(t.cuda() for t in inputs)

        
        # 推定結果算出
        output = model(inputs[0], # tokenizeしたtext
                       inputs[1], # マスク
                       inputs[2]) # segment id

        # 一番確率の高いラベルが予測
        pred_label = torch.argmax(output,axis=1).cpu()
        prediction.append(list(pred_label.numpy()))

prediction = np.array(prediction)
prediction = np.reshape(prediction,(1,-1))[0,:]
prediction = pd.Series(prediction, index=test.index)



test["target"] = prediction
        
print(test)
    
# load submission file
submission = pd.read_csv(os.path.join("submission","sample_submission.csv"))

# my submission file
test = test.sort_index()
test["id"] = test.index

mysub = test.iloc[submission["id"]]
mysub["id"] = mysub["id"].astype("int64")

print(mysub)
mysub[["id","target"]].to_csv(os.path.join("submission","BERT.csv"),index=False)
