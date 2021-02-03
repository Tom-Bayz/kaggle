#!/usr/bin/env python
# coding: utf-8

# ### **Data cleansing**
# - remove or replace undesireble word

# In[33]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import numba
import os


# **1. loading data**

# In[34]:


train = pd.read_csv('rawdata/train.csv')
test = pd.read_csv('rawdata/test.csv')



import gensim
import os
model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join("D:","dataset","word_vector","GoogleNews-vectors-negative300.bin"), binary=True)


# In[114]:
from tqdm import tqdm
def doc2vec(data):
    text = data[["text"]]

    features = ["fig"+str(i) for i in range(300)] #300 dim
    
    header = pd.DataFrame(columns=["id","word_id","word"])
    word2vec = pd.DataFrame(columns=features)
    
    for doc_id in tqdm(data["id"]):
        doc = text.loc[data["id"] == doc_id].values[0][0]
        for word_id, word in tqdm(enumerate(doc.split(" "))):
            try:
                vec = model[word]
            except:
                vec = np.array([np.nan]*300)
            
            vec = pd.DataFrame(columns = features,
                               data = [vec])
            #print(word)
            h = pd.DataFrame(columns=["id","word_id","word"])
            h["id"] = [doc_id]
            h["word_id"] = [word_id]
            h["word"] = [word]
            
            #print(h)
            header = pd.concat([header,h],sort=False)
            word2vec = pd.concat([word2vec,vec],sort=False)
            
            
    
    return pd.concat([header,word2vec],axis=1,sort=False)

from multiprocessing import Pool    
def multi_process_doc2vec(data):
    
    with Pool(processes=4) as p:
        out = p.map(func=doc2vec, 
                    iterable=np.array_split(data,len(data)))
    
    new_train = pd.DataFrame()
    for d in tqdm(out):
        new_train = pd.concat([new_train,d])
    
    return new_train

if __name__ == "__main__":
    
    new_train = multi_process_doc2vec(train)
    new_train.to_csv(os.path.join("data","w2v_train.csv"))

    new_test = multi_process_doc2vec(test)
    new_test.to_csv(os.path.join("data","w2v_test.csv"))