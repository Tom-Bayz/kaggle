#!/usr/bin/env python
# coding: utf-8

# ## **XGBoost Regression**
# https://rdrr.io/cran/xgboost/man/xgb.cv.html

# In[ ]:

import pandas as pd
import numpy as np
import os
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000

import gc
from myUtils import *
from feature_generator import feature_v1, feature_v2, feature_v3
import xgboost as xgb
import pickle
from lightgbm import LGBMRegressor

# In[20]:


###################################################################
# make features
##################################################################
feature_maker = feature_v3.FeaturesMaker_v3(target_col="item_cnt")
#feature_maker = feature_v1.FeaturesMaker_v1(target_col="item_cnt")

base_data = "sales_train_eval"
data = pd.read_pickle(os.path.join("mydata",base_data+".pickle"))
#data = pd.read_pickle(os.path.join("mydata","sales_train_eval.pickle"))

data_path = os.path.join("mydata",base_data+"_"+feature_maker.name+".pickle")
if os.path.exists(data_path):
    with open(data_path,"rb") as f:
        data = pickle.load(f)
    print("loaded data")
else:
    data = feature_maker.make_feature(data)
    with open(data_path,"wb") as f:
        pickle.dump(data,f)


# In[22]:


print("evaluation")
print(data["evaluation"][0].isna().sum())
print()

print("validation")
print(data["validation"][0].isna().sum())
print()

print("train")
print(data["train"][0].isna().sum())
print()


# In[24]:
model_path = os.path.join("models","LightGBM_"+feature_maker.name+".mdl")

if os.path.exists(model_path):
    print("loading trained model...")
    with open(model_path,"rb") as f:
        model = pickle.load(f)

else:
    print("start training XGBoost")
    model = LGBMRegressor(n_estimators=1000,
                            learning_rate=0.3,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            max_depth=8,
                            num_leaves=50,
                            min_child_weight=300)

    model.fit(data["train"][0],
              data["train"][1],
              eval_set=[data["train"],data["validation"]],
              eval_metric='rmse',
              verbose=20,
              early_stopping_rounds=20
             )

    with open(model_path,"wb") as f:
        pickle.dump(model,f)

print("  -- completed\n")


# #### [prediction]

# In[6]:


# prediction
print("start prediction")
pred_mask = data["evaluation"][1].isna()
data["evaluation"][1].loc[pred_mask] = model.predict(data["evaluation"][0])
print("  -- completed\n")


# In[17]:


data["evaluation"][0]


# In[16]:


tmp = data["evaluation"][1].reset_index()
mask = tmp["id"] == "HOBBIES_1_001_CA_1_evaluation"

tmp[mask]


# #### [submission]

# In[71]:


# submission
print("start submission")
sub_path = os.path.join("submission_point","XGBoost_"+feature_maker.name+"_submission.csv")


sub_cols = ["id"] + [f"F{i}" for i in range(1, 29)]

valid = data["validation"][1]
evalu = data["evaluation"][1]

del data
gc.collect()

valid = pd.DataFrame(valid.values,
                     index=valid.index,
                     columns=[feature_maker.target_col])
evalu = pd.DataFrame(evalu.values,
                     index=evalu.index,
                     columns=[feature_maker.target_col])

valid = valid.reset_index()
evalu = evalu.reset_index()

valid = pd.pivot(valid,
                 index="id",
                 columns="d",
                 values=feature_maker.target_col)
evalu = pd.pivot(evalu,
                 index="id",
                 columns="d",
                 values=feature_maker.target_col)

valid = valid.reset_index()
evalu = evalu.reset_index()

valid.columns = sub_cols
evalu.columns = sub_cols

valid["id"] = valid["id"].str.replace("_evaluation","_validation")

pd.concat([valid,evalu]).to_csv(sub_path,index=False)
print("  -- completed")
