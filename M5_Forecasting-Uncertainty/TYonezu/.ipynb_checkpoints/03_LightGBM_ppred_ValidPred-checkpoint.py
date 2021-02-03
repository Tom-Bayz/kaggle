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
#feature_maker = feature_v3.FeaturesMaker_v3(target_col="item_cnt")
feature_maker = feature_v1.FeaturesMaker_v1(target_col="item_cnt")

base_data = "sales_train_eval_28"
data = pd.read_pickle(os.path.join("mydata",base_data+".pickle"))
#data = pd.read_pickle(os.path.join("mydata","sales_train_eval.pickle"))

data_path = os.path.join("mydata",base_data+"_"+feature_maker.name+".pickle")

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
model = LGBMRegressor(n_estimators=1000,
                        learning_rate=0.1,
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
data["validation"][1].loc[:] = model.predict(data["validation"][0])



# In[24]:
model = LGBMRegressor(n_estimators=1000,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        max_depth=8,
                        num_leaves=50,
                        min_child_weight=300)

model.fit(pd.concat([data["train"][0],data["validation"][0]]),
          pd.concat([data["train"][1],data["validation"][1]]),
          eval_set=[data["train"],data["validation"]],
          eval_metric='rmse',
          verbose=20,
          early_stopping_rounds=20
         )
data["evaluation"][1].loc[:] = model.predict(data["evaluation"][0])


# #### [submission]

# In[71]:


# submission
print("start submission")
sub_path = os.path.join("submission_point","LightGBM_"+feature_maker.name+"_submission.csv")


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
