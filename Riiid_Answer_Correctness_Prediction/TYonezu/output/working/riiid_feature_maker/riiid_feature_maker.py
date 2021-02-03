import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
    #
    return df

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns: #columns毎に処理
        col_type = df[col].dtypes
        if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
    

def make_base_features(df):
    
    target = "answered_correctly"
    features = ['row_id',
                'timestamp',
                'user_id',
                'content_id',
                'content_type_id',
                'task_container_id',
                'user_answer',
                'prior_question_elapsed_time',
                'prior_question_had_explanation']
    
    
    necessary_col = features + [target]
    
    df = df[necessary_col]
    
    # make features from here    
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].astype("float")
    df = reduce_mem_usage(df,verbose=False)
    
    new_features = ['row_id',
                    'timestamp',
                    'user_id',
                    'content_id',
                    'content_type_id',
                    'task_container_id',
                    'user_answer',
                    'answered_correctly',
                    'prior_question_elapsed_time',
                    'prior_question_had_explanation'] + [target]

    return (new_features,df)


###################################################################################
# ここから特徴量追加 関数を定義
# データフレームを入力にして, 出力を (追加された特徴量名str, 新しいDataFrame)で定義
# wirtten by T.Yonezu
####################################################################################

################################################################################
# Userとcontentsに関して, answered_correctlyの平均, 合計, カウントを新たな特徴とする
################################################################################
def add_TargetFeatures(df, train_df):
    
    train_df = train_df[["user_id","content_id","answered_correctly"]]
    
    ##### features about users #####################################################################
    answered_correctly_for_users = train_df.groupby("user_id").agg({"answered_correctly":["mean","sum","count"]})
    answered_correctly_for_users.columns = ["answered_correctly_users_mean",
                                            "answered_correctly_users_sum",
                                            "answered_correctly_users_count"]
    answered_correctly_for_users = answered_correctly_for_users.reset_index()
    
    df = pd.merge(df,answered_correctly_for_users,on="user_id",how="left")
    
    ##### features about contents ####################################################################
    answered_correctly_for_contents = train_df.groupby("content_id").agg({"answered_correctly":["mean","sum","count"]})
    answered_correctly_for_contents.columns = ["answered_correctly_contents_mean",
                                               "answered_correctly_contents_sum",
                                               "answered_correctly_contents_count"]
    answered_correctly_for_contents = answered_correctly_for_contents.reset_index()
    
    df = pd.merge(df,answered_correctly_for_contents,on="content_id",how="left")
    
    ###################################################################################################
    
    new_features = ["answered_correctly_users_mean",
                    "answered_correctly_users_sum",
                    "answered_correctly_users_count",
                    "answered_correctly_contents_mean",
                    "answered_correctly_contents_sum",
                    "answered_correctly_contents_count"]
    
    df[new_features] = df[new_features].fillna(df[new_features].mean())
    
    return (new_features, df)















#########################################################################
# メモリ削減バージョン, userごとのanswered_correctlyの合計, カウント, 平均
#########################################################################
from collections import defaultdict
from tqdm import tqdm

def init_user_dict(train_df):
    
    train_df = train_df[["user_id","answered_correctly"]]
    train_df = train_df[train_df["answered_correctly"] != -1]
    
    tmp = train_df.groupby("user_id").agg({"answered_correctly":["sum","count"]})
    tmp.columns = ["answered_correctly_user_sum",
                   "answered_correctly_user_count"]
    tmp = tmp.reset_index()
    
    user_sum_dict = defaultdict(int, zip(tmp["user_id"].values,tmp["answered_correctly_user_sum"].values))
    user_count_dict = defaultdict(int, zip(tmp["user_id"].values,tmp["answered_correctly_user_count"].values))
    
    return (user_sum_dict, user_count_dict)

def update_user_dict(train_df, user_sum_dict, user_count_dict):
    
    train_df = train_df[["user_id","answered_correctly"]]
    train_df = train_df[train_df["answered_correctly"] != -1]
    
    for idx, row in (train_df.iterrows()):
        user_sum_dict[row["user_id"]] += row["answered_correctly"]                                    # sum
        user_count_dict[row["user_id"]] += 1                                                          # count
        
    return (user_sum_dict, user_count_dict)

def add_UserFeatures(df, user_dict):
    
    cols=["answered_correctly_users_sum",
          "answered_correctly_users_count"]
    
    tmp = pd.concat([pd.Series(user_dict[0]), pd.Series(user_dict[1])] ,axis=1)
    tmp.index.name = "user_id"
    tmp.columns = cols
    tmp = tmp.reset_index()
    df = pd.merge(df,tmp,on="user_id",how="left")

    df["answered_correctly_users_mean"] = df["answered_correctly_users_sum"]/df["answered_correctly_users_count"]
    
    new_features = ["answered_correctly_users_sum",
                    "answered_correctly_users_count",
                    "answered_correctly_users_mean"]
    
    df[new_features] = df[new_features].fillna(df[new_features].mean())
    
    return (new_features, df)












#########################################################################
# メモリ削減バージョン, contentごとのanswered_correctlyの合計, カウント, 平均
#########################################################################
from collections import defaultdict
from tqdm import tqdm

def init_content_dict(train_df):
    
    train_df = train_df[["content_id","answered_correctly"]]
    train_df = train_df[train_df["answered_correctly"] != -1]
    
    tmp = train_df.groupby("content_id").agg({"answered_correctly":["sum","count"]})
    tmp.columns = ["answered_correctly_contents_sum",
                   "answered_correctly_contents_count"]
    tmp = tmp.reset_index()
    
    content_sum_dict = defaultdict(int, zip(tmp["content_id"].values,tmp["answered_correctly_contents_sum"].values))
    content_count_dict = defaultdict(int, zip(tmp["content_id"].values,tmp["answered_correctly_contents_count"].values))
    
    return (content_sum_dict, content_count_dict)

def update_content_dict(train_df, content_sum_dict, content_count_dict):
    
    train_df = train_df[["content_id","answered_correctly"]]
    train_df = train_df[train_df["answered_correctly"] != -1]
    
    for idx, row in (train_df.iterrows()):
        content_sum_dict[row["content_id"]] += row["answered_correctly"]                                    # sum
        content_count_dict[row["content_id"]] += 1                                                          # count
        
    return (content_sum_dict, content_count_dict)


def add_ContentFeatures(df, content_dict):
    
    cols=["answered_correctly_contents_sum",
          "answered_correctly_contents_count"]
    
    tmp = pd.concat([pd.Series(content_dict[0]), pd.Series(content_dict[1])] ,axis=1)
    tmp.index.name = "user_id"
    tmp.columns = cols
    tmp = tmp.reset_index()
    df = pd.merge(df,tmp,on="user_id",how="left")
    
    df["answered_correctly_contents_mean"] = df["answered_correctly_contents_sum"]/df["answered_correctly_contents_count"]
    
    new_features = ["answered_correctly_contents_mean",
                    "answered_correctly_contents_sum",
                    "answered_correctly_contents_count"]
    
    df[new_features] = df[new_features].fillna(df[new_features].mean())
    
    return (new_features, df)











#########################################################################
# questionの特徴量を追加する
#########################################################################
def add_QuestionFeatures(df, questions):
    questions = questions.rename(columns={"question_id":"content_id",
                                          "part":"question_part"})  
    df = pd.merge(df, questions[['content_id', 'question_part']], on='content_id', how='left')  
    
    new_features = ["question_part"]
    
    return (new_features, df)



#########################################################################
# メモリ削減バージョン, userごとのlectureの数, 平均
#########################################################################
from collections import defaultdict
from tqdm import tqdm

def init_lecture_dict(train_df):
    
    train_df = train_df[["user_id","content_type_id"]]
    
    tmp = train_df.groupby("user_id").agg({"content_type_id":["sum","count"]})
    tmp.columns = ["lecture_user_sum",
                   "lecture_user_count"]
    tmp = tmp.reset_index()
    
    lecture_sum_dict = defaultdict(int, zip(tmp["user_id"].values,tmp["lecture_user_sum"].values))
    lecture_count_dict = defaultdict(int, zip(tmp["user_id"].values,tmp["lecture_user_count"].values))
    
    return (lecture_sum_dict, lecture_count_dict)

def update_lecture_dict(train_df, lecture_sum_dict, lecture_count_dict):
    
    train_df = train_df[["user_id","content_type_id"]]
    
    for idx, row in (train_df.iterrows()):
        lecture_sum_dict[row["user_id"]] += row["content_type_id"]                                    # sum
        lecture_count_dict[row["user_id"]] += 1                                                          # count
        
    return (lecture_sum_dict, lecture_count_dict)


def add_LectureLvFeatures(df, lecture_dict):
    
    cols=["lecture_user_sum",
          "lecture_user_count"]
    
    tmp = pd.concat([pd.Series(lecture_dict[0]), pd.Series(lecture_dict[1])] ,axis=1)
    tmp.index.name = "user_id"
    tmp.columns = cols
    tmp = tmp.reset_index()
    df = pd.merge(df,tmp,on="user_id",how="left")
    
    df["lecture_user_mean"] = df["lecture_user_sum"]/df["lecture_user_count"]
    
    new_features = ["lecture_user_mean",
                    "lecture_user_count",
                    "lecture_user_sum"]
    
    df[new_features] = df[new_features].fillna(df[new_features].mean())
    
    return (new_features, df)







#########################################################################
# lectureの特徴量を追加する
#########################################################################
def add_LectureFeatures(df, lectures):
    lectures = lectures.rename(columns={"lecture_id":"content_id",
                                        "part":"lecture_part",
                                        "type_of":"lecture_type",
                                        "tag":"lecture_tag"})
    
    lectures = encode_categorical(lectures, cols=["lecture_type"])
    df = pd.merge(df, lectures[['content_id', "lecture_part","lecture_type","lecture_tag"]], on='content_id', how='left') 
    
    new_features = ["lecture_part",
                    "lecture_type",
                    "lecture_tag"]
    
    return (new_features, df)






#########################################################################
# 時間の特徴量を追加する
#########################################################################
def add_TimeFeatures(df,train_df):
    
    train_df = train_df[["user_id","content_id","timestamp"]]
    
    df["elapsed_time"] = df.groupby(by="user_id")[["timestamp"]].diff()
    df["elapsed_time"] = df["elampsed_time"].fillna(0)
    
    lecture_df = train_df[train_df[""]]
    
    new_features = ["elapsed_time"]
    
    return (new_features, df)
    
    
    
def fillna_prior_question_elapsed_time(df, train_df):  
    df['prior_question_elapsed_time'] = df['prior_question_elapsed_time'].fillna(train_df['prior_question_elapsed_time'])
    return df



###################################################################################
# xxxx
# xxxx
# wirtten by H.Kato
####################################################################################

def exclude_lecture_rows(df):
    # content_type_id が 0 (問題) の行だけ残す
    df = df.loc[df['content_type_id'] == 0].reset_index(drop=True)
    return df
    
def add_answered_correctly_avg_c(df_train, df_valid):
    
    # content_id ごとの answered_correctly の平均を特徴量として追加
    df_content = df_train[['content_id', 'answered_correctly']].groupby(['content_id']).agg(['mean']).reset_index() # 平均は train の情報のみで計算
    df_content.columns = ['content_id', 'answered_correctly_avg_c']
    df_train = pd.merge(df_train, df_content, on=['content_id'], how='left')
    df_valid = pd.merge(df_valid, df_content, on=['content_id'], how='left')
    
    return df_train, df_valid, df_content

def add_user_features(df, answered_correctly_sum_u_dict, count_u_dict):
    # user_id ごとの answered_correctly 値の平均, 値の合計, インスタンス数の合計を特徴量として追加
    # answered_correctly_sum_u_dict : key = user_id, valud = answered_correctly の合計値
    # count_u_dict : key = user_id, valud = answered_correctly のインスタンス数の合計
    # 辞書 answered_correctly_sum_u_dict, count_u_dict 自体の更新も行う
    
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    for cnt, row in enumerate(tqdm(df[['user_id', 'answered_correctly']].values)):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
        answered_correctly_sum_u_dict[row[0]] += row[1]
        count_u_dict[row[0]] += 1
        
    df_user_features = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu})
    df_user_features['answered_correctly_avg_u'] = df_user_features['answered_correctly_sum_u'] / df_user_features['count_u']
    df = pd.concat([df, df_user_features], axis=1)
    return df

def add_user_features_without_update(df, answered_correctly_sum_u_dict, count_u_dict):
    
    # answered_correctly_sum_u_dict, count_u_dict 自体の更新は行わない
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    for cnt, row in enumerate(df[['user_id']].values):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
    df_user_features = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu})
    df_user_features['answered_correctly_avg_u'] = df_user_features['answered_correctly_sum_u'] / df_user_features['count_u']
    df = pd.concat([df, df_user_features], axis=1)
    return df

def update_user_features(df, answered_correctly_sum_u_dict, count_u_dict):
    for row in df[['user_id', 'answered_correctly', 'content_type_id']].values:
        if row[2] == 0:
            answered_correctly_sum_u_dict[row[0]] += row[1]
            count_u_dict[row[0]] += 1

def fillna_prior_question_elapsed_time(df_train, df_valid):
    prior_question_elapsed_time_mean = df_train['prior_question_elapsed_time'].dropna().values.mean()
    df_train['prior_question_elapsed_time'] = df_train['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean)
    df_valid['prior_question_elapsed_time'] = df_valid['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean)
    
def add_question_part(df, questions):
    df = pd.merge(df, questions[['question_id', 'part']], left_on='content_id', right_on='question_id', how='left')
    return df
    
def fill_prior_question_had_explanation(df):
    df['prior_question_had_explanation'] = df['prior_question_had_explanation'].fillna(False).astype('int8')
    return df
    