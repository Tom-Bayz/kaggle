import numpy as np 
import pandas as pd

def make_base_features(df):
    
    # necessary columns to make output
    necessary_col = [
                     'timestamp',
                     'user_id',
                     'content_id',
                     'content_type_id',
                     'task_container_id',
                     'user_answer',
                     'answered_correctly',
                     'prior_question_elapsed_time',
                     'prior_question_had_explanation',
                     "answered_correctly"
                    ]
    # check columns
    df = df[necessary_col]
        
    # make features from here    
    train_df["prior_question_had_explanation"] = train_df["prior_question_had_explanation"].astype("float")
    
    return df