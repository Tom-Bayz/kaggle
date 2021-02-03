class config:
    device = torch.device("cuda") 
    MAX_SEQ = 100 
    EMBED_DIMS = 512 
    ENC_HEADS = DEC_HEADS = 8
    NUM_ENCODER = NUM_DECODER = 4
    BATCH_SIZE = 64
    TRAIN_FILE = "../input/riiid-test-answer-prediction/train.csv"
    TOTAL_EXE = 13523
    TOTAL_CAT = 10000
        
        
#####################
# DataSet
#####################
class DKTDataset(Dataset):
    def __init__(self,samples,max_seq,start_token=0): 
        super().__init__()
        self.samples = samples
        self.max_seq = max_seq
        self.start_token = start_token
        self.data = []
        for id in self.samples.index:
            exe_ids,answers,ela_time,categories = self.samples[id]
            if len(exe_ids)>max_seq:
                for l in range((len(exe_ids)+max_seq-1)//max_seq):
                    self.data.append((exe_ids[l:l+max_seq],answers[l:l+max_seq],ela_time[l:l+max_seq],categories[l:l+max_seq]))
            elif len(exe_ids)<self.max_seq and len(exe_ids)>10:
                self.data.append((exe_ids,answers,ela_time,categories))
            else :
                continue

    def __len__(self):
        return len(self.data)
  
    def __getitem__(self,idx):
        question_ids,answers,ela_time,exe_category = self.data[idx]
        seq_len = len(question_ids)

        exe_ids = np.zeros(self.max_seq,dtype=int)
        ans = np.zeros(self.max_seq,dtype=int)
        elapsed_time = np.zeros(self.max_seq,dtype=int)
        exe_cat = np.zeros(self.max_seq,dtype=int)
        if seq_len<self.max_seq:
            exe_ids[-seq_len:] = question_ids
            ans[-seq_len:] = answers
            elapsed_time[-seq_len:] = ela_time 
            exe_cat[-seq_len:] = exe_category
        else:
            exe_ids[:] = question_ids[-self.max_seq:]
            ans[:] = answers[-self.max_seq:]
            elapsed_time[:] = ela_time[-self.max_seq:]
            exe_cat[:] = exe_category[-self.max_seq:]

        input_rtime = np.zeros(self.max_seq,dtype=int)
        input_rtime = np.insert(elapsed_time,0,self.start_token)
        input_rtime = np.delete(input_rtime,-1)

        input = {"input_ids":exe_ids,"input_rtime":input_rtime.astype(np.int),"input_cat":exe_cat}
        answers = np.append([0],ans[:-1]) #start token
        assert ans.shape[0]==answers.shape[0] and answers.shape[0]==input_rtime.shape[0], "both ans and label should be same len with start-token"
        return input,answers,ans
    

#####################
# SAINT+ Model
#####################
class FFN(nn.Module):
    def __init__(self,in_feat):
        super(FFN,self).__init__()
        self.linear1 = nn.Linear(in_feat,in_feat)
        self.linear2 = nn.Linear(in_feat,in_feat)
        self.drop = nn.Dropout(0.2)

    def forward(self,x):
        out = F.relu(self.drop(self.linear1(x)))
        out = self.linear2(out)
        return out 


class EncoderEmbedding(nn.Module):
    def __init__(self,n_exercises,n_categories,n_dims,seq_len):
        super(EncoderEmbedding,self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.exercise_embed = nn.Embedding(n_exercises,n_dims)
        self.category_embed = nn.Embedding(n_categories,n_dims)
        self.position_embed = nn.Embedding(seq_len,n_dims)

    def forward(self,exercises,categories):
        e = self.exercise_embed(exercises)
        c = self.category_embed(categories)
        seq = torch.arange(self.seq_len,device=config.device).unsqueeze(0)
        p = self.position_embed(seq)
        return p + c + e

class DecoderEmbedding(nn.Module):
    def __init__(self,n_responses,n_dims,seq_len):
        super(DecoderEmbedding,self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.response_embed = nn.Embedding(n_responses,n_dims)
        self.time_embed = nn.Linear(1,n_dims,bias=False)
        self.position_embed = nn.Embedding(seq_len,n_dims)

    def forward(self,responses):
        e = self.response_embed(responses)
        seq = torch.arange(self.seq_len,device=config.device).unsqueeze(0)
        p = self.position_embed(seq)
        return p + e 


# layers of encoders stacked onver, multiheads-block in each encoder is n.
# Stacked N MultiheadAttentions 
class StackedNMultiHeadAttention(nn.Module):
    def __init__(self,n_stacks,n_dims,n_heads,seq_len,n_multihead=1,dropout=0.2):
        super(StackedNMultiHeadAttention,self).__init__()
        self.n_stacks = n_stacks
        self.n_multihead = n_multihead
        self.n_dims = n_dims 
        self.norm_layers = nn.LayerNorm(n_dims)
        #n_stacks has n_multiheads each
        self.multihead_layers = nn.ModuleList(n_stacks*[nn.ModuleList(n_multihead*[nn.MultiheadAttention(embed_dim = n_dims,
                                                          num_heads = n_heads,
                                                            dropout = dropout),]),])
        self.ffn = nn.ModuleList(n_stacks*[FFN(n_dims)])
        self.mask = torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool)

    def forward(self,input_q,input_k,input_v,encoder_output=None,break_layer=None):
        for stack in range(self.n_stacks):
            for multihead in range(self.n_multihead):
                norm_q = self.norm_layers(input_q)
                norm_k = self.norm_layers(input_k)
                norm_v = self.norm_layers(input_v) 
                heads_output,_ = self.multihead_layers[stack][multihead](query=norm_q.permute(1,0,2),
                                                                        key=norm_k.permute(1,0,2),
                                                                        value=norm_v.permute(1,0,2),
                                                                        attn_mask=self.mask.to(config.device))
                heads_output = heads_output.permute(1,0,2)
                #assert encoder_output != None and break_layer is not None     
            if encoder_output != None and multihead == break_layer:
                assert break_layer <= multihead, " break layer should be less than multihead layers and postive integer"
                    input_k = input_v = encoder_output
                    input_q =input_q + heads_output
                else:
                    input_q =input_q+ heads_output
                    input_k =input_k+ heads_output
                    input_v =input_v +heads_output
            last_norm = self.norm_layers(heads_output)
            ffn_output = self.ffn[stack](last_norm)
            ffn_output =ffn_output+ heads_output
        return ffn_output
    

def get_dataloaders():              
    dtypes = {'timestamp': 'int64', 'user_id': 'int32' ,'content_id': 'int16',
                'answered_correctly':'int8',"content_type_id":"int8",
                  "prior_question_elapsed_time":"float32","task_container_id":"int16"}
    print("loading csv.....")
    train_df = pd.read_csv(config.TRAIN_FILE,usecols=[1,2,3,4,5,7,8],dtype=dtypes)
    print("shape of dataframe :",train_df.shape) 

    train_df = train_df[train_df.content_type_id==0] 
    train_df.prior_question_elapsed_time.fillna(0,inplace=True)
    train_df.prior_question_elapsed_time /=3600 
    #train_df.prior_question_elapsed_time.clip(lower=0,upper=300,inplace=True)
    train_df.prior_question_elapsed_time = train_df.prior_question_elapsed_time.astype(np.int)
    
    train_df = train_df.sort_values(["timestamp"],ascending=True).reset_index(drop=True)
    n_skills = train_df.content_id.nunique() 
    print("no. of skills :",n_skills)
    print("shape after exlusion:",train_df.shape)

    #grouping based on user_id to get the data supplu
    print("Grouping users...") 
    group = train_df[["user_id","content_id","answered_correctly","prior_question_elapsed_time","task_container_id"]]\
                    .groupby("user_id")\
                    .apply(lambda r: (r.content_id.values,r.answered_correctly.values,\
                                      r.prior_question_elapsed_time.values,r.task_container_id.values))
    del train_df
    gc.collect() 
    print("splitting") 
    train,val = train_test_split(group,test_size=0.2) 
    print("train size: ",train.shape,"validation size: ",val.shape)
    train_dataset = DKTDataset(train,max_seq = config.MAX_SEQ)
    val_dataset = DKTDataset(val,max_seq = config.MAX_SEQ)
    train_loader = DataLoader(train_dataset,
                          batch_size=config.BATCH_SIZE,
                          num_workers=8,
                          shuffle=True) 
    val_loader = DataLoader(val_dataset,
                          batch_size=config.BATCH_SIZE,
                          num_workers=8,
                          shuffle=False)
    del train_dataset,val_dataset 
    gc.collect() 
    return train_loader, val_loader 
train_loader, val_loader = get_dataloaders()