#!/usr/bin/env python
# coding: utf-8

# In[2]:


rand_state=1111
import warnings
warnings.filterwarnings('ignore')
import os
import copy
import heapq
import time
import pickle
import random
# import imblearn
from datetime import datetime
from scipy.stats import zscore
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
# from imblearn.over_sampling import SMOTE
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

def qda_pipeline(client_info,tranx,threshold):
    start_time = datetime.now()
    print(start_time)
    
    client_info.columns= client_info.columns.str.lower()
    tranx.columns= tranx.columns.str.lower()
    
    cwd= os.getcwd()
    input_df= pd.merge(tranx,client_info, how='inner',on=['cont_id'])
    
    cons_df= copy.deepcopy(input_df)
    
    cons_df.drop(['aureq_env_a_id_id','aureq_env_cpl_pan','aureq_tx_mrchntctgyc',\
                  'authorresult_rspnt','filler1','tx_id'], axis=1, inplace= True)
    
    cons_df.loc[cons_df['annual_invest']<=0,'annual_invest']=1
    cons_df['log_invest']= np.log(cons_df['annual_invest'])
    
    cons_df['log_income']= np.log(cons_df['annual_income'])
    
    cons_df['log_ttlamt']= np.log(cons_df['aureq_tx_dt_ttlamt'])
        
    cons_df.drop(['aureq_tx_dt_ttlamt','annual_invest','annual_income'],axis=1,inplace=True)
    
    cons_df['log_ageyears']= np.log(cons_df['age_years'])
    
    cons_df.drop(['age_years','age'],axis=1,inplace=True)
    
    cons_df['time']= cons_df['hdr_credtt'].apply(lambda x: x.split('T')[1])
    cons_df['date']= cons_df['hdr_credtt'].apply(lambda x: x.split('T')[0])
    cons_df['day']= cons_df['date'].apply(lambda x: x.split('-')[2])
    cons_df['hour']= cons_df['time'].apply(lambda x: x.split(':')[0])
    cons_df['minute']= cons_df['time'].apply(lambda x: x.split(':')[1])
    cons_df.drop(['time','date'],axis=1, inplace=True)
    
    
#     year=[]
#     month=[]
#     for i in tqdm(range(cons_df.shape[0])):
#         a= cons_df.loc[i,'hdr_credtt'].split('-')
#         year.append(a[0])
#         month.append(a[1])
    


    new_date= pd.to_datetime(cons_df['hdr_credtt'].str.split('T', n = 1, expand = True)[0])   
    cons_df['month']= new_date.dt.month
    cons_df['year']= new_date.dt.year
    cons_df.drop(['hdr_credtt','year'],axis=1, inplace=True)
    
    with open(cwd + '/' + 'query_count_mid_id.pkl','rb') as cnt_mid:
        count_mid_id= pickle.load(cnt_mid)

    with open(cwd + '/' + 'query_count_cmonnm.pkl','rb') as cnt_cmon:
        count_cmonnm= pickle.load(cnt_cmon)

    with open(cwd + '/' + 'query_count_postc.pkl','rb') as cnt_pst:
        count_postc= pickle.load(cnt_pst)

    with open(cwd + '/' + 'query_count_contid.pkl','rb') as cnt_cid:
        count_contid= pickle.load(cnt_cid)
    
    cons_df['month']= cons_df['month'].astype(str)
    cons_df['gender']= cons_df['gender'].map({0:'g0',1:'g1'})
    
    cons_df['aureq_env_m_id_id']= cons_df['aureq_env_m_id_id'].map(count_mid_id)
    cons_df['aureq_env_m_cmonnm']= cons_df['aureq_env_m_cmonnm'].map(count_cmonnm)
    cons_df['mdm_postal_code_id']= cons_df['mdm_postal_code_id'].map(count_postc)
    cons_df['cont_id']= cons_df['cont_id'].map(count_contid)
    
    cons_df= cons_df.fillna(0)
    
    cons_df['month'] = cons_df['month'].astype(str)
    
    with open(cwd + '/' + 'query_card_enc.pkl','rb') as c_enc:
        card_enc= pickle.load(c_enc)
    
    with open(cwd + '/' + 'query_month_enc.pkl','rb') as m_enc:
        month_enc= pickle.load(m_enc)
    
    with open(cwd + '/' + 'query_gen_enc.pkl','rb') as g_enc:
        gen_enc= pickle.load(g_enc)
    
    cdums= card_enc.transform(cons_df['aureq_env_c_cardbrnd'].values.reshape(-1,1)).toarray()
    card_dum= pd.DataFrame(cdums,columns= list(card_enc.categories_[0]))
    card_dum.columns= card_dum.columns.str.lower()

    mdums= month_enc.transform(cons_df['month'].values.reshape(-1,1)).toarray()
    month_dum= pd.DataFrame(mdums,columns= list(month_enc.categories_[0]))
    month_dum.columns= ['month_1','month_2','month_3','month_4','month_5','month_6','month_7','month_8', 'month_9','month_10','month_11','month_12']
 
    gdums= gen_enc.transform(cons_df['gender'].values.reshape(-1,1)).toarray()
    gen_dum= pd.DataFrame(gdums,columns= list(gen_enc.categories_[0]))
    gen_dum.columns= ['g0','g1'] 
    
    cons_df= pd.concat([cons_df,card_dum,month_dum,gen_dum],axis=1)
    cons_df.drop(['aureq_env_c_cardbrnd','month','gender','g0'],axis=1, inplace=True)
    
    with open(cwd + '/' + 'query_std_scl.pkl','rb') as std_scl:
        std_scaler= pickle.load(std_scl)
    
    consdf_scl= pd.DataFrame(std_scaler.transform(cons_df))
    consdf_scl.columns= cons_df.columns
    
    to_del= ['amex platinum','credit card','debit card','visa gold']
    
    consdf_scl.drop(to_del,axis=1,inplace=True)
    
    with open(cwd + '/' + 'query_qda_model.pkl','rb') as qda_mod:
        model= pickle.load(qda_mod)
    
    cons_prob_p = model.predict_proba(consdf_scl)
    frd_probs_cons_p = cons_prob_p[:, 1]
    
    input_df['fraud_probability']= frd_probs_cons_p
    input_df['predict']= 'non_fraud'
    input_df.loc[input_df['fraud_probability']>threshold, 'predict']= 'fraud'
    
    print("All transactions with fraud probability greater than 0.98 are considered as Fraudulent Transactions")
    
    fraud_tranx= input_df[input_df['predict']=='fraud']
    
    end_time = datetime.now()
    print(end_time)
    duration= end_time - start_time
    model_time = time.ctime(time.time())
    need_time = datetime.strptime(model_time, '%a %b %d %H:%M:%S %Y').strftime('%d/%m/%Y  %H:%M:%S')


    return fraud_tranx, input_df, duration,need_time