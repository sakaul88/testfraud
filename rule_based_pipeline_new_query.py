rand_state=1111
import warnings
warnings.filterwarnings('ignore')
import os
import copy
import heapq
import time
import pickle
import random
import imblearn
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
from imblearn.over_sampling import SMOTE
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

def test_pipeline(client_info,tranx,n_cl):
    start_time = datetime.now()
    print(start_time)
    
    client_info.columns= client_info.columns.str.lower()
    tranx.columns= tranx.columns.str.lower()
    
    cwd= os.getcwd()
    cons_df= pd.merge(tranx,client_info, how='inner',on=['cont_id'])
    
    input_df= copy.deepcopy(cons_df)
    
    cons_df.drop(['aureq_env_a_id_id','aureq_env_cpl_pan','aureq_tx_mrchntctgyc',\
                  'authorresult_rspnt','filler1','tx_id'], axis=1, inplace= True)
    
    col_num= cons_df.columns.get_loc('annual_invest')
    cons_df.loc[cons_df['annual_invest']<=0,'annual_invest']=1
    cons_df['log_invest']= np.log(cons_df['annual_invest'])
    cons_df['log_income']= np.log(cons_df['annual_income'])
    cons_df['log_ttlamt']= np.log(cons_df['aureq_tx_dt_ttlamt'])
    
    cons_df.drop(['aureq_tx_dt_ttlamt','annual_invest','annual_income'],axis=1,inplace=True)
    
    cons_df['log_ageyears']= np.log(cons_df['age_years'])
    
    cons_df.drop(['age_years','age'],axis=1,inplace=True)
    
#     year=[]
#     month=[]
#     for i in tqdm(range(cons_df.shape[0])):
#         a= cons_df.loc[i,'hdr_credtt'].split('-')
#         year.append(a[0])
#         month.append(a[1])
        
#     cons_df['month']= month
#     cons_df['month']= cons_df['month'].str.replace(r'^(0+)', '')
#     cons_df['year']= year

    new_date= pd.to_datetime(cons_df['hdr_credtt'].str.split('T', n = 1, expand = True)[0])   
    cons_df['month']= new_date.dt.month
    cons_df['year']= new_date.dt.year
    cons_df['year']= cons_df['year'].astype(str)
    cons_df['month']= cons_df['month'].astype(str)
    
    cons_df= cons_df[cons_df['year']=='2013']
    
    cons_df['time']= cons_df['hdr_credtt'].apply(lambda x: x.split('T')[1])
    cons_df['date']= cons_df['hdr_credtt'].apply(lambda x: x.split('T')[0])
    cons_df['day']= cons_df['date'].apply(lambda x: x.split('-')[2])
    cons_df['hour']= cons_df['time'].apply(lambda x: x.split(':')[0])
    cons_df['minute']= cons_df['time'].apply(lambda x: x.split(':')[1])

    cons_df.drop(['hdr_credtt','time','year','date'],axis=1, inplace=True)
    
    with open(cwd + '/' + 'query_rb_count_mid_id.pkl','rb') as cnt_mid:
        count_mid_id= pickle.load(cnt_mid)

    with open(cwd + '/' + 'query_rb_count_cmonnm.pkl','rb') as cnt_cmon:
        count_cmonnm= pickle.load(cnt_cmon)

    with open(cwd + '/' + 'query_rb_count_postc.pkl','rb') as cnt_pst:
        count_postc= pickle.load(cnt_pst)

    with open(cwd + '/' + 'query_rb_count_contid.pkl','rb') as cnt_cid:
        count_contid= pickle.load(cnt_cid)
    
    cons_df['month']= cons_df['month'].astype(str)
    cons_df['gender']= cons_df['gender'].map({0:'g0',1:'g1'})
    
    cons_df['aureq_env_m_id_id']= cons_df['aureq_env_m_id_id'].map(count_mid_id)
    cons_df['aureq_env_m_cmonnm']= cons_df['aureq_env_m_cmonnm'].map(count_cmonnm)
    cons_df['mdm_postal_code_id']= cons_df['mdm_postal_code_id'].map(count_postc)
    cons_df['cont_id']= cons_df['cont_id'].map(count_contid)
    
    with open(cwd + '/' + 'query_rb_card_enc.pkl','rb') as c_enc:
        card_enc= pickle.load(c_enc)
    
    with open(cwd + '/' + 'query_rb_month_enc.pkl','rb') as m_enc:
        month_enc= pickle.load(m_enc)
    
    with open(cwd + '/' + 'query_rb_gen_enc.pkl','rb') as g_enc:
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
    
    with open(cwd + '/' + 'query_rb_std_scl.pkl','rb') as std_scl:
        std_scaler= pickle.load(std_scl)
    
    consdf_scl= pd.DataFrame(std_scaler.transform(cons_df))
    consdf_scl.columns= cons_df.columns
    
    with open(cwd + '/' + 'query_rb_to_del.pkl','rb') as vi_del:
        to_del= pickle.load(vi_del)
    
    consdf_scl.drop(to_del,axis=1,inplace=True)
        
    cons_zscr_df= pd.DataFrame()
    col_names= consdf_scl.columns
    
    with open(cwd + '/' + 'query_rb_dict_params.pkl','rb') as par_dict:
        dict_params= pickle.load(par_dict)
    
    loop_in_time = datetime.now()
    for col in tqdm(col_names):
        zv= np.abs((consdf_scl[col]-dict_params[col][0])/dict_params[col][1])
        cons_zscr_df[col]= (zv-dict_params[col][2])/(dict_params[col][3]-dict_params[col][2])
#     for col in tqdm(col_names):
#         zval=[]
#         mean= dict_params[col][0]
#         std= dict_params[col][1]
#         z_min= dict_params[col][2]
#         z_max= dict_params[col][3]
    
#         for i in range(consdf_scl.shape[0]):
#             zval.append(np.abs((consdf_scl[col][i]-mean)/std))
#         zval= np.array(zval)
#         cons_zscr_df[col]= (zval-z_min)/(z_max-z_min)
    
    loop_time = (datetime.now()-loop_in_time)
    print(loop_time)
    
    cons_zscr_df= cons_zscr_df.add_prefix('zscore_')
    
    with open(cwd + '/' + 'query_rb_del_cols.pkl','rb') as dcols:
        del_cols= pickle.load(dcols)
    
    cons_zscr_df.drop(del_cols,axis=1,inplace=True)
    
    with open(cwd + '/' + 'query_rb_allcols_zscr.pkl','rb') as tr_allcols:
        tr_allcols_zscr= pickle.load(tr_allcols)
    
    allcols_zscr = cons_zscr_df.mean(axis=1)
    cons_zscr_df["all_cols_zscore"] = (allcols_zscr - np.min(tr_allcols_zscr))/(np.max(tr_allcols_zscr) - np.min(tr_allcols_zscr))
    
    with open(cwd + '/' + 'query_rb_frd_mean.pkl','rb') as fmean:
        frd_mean= pickle.load(fmean)

    with open(cwd + '/' + 'query_rb_frd_std.pkl','rb') as fstd:
        frd_std= pickle.load(fstd)    
    
    with open(cwd + '/' + 'query_rb_nonfrd_mean.pkl','rb') as nfmean:
        nonfrd_mean= pickle.load(nfmean)

    with open(cwd + '/' + 'query_rb_nonfrd_std.pkl','rb') as nfstd:
        nonfrd_std= pickle.load(nfstd)
    
    frd_ll= (frd_mean)-(n_cl*frd_std)
    frd_ul= (frd_mean)+(n_cl*frd_std)
    
    cons_zscr_df['dist_frd_mean']= cons_zscr_df['all_cols_zscore']-frd_mean
    cons_zscr_df['dist_nonfrd_mean']= cons_zscr_df['all_cols_zscore']-nonfrd_mean

    cons_zscr_df['dist_frd_mean']= cons_zscr_df['dist_frd_mean'].abs() 
    cons_zscr_df['dist_nonfrd_mean']= cons_zscr_df['dist_nonfrd_mean'].abs()

    cons_zscr_df['diff']= cons_zscr_df['dist_frd_mean']-cons_zscr_df['dist_nonfrd_mean']
    
    pos_index= cons_zscr_df[(cons_zscr_df['all_cols_zscore']>frd_ll) & (cons_zscr_df['all_cols_zscore']<frd_ul)].index
    
    input_df['predict']='non_fraud'
    
    input_df.loc[pos_index,'predict']= 'fraud'
    
    input_df['dist_frd_mean']= cons_zscr_df['dist_frd_mean']
    input_df['dist_nonfrd_mean']= cons_zscr_df['dist_nonfrd_mean']
    input_df['diff']= cons_zscr_df['diff']
    
    end_time = datetime.now()
    duration= end_time - start_time
    
    fraud_tranx= input_df[input_df['predict']=='fraud']
    model_time = time.ctime(time.time())
    need_time = datetime.strptime(model_time, '%a %b %d %H:%M:%S %Y').strftime('%d/%m/%Y  %H:%M:%S')
    
    return fraud_tranx, input_df, duration,need_time