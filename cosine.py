import time
import random
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyTigerGraph as tg


def cosine_result(connection,clients, transactions,sim_ver):
    startTime = time.time()
    cosine_res = connection.runInstalledQuery("coisine", params = {"vertex_type":"user","edge_type":"Transactions","edge_attribute":"amount","top_k":sim_ver,"print_accum":"true","file_path":"","similarity_edge":"","num_of_batches":1})
    executionTime = (time.time() - startTime)
    
    pri_id=[]
    sim_id=[]
    only_id=[]

    for i in tqdm(range(len(cosine_res[0]['start']))):
        pri_id.append(int(cosine_res[0]['start'][i]['v_id']))
        sim_id.append(cosine_res[0]['start'][i]['attributes']['start.@heap'])
        
        temp_id=[]
        for x in range(sim_ver):
            temp_id.append(cosine_res[0]['start'][i]['attributes']['start.@heap'][x]['ver'])
        only_id.append(temp_id)

    cos_df= pd.DataFrame()
    cos_df['cust_id']= pri_id
    cos_df['similar_id']= sim_id
    cos_df['only_id']= only_id
    
    frd_grps= transactions.groupby(['orig_id'])
    frds=[]
    sim_frds=[]

    for j in tqdm(range(cos_df.shape[0])):
        local_frd=[]
        frds.append(frd_grps.get_group(cos_df['cust_id'][j])[frd_grps.get_group(cos_df['cust_id'][j])['isfraud']==1].shape[0])
    
        for k in range(sim_ver):
            local_frd.append(frd_grps.get_group(int(cos_df['similar_id'][j][k]['ver']))[frd_grps.get_group(int(cos_df['similar_id'][j][k]['ver']))['isfraud']==1].shape[0])
        sim_frds.append(local_frd)
    
    cos_df['frds']= frds
    cos_df['sim_frds']= sim_frds
    
    cos_df.drop(['similar_id'],axis=1,inplace=True)
    
    executionTime = (time.time() - startTime)
    
    return cos_df,executionTime