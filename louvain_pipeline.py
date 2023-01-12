import pyTigerGraph as tg
import pandas as pd
from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import collections
from tqdm import tqdm


def louvain_result(connection,clients,transactions):
    startTime = time.time()
    connection.runInstalledQuery("user_transactioncount")
    louv_res = connection.runInstalledQuery("louvien_query", params = {"v_type":["user"],"e_type":["user_edge"],"wt_attr":"amount","max_iter":10,"result_attr":"","print_info":"true"})    
    louvain_communities= list(louv_res[-1].get('FinalCommunityCount').keys())
        
    
    louv_comm=[]

    for i in tqdm(range(clients.shape[0])):
        c_list=[]
        for k in list(louvain_communities):
            v_list= louv_res[-1]['FinalCommunityCount'][k]
     
            if str(clients['cont_id'][i]) in v_list:
                c_list.append(k)
    
        louv_comm.append(c_list[0])
    
    clients['louvain_communities']= louv_comm
    
    merged_df= pd.merge(transactions,clients,how='inner',left_on='orig_id',right_on='cont_id')
    
    louv_groups= pd.DataFrame(merged_df.groupby(['louvain_communities'])['isfraud'].value_counts())
    louv_groups.columns=['counts']
    louv_groups.reset_index(inplace=True)
    
    no_frd= list(set(louv_groups[louv_groups['isfraud']==0]['louvain_communities']).difference(set(louv_groups[louv_groups['isfraud']==1]['louvain_communities'])))
    nf_cnts= np.zeros(len(no_frd))

    nf_df= pd.DataFrame(list(zip(no_frd,nf_cnts)),columns=['louvain_communities','counts'])
    louv_frd_grp= louv_groups[louv_groups['isfraud']==1]
    louv_frd_grp.drop(['isfraud'],axis=1,inplace=True)
    louv_frd_grp= louv_frd_grp.append(nf_df, ignore_index = True)
    
    new_merge= pd.merge(clients,louv_groups, how='inner',left_on='louvain_communities',right_on='louvain_communities')
    new_merge.rename(columns={'counts':'frauds_in_community'},inplace=True)
    
    merged_df_new= pd.merge(merged_df,louv_frd_grp, how='inner',left_on='louvain_communities',right_on='louvain_communities')
    merged_df_new.rename(columns={'counts':'frauds_in_community'},inplace=True)

    frd_df= new_merge[new_merge['isfraud']==1]
    frauds= sum(frd_df['frauds_in_community'].unique())
    merged_df_new.loc[:,'fraud_penetration']= np.round(np.round(merged_df_new['frauds_in_community']/frauds,4)*100,2)
    
    executionTime = (time.time() - startTime)
    
    return new_merge,merged_df_new,executionTime,louvain_communities