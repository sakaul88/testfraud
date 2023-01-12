#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import pyTigerGraph as pyTG
import pyTigerGraph as tg
# conn = pyTG.TigerGraphConnection("http://20.58.58.175", "TX_CRD", gsqlVersion="3.1.2")
# res3 = conn.runInstalledQuery("tr_rule", params = {"start_date":'2013-03-27',"end_date":'2014-03-28'})

def Data_Base_Value(res3):
    tranx= pd.json_normalize(res3[0], record_path =['all_transac'], max_level=0)['attributes'].apply(pd.Series)
    client_info= pd.json_normalize(res3[0], record_path =['with_client_info'], max_level=0)['attributes'].apply(pd.Series)
    lon_lat_df=tranx[['TX_ID','Lat','Lon']]  
    
    tranx.drop(['Postcode', 'Lat', 'Lon','id','Fraud_Id'], axis=1, inplace=True)
    tranx= tranx.rename(columns={'TX_ID':'TX_Id','TX_Date':'HDR_CREDTT','TX_Amount':'AUREQ_TX_DT_TTLAMT','Bank_Name':'AUREQ_ENV_A_ID_ID'})
    tranx= tranx.rename(columns={'Client_ID':'CONT_ID'})
    
    client_info.drop(['id'],axis=1,inplace=True)
    client_info = client_info.rename(columns={ 'Client_ID':'CONT_ID','Gender':'GENDER','Age':'AGE_YEARS','Highest_Edu':'HIGHEST_EDU','Ann_Invest':'ANNUAL_INVEST','Activity_Level':'ACTIVITY_LEVEL','Ann_Income':'ANNUAL_INCOME','Churn':'CHURN'})
    

    return client_info,tranx ,lon_lat_df
    
    
    
                                                     
                           

    
    

                      
                      
                      
                     



