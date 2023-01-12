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
from colour import Color
import webcolors
import copy

louvain_result_df = pd.DataFrame()


def louvain_graph(louv_result,louvain_trans_df):

    large= pd.merge(louvain_trans_df,louv_result[louv_result['isfraud']==0][['cont_id','louvain_communities']],left_on='dest_id',right_on='cont_id',how='inner')
    large.rename(columns={'louvain_communities_x':'orig_louv_comm','louvain_communities_y':'dest_louv_comm'},inplace = True)
    large.drop(['cont_id_y'],axis=1,inplace=True)
    
    global louvain_df
    global louvain_result_df
    global closest_name
    
    louvain_result_df = copy.deepcopy(louvain_trans_df)
    large_dummy = copy.deepcopy(large)
    louvain_df = large
    
    large_dummy_sorted = large_dummy.sort_values('frauds_in_community',ascending=False)
    
    trnx_counts= large_dummy_sorted.groupby('orig_louv_comm')['dest_louv_comm'].value_counts().to_dict()
    cnt_df= pd.DataFrame(trnx_counts.values(),index = trnx_counts.keys())
    cnt_df.reset_index(inplace=True)
    cnt_df= cnt_df.rename(columns={'level_0':'orig_community','level_1':'dest_community',0:'counts'})
    
    louvain_communities = list(large_dummy_sorted.orig_louv_comm.unique())
    
    color_list_random = []
    import numpy as np
    for i in range(len(louvain_communities)):
        random_color=list(np.random.choice(range(255),size=3))
        color_list_random.append(random_color)


    def closest_colour(requested_colour):
        min_colours = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    def get_colour_name(requested_colour):
        color_names = []
        for color_name in requested_colour:
            try:
                closest_name = webcolors.rgb_to_name(color_name)
            except ValueError:
                closest_name = closest_colour(color_name)
            color_names.append(closest_name)
        return color_names

    requested_colour = color_list_random
    closest_name = get_colour_name(requested_colour)
#     red = Color("red")
#     colors = list(red.range_to(Color("green"),len(louvain_communities)))
    
#     color_list = [str(color) for color in colors]
         
    global color_mapping
    
    color_mapping = dict(zip(louvain_communities,closest_name))
    community_list = list(color_mapping.keys())
    community_list = [int(x) for x in community_list]
    
    cnt_df['color_orig'] = cnt_df.orig_community.map(color_mapping)
    cnt_df['color_dest'] = cnt_df.dest_community.map(color_mapping)
    
    cnt_df['orig_community'] = cnt_df['orig_community'].astype(int)
    cnt_df['dest_community'] = cnt_df['dest_community'].astype(int)
    large_dummy_sorted['orig_louv_comm'] = large_dummy_sorted['orig_louv_comm'].astype(int)
    
    orig_comm_node = cnt_df['orig_community']
    dest_comm_node = cnt_df['dest_community']
    orig_comm_node = [int(x) for x in orig_comm_node]
    dest_comm_node = [int(x) for x in dest_comm_node]
    
    node_size = len(community_list)*10
    
    pyvis_network = Network('500px','100%',directed=True,filter_menu=True)
    
    for comm_node in tqdm(community_list):
        color_value = cnt_df.loc[cnt_df['orig_community'] == comm_node,'color_orig'].iloc[0]
        fraud_penetration = large_dummy_sorted.loc[large_dummy_sorted['orig_louv_comm']==comm_node,'fraud_penetration'].iloc[0]

        comm_node_info = f'Community ID: {comm_node}' + '\n' + f'Fraud Penetration(%): {fraud_penetration}'

        pyvis_network.add_node(comm_node,label=comm_node_info,color = color_value,size=node_size,shape='star')
        node_size-=10
    
    for source, destination in tqdm(zip(orig_comm_node,dest_comm_node)):
        transaction_count = cnt_df.loc[(cnt_df['orig_community'] == source)&(cnt_df['dest_community'] == destination),'counts'].iloc[0]
        edge_info = f'Transaction count: {transaction_count}'

        pyvis_network.add_edge(source,destination,label=edge_info)
    
    options = {
      "physics": {
        "forceAtlas2Based": {
          "springLength": 0,
          "springConstant": 0
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    pyvis_network.options = options
    pyvis_network.save_graph(f'./static/Louvain_Master_Visualisation.html')
    return "http://10.2.0.20:8888/static/Louvain_Master_Visualisation.html",louvain_communities #for onprem server


def community_level_graph(transactions,louv_result,community_id):
    cust_trnx= transactions.groupby(['orig_id'])['dest_id'].value_counts().to_dict()
    cust_df= pd.DataFrame(cust_trnx.values(),index = cust_trnx.keys())
    cust_df.reset_index(level=(0,1),inplace=True)
    cust_df= cust_df.rename(columns={'level_0':'custid_1','level_1':'custid_2',0:'total_trnx'})

    cust_df_1= pd.merge(cust_df,louv_result[louv_result['isfraud']==0][['cont_id','louvain_communities']], left_on='custid_1', right_on='cont_id',how='inner')
    cust_df_1.drop(['cont_id'],axis=1,inplace=True)
    cust_df_1= cust_df_1.rename(columns={'louvain_communities':'custid_1_communuity'})

    cust_df_1= pd.merge(cust_df_1,louv_result[louv_result['isfraud']==0][['cont_id','louvain_communities']], left_on='custid_2', right_on='cont_id',how='inner')
    cust_df_1.drop(['cont_id'],axis=1,inplace=True)
    cust_df_1= cust_df_1.rename(columns={'louvain_communities':'custid_2_communuity'})

    cust_frd_trnx= transactions[transactions['isfraud']==1].groupby(['orig_id'])['dest_id'].value_counts().to_dict()
    cust_frd_df= pd.DataFrame(cust_frd_trnx.values(),index = cust_frd_trnx.keys())
    cust_frd_df.reset_index(level=(0,1),inplace=True)
    cust_frd_df= cust_frd_df.rename(columns={'level_0':'orig','level_1':'dest',0:'frd_trnx'})

    cust_df_2= pd.merge(cust_frd_df,cust_df_1, right_on=['custid_1','custid_2'], left_on=['orig','dest'], how='right')
    cust_df_2.drop(['orig','dest'], axis=1, inplace=True)

    cust_df_2= pd.merge(cust_df_1,cust_frd_df, left_on=['custid_1','custid_2'], right_on=['orig','dest'], how='left')
    cust_df_2.drop(['orig','dest'], axis=1, inplace=True)
    cust_df_2['frd_trnx']= cust_df_2['frd_trnx'].fillna(0)
    cust_df_2['frd_trnx'] = cust_df_2['frd_trnx'].astype(int)

    cust_df_comm = cust_df_2[(cust_df_2['custid_1_communuity']==community_id)&(cust_df_2['frd_trnx']==1)]

    cust_df_comm['color_orig'] = cust_df_comm.custid_1_communuity.map(color_mapping)
    cust_df_comm['color_dest'] = cust_df_comm.custid_2_communuity.map(color_mapping)

    origin_node = cust_df_comm['custid_1']
    origin_node = list(map(str,origin_node))

    destination_node = cust_df_comm['custid_2']
    destination_node = list(map(str,destination_node))

    pyvis_net_community = Network('500px','100%',filter_menu = True, directed=True)

    for orig_node in tqdm(origin_node):
        color_value = cust_df_comm.loc[cust_df_comm['custid_1'] == int(orig_node),'color_orig'].iloc[0]
        comm_id = cust_df_comm.loc[cust_df_comm['custid_1'] == int(orig_node),'custid_1_communuity'].iloc[0]
        fraud_trans = cust_df_comm.loc[cust_df_comm['custid_1'] == int(orig_node),'frd_trnx'].iloc[0]
        orig_node_info = f'Community ID: {comm_id}' + '\n' + f'Customer ID: {orig_node}' + '\n' + f'Involved in {fraud_trans} frauds'

        pyvis_net_community.add_node(int(orig_node),label=orig_node_info,size=30,color=color_value)

    for dest_node in tqdm(destination_node):
        color_value = cust_df_comm.loc[cust_df_comm['custid_2'] == int(dest_node),'color_dest'].iloc[0]
        comm_id = cust_df_comm.loc[cust_df_comm['custid_2'] == int(dest_node),'custid_2_communuity'].iloc[0]
        fraud_trans = cust_df_comm.loc[cust_df_comm['custid_2'] == int(dest_node),'frd_trnx'].iloc[0]
        dest_node_info = f'Community ID: {comm_id}' + '\n' + f'Customer ID: {dest_node}' + '\n' + f'Involved in {fraud_trans} frauds'

        pyvis_net_community.add_node(int(dest_node),label=dest_node_info,size=30,color=color_value)

    from_node = origin_node
    to_node = destination_node 

    for source,destination in tqdm(zip(from_node,to_node)):
        transaction_count = cust_df_comm.loc[(cust_df_comm['custid_1'] == int(source))&(cust_df_comm['custid_2'] == int(destination)),'total_trnx'].iloc[0]
        frauds_count = cust_df_comm.loc[(cust_df_comm['custid_1'] == int(source))&(cust_df_comm['custid_2'] == int(destination)),'frd_trnx'].iloc[0]

        edge_info = f'Total Transactions: {transaction_count}' + '\n' + f'Fraudulent Transactions: {frauds_count}'

        if frauds_count != 0:
            pyvis_net_community.add_edge(int(source), int(destination),label =edge_info,color='red')

        else:
            pyvis_net_community.add_edge(int(source), int(destination),label =edge_info,color='green')

    options = {
      "physics": {
        "forceAtlas2Based": {
          "springLength": 0,
          "springConstant": 0
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    pyvis_net_community.options = options
    pyvis_net_community.save_graph(f'./static/Louvain_Community_Graph.html')

    return "http://10.2.0.20:8888/static/Louvain_Community_Graph.html" #for onprem server

def customer_level_graph(louv_result,community_id,customer_id):
    louvain_result_df_filtered = louvain_result_df[(louvain_result_df['orig_id']==int(customer_id))&(louvain_result_df['louvain_communities']==community_id)]

    large= pd.merge(louvain_result_df_filtered,louv_result[louv_result['isfraud']==0][['cont_id','louvain_communities']],left_on='dest_id',right_on='cont_id',how='inner')
    large.rename(columns={'louvain_communities_x':'orig_louv_comm','louvain_communities_y':'dest_louv_comm'},inplace = True)
    large.drop(['cont_id_y'],axis=1,inplace=True)

    large['color_orig'] = large.orig_louv_comm.map(color_mapping)
    large['color_dest'] = large.dest_louv_comm.map(color_mapping)

    large['visited']=0

    dest_node_cust = large['dest_id']
    dest_node_cust = list(map(str,dest_node_cust))

    pyvis_louvain_net_customer = Network('500px','100%',filter_menu=True,directed=True)
    print('dataset, ',large[:10])

    color_value = large.loc[large['orig_id'] == int(customer_id),'color_orig'].iloc[0]
    print(color_value)

    pyvis_louvain_net_customer.add_node(customer_id,label=customer_id,size=30,color=color_value)

    for dest_node in tqdm(set(dest_node_cust)):
        color_value = large.loc[large['dest_id'] == int(dest_node),'color_dest'].iloc[0]
        pyvis_louvain_net_customer.add_node(dest_node,label=dest_node,size=30,color=color_value)

    from_cust_node = customer_id
    to_cust_node = dest_node_cust

    for destination in tqdm(to_cust_node):
        tx_id_val = large.loc[(large['dest_id'] == int(destination))&(large['visited']==0),'v_id_x'].iloc[0]
        is_fraud = large.loc[(large['dest_id'] == int(destination))&(large['visited']==0),'isfraud'].iloc[0]


        large['visited'] = np.where((large['dest_id'] == int(destination))&(large['v_id_x'] == tx_id_val),1,large['visited'])

        if is_fraud == 1:
            pyvis_louvain_net_customer.add_edge(from_cust_node,destination,color='red',value=100)
            print(tx_id_val)
        else:
            pyvis_louvain_net_customer.add_edge(from_cust_node,destination,color='green')

    pyvis_louvain_net_customer.show_buttons(filter_=['physics'])

    pyvis_louvain_net_customer.save_graph(f'./static/Louvain_Customer_Graph.html')

    return "http://10.2.0.20:8888/static/Louvain_Customer_Graph.html" #for onprem server

