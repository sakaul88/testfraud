import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import json
import mpld3
import base64
import seaborn as sns
import numpy as np

def plot_by_category(list1,list2):
    
    fig1, ax1 = plt.subplots(figsize = (10, 5))
    cmap1 = plt.cm.Blues
    cmap2 = plt.cm.Reds
    cmap3 = plt.cm.Greens
    cmap4 = plt.cm.Purples
    outer_colors = [cmap1(.4), cmap2(.4), cmap3(.4),cmap4(.4)]
    ax1.pie(list1, labels=list2, autopct='%1.1f%%',
        shadow=True, startangle=90, colors = outer_colors,wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'black' })
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig("./static/fig_category.png",
                    bbox_inches ="tight"
                    )         
    pic1 = json.dumps({"my_figure_1": "http://10.2.0.20:8888/static/fig_category.png"}) #onprem
#   pic1 = json.dumps({"my_figure_1": "http://20.58.58.175:443/static/fig_category.png"}) #azure
    # pic1 = json.dumps({"my_figure_1": "http://192.168.25.46:8888/static/fig_category.png"}) #local
#   pic1 = json.dumps({"my_figure_1": "http://172.30.168.28:8888/static/fig_category.png"}) #aws
    pic2 = json.loads(pic1)
    return pic2
    

def plot_by_date(list1,list2):
    fig = plt.figure(figsize = (10, 5))
    list2 = [ round(elem, 2) for elem in list2 ]
    if len(list1) <= 10:
        sns.relplot(
        x=list1, y=list2,
        kind="line",
        label = list2,
        linewidth=5,
        height=6, aspect=2)
       
    elif len(list1) > 10:
        list1 = list1[-10:]
        list2 = list2[-10:]
        sns.relplot(  
        x=list1, y=list2,
        kind="line",
        label = list2,
        linewidth=5,
        height=6, aspect=2 )

    plt.grid()   
    plt.xlabel("Date")
    plt.ylabel("Fraud Percentage (%)")
    for i,j in zip(list1,list2):
        if j>0.0:
            plt.annotate(str(j), xy=(i, j))
        elif j == 0.0:
            plt.annotate(str(j), xy=(i, j-2))

    plt.savefig("./static/fig_date.png",
                    bbox_inches ="tight"
                    )          
    pic3 = json.dumps({"my_figure_2": "http://10.2.0.20:8888/static/fig_date.png"}) #onprem
    # pic3 = json.dumps({"my_figure_2": "http://20.58.58.175:443/static/fig_date.png"}) #azure
    # pic3 = json.dumps({"my_figure_2": "http://192.168.25.46:8888/static/fig_date.png"}) #local
#     pic3 = json.dumps({"my_figure_2": "http://172.30.168.28:8888/static/fig_date.png"}) #aws
    pic4 = json.loads(pic3)
    return pic4


def plot_by_merchant(list1,list2):
    fig = plt.figure(figsize = (10, 5))
    print(list1)
    print(list2)
    c = plt.cm.Blues
    plt.barh(list1,list2, color=c(.4))
    plt.xlabel("Fraud Percentage (%)")
    plt.ylabel("Merchant Name")
    plt.savefig("./static/fig_merchant_code.png",
                    bbox_inches ="tight"
                    )             

    pic9 = json.dumps({"my_figure_4": "http://10.2.0.20:8888/static/fig_merchant_code.png"}) #onprem
    # pic9 = json.dumps({"my_figure_4": "http://20.58.58.175:443/static/fig_merchant_code.png"}) #azure
    # pic9 = json.dumps({"my_figure_4": "http://192.168.25.46:8888/static/fig_merchant_code.png"}) #local
#     pic9 = json.dumps({"my_figure_4": "http://172.30.168.28:8888/static/fig_merchant_code.png"}) #aws
    pic10 = json.loads(pic9)
    return pic10        
  

def plot_confusion_matrix(conf_matrix):
    figsize=(9,4)
    fig, ax = plt.subplots(figsize=figsize)
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                conf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                     conf_matrix.flatten()/np.sum(conf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(conf_matrix,
                  annot_kws={"size": 13},
                  fmt='', annot=labels, cmap='Blues')
    plt.savefig("./static/cf.png",
                    bbox_inches ="tight"
                    )        
    
    # pic7 = json.dumps({"my_cm_figure": "http://192.168.43.42:8888/static/cf.png"})
    pic7 = json.dumps({"my_cm_figure": "http://10.2.0.20:8888/static/cf.png"}) #onprem 
    # pic7 = json.dumps({"my_cm_figure": "http://20.58.58.175:443/static/cf.png"}) #azure
    # pic7 = json.dumps({"my_cm_figure": "http://192.168.25.46:8888/static/cf.png"}) #local
    # pic7 = json.dumps({"my_cm_figure": "http://172.30.168.28:8888/static/cf.png"}) #aws
    pic8 = json.loads(pic7)
    return pic8   