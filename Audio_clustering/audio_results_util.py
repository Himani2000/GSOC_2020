import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler,StandardScaler,normalize
import matplotlib.pyplot as plt 
#importing the header files 
from collections import Counter
import os
import numpy as np
import pandas as pd
import IPython.display as ipd
from sklearn.manifold import TSNE
import time
import sklearn
from sklearn.decomposition import PCA 
import librosa
from sklearn import mixture
from numpy import unique
from numpy import where
from matplotlib import pyplot as plt
import librosa.display
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.model_selection import GridSearchCV
from kneed import DataGenerator, KneeLocator
import plotly.express as px
from collections import Counter 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import umap

def load_dataset(filename):
    train_data=pd.read_csv(filename)
    true_labels=train_data['true_labels']
    true_encoded_labels=train_data['true_encoded_labels']
    train_data=train_data.iloc[:,0:-3]
    
    scaler =MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    train_data=pd.DataFrame(train_data)
    train_data=train_data.fillna(0)
   
    train_data=train_data.fillna(0)
    return train_data,true_labels,true_encoded_labels

def plotData3d(train_data,actual_labels):
    actual_labels=list(actual_labels)
    x = np.array(train_data.iloc[:,0])
    y = np.array(train_data.iloc[:,1])
    z = np.array(train_data.iloc[:,2])
    color_discrete_map = {'ai': 'rgb(255,0,0)', 'ee': 'rgb(0,255,0)'}
    color_discrete_map={'0': 'rgb(255,0,0)', '1': 'rgb(0,255,0)'}
    fig = px.scatter_3d(train_data,x, y, z,color=actual_labels,opacity=0.5,color_discrete_map=color_discrete_map)
    fig.show()
    
def plotData2d(train_data,actual_labels):
    actual_labels=list(actual_labels)
    x = np.array(train_data.iloc[:,0])
    y = np.array(train_data.iloc[:,1])
    color_discrete_map = {'ai': 'red', 'ee': 'rgb(0,255,0)'}
    color_discrete_map={'0': 'rgb(255,0,0)', '1': 'rgb(0,255,0)'}
    fig = px.scatter(train_data,x, y,color=actual_labels,opacity=0.5,color_discrete_map=color_discrete_map)
    fig.show()
    
def encodedLabels(labels):
    encoded_list=[]
    for label in labels:
      #  if(label==0):
       #     encoded_list.append('ai')
       # elif(label==1):
        #    encoded_list.append('ee')

        #else:
            encoded_list.append(str(label))
    return encoded_list


def plotClusters(train_data,filename):
    labels=np.load(filename)
    
   
    if(len(set(labels))>20):
        print("Too many labels to show")
    else:
        print("predicted_labels-->",Counter(labels))
        encoded_list=encodedLabels(labels)

        print(" 2D representation")

        plotData2d(train_data,encoded_list)
        print(" 3D representation")
        plotData3d(train_data,encoded_list)

    
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)


def plotAllClusterModels(train_data,filename):
    print(f'{filename}_kmeans_labels.npy')
    print("K-Means")
    plotClusters(train_data,f'{filename}_kmeans_labels.npy')
    
    print("Agglomerative")
    plotClusters(train_data,f'{filename}_agg_labels.npy')
    
    print("Birch")
    plotClusters(train_data,f'{filename}_birch_labels.npy')
    
    print("DBSCAN")
    plotClusters(train_data,f'{filename}_dbscan_labels.npy')
    
    print("OPTICS")
    plotClusters(train_data,f'{filename}_optics_labels.npy')
    
    print("MEAN-SHIFT")
    plotClusters(train_data,f'{filename}_mean_shift_labels.npy')
    
    print("Gaussian-Mixture")
    plotClusters(train_data,f'{filename}_gaussian_labels.npy')
    
    
def actualDistribution(train_data,true_labels,true_encoded_labels):
    print("Actual Distribution of the labels in the dataset: --> ")
    print(Counter(true_labels))
    print(Counter(true_encoded_labels))

    plotData2d(train_data,encodedLabels(true_encoded_labels))
    plotData3d(train_data,encodedLabels(true_encoded_labels))
    
    