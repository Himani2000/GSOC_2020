#importing the header files 
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
import os, shutil, glob, os.path
from PIL import Image as pil_image
from matplotlib import pyplot as plt
import hdbscan
from scipy.spatial import distance
from scipy.cluster import hierarchy
#import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from numpy import load,save


def load_filepaths():
    imdir_ideology = 'ideology_image_dataset/'
    imdir_muslim='muslim_image_dataset/'
    ideology_files=os.listdir('ideology_image_dataset/')
    muslim_files=os.listdir('muslim_image_dataset/')
    len(ideology_files),len(muslim_files)

    ideology_files_path=[os.path.join(imdir_ideology,file) for file in ideology_files ]
    muslim_files_path=[os.path.join(imdir_muslim,file) for file in muslim_files]
    return ideology_files_path,muslim_files_path

def loadFeatures(filename):
    print("Loading file : ",filename)
    features= load(filename)
    return features




def evaluation_Score(features,y_pred,output_df,model):
    try:
        
        num_labels=len(set(y_pred))
        total_samples=len(y_pred)
        if(num_labels==1 or num_labels==total_samples):
            output_df.loc[model,'silhouette'] =-1
            output_df.loc[model,'calinski'] =-1
            output_df.loc[model,'davies'] =-1
            
        else:
            output_df.loc[model,'silhouette'] =metrics.silhouette_score(features,y_pred)
            output_df.loc[model,'calinski'] =metrics.calinski_harabasz_score(features, y_pred)
            output_df.loc[model,'davies'] =metrics.davies_bouldin_score(features,y_pred)
            
    

    except Exception as e:
        print(e)
        pass
        

        
    
    return output_df
    
    
    




def runModels(train_data,output_df,filename):
    
    print("Agglomerative2-scipy clustering results")
    sim=0.5
    timestamps=None
    alpha=0
    method='average'
    metric='euclidean'
    extra_out=False
    print_stats=True
    min_csize=2
    dfps = distance.pdist(np.array(list(train_data)), metric)
    Z = hierarchy.linkage(dfps, method=method, metric=metric)
    predicted_labels = hierarchy.fcluster(Z, t=dfps.max()*(1.0-sim), criterion='distance')
    if(set(predicted_labels).issuperset({-1})):
        n_clustersLen=len(set(predicted_labels))-1
    else:
        n_clustersLen=len(set(predicted_labels))
    
    
    evaluation_Score(train_data,predicted_labels,output_df,'Agglomerative clustering-scipy')
    output_df.loc['Agglomerative clustering-scipy','n_clusters']=n_clustersLen
          
    saved_filename=os.path.join('image-results2',filename)
    np.save(saved_filename+'_agg-scipy_labels.npy', np.array(predicted_labels))
   
    print(output_df)
    print("HDBSCAN")
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20,cluster_selection_epsilon= 0.01,min_samples= 1)
    predicted_labels = clusterer.fit_predict(features)
    if(set(predicted_labels).issuperset({-1})):
        n_clustersLen=len(set(predicted_labels))-1
    else:
        n_clustersLen=len(set(predicted_labels))
        
    evaluation_Score(train_data,predicted_labels,output_df,'HDBSCAN')
    output_df.loc['HDBSCAN','n_clusters']=n_clustersLen
          
    saved_filename=os.path.join('image-results2',filename)
    np.save(saved_filename+'_hdbscan_labels.npy', np.array(predicted_labels))
    print(output_df)
    
        
    print("Agglomerative clustering results")
  #  params_dict={'linkage':['ward','complete','average','single'],'distance_threshold':[500,1000,2000],'n_clusters':[None]}
  #  predicted_labels=runGridSearch(sklearn.cluster.AgglomerativeClustering(),params_dict,train_data)
    predicted_labels=sklearn.cluster.AgglomerativeClustering().fit(train_data).labels_
    evaluation_Score(train_data,predicted_labels,output_df,'Agglomerative clustering')
    if(set(predicted_labels).issuperset({-1})):
        n_clustersLen=len(set(predicted_labels))-1
    else:
        n_clustersLen=len(set(predicted_labels))
    output_df.loc['Agglomerative clustering','n_clusters']=n_clustersLen
    
    saved_filename=os.path.join('image-results2',filename)
    np.save(saved_filename+'_agg_labels.npy', np.array(predicted_labels))
    #print(output_df)
    
    
    
    print("DBSCAN")
    #epsilons=getEpsilon(train_data)
    #params_dict = {'eps':epsilons,'min_samples':[20,30,40],'metric':['euclidean','manhattan','mahalanobis', 'minkowski']}
    predicted_labels=sklearn.cluster.DBSCAN().fit(train_data).labels_
    evaluation_Score(train_data,predicted_labels,output_df,'DBSCAN')
    if(set(predicted_labels).issuperset({-1})):
        n_clustersLen=len(set(predicted_labels))-1
    else:
        n_clustersLen=len(set(predicted_labels))
    output_df.loc['DBSCAN','n_clusters']=n_clustersLen
    print(output_df)
        
    saved_filename=os.path.join('image-results2',filename)
    np.save(saved_filename+'_dbscan_labels.npy', np.array(predicted_labels))
    
    print("Mean shift")
    #quantiles=[0.2,0.5,0.8,1]
    #params_dict={}
    #params_dict['bandwidth']=[]
    #for quantile in quantiles:
    #    params_dict['bandwidth'].append(sklearn.cluster.estimate_bandwidth(train_data, quantile=quantile, n_samples=500))

    predicted_labels=sklearn.cluster.MeanShift().fit(train_data).labels_
    evaluation_Score(train_data,predicted_labels,output_df,'Mean-shift')
    if(set(predicted_labels).issuperset({-1})):
        n_clustersLen=len(set(predicted_labels))-1
    else:
        n_clustersLen=len(set(predicted_labels))
    output_df.loc['Mean-shift','n_clusters']=n_clustersLen
    
    
    print(output_df)
        
    saved_filename=os.path.join('image-results2',filename)
    np.save(saved_filename+'_mean_shift_labels.npy', np.array(predicted_labels))
            
    print("Optics")
   # epsilons=getEpsilon(train_data)
   # params_dict = {'eps':epsilons,'min_samples':[20,30,40],'metric':['euclidean','manhattan','mahalanobis', 'minkowski']}
    predicted_labels=sklearn.cluster.OPTICS().fit(train_data).labels_
    if(set(predicted_labels).issuperset({-1})):
        n_clustersLen=len(set(predicted_labels))-1
    else:
        n_clustersLen=len(set(predicted_labels))
    evaluation_Score(train_data,predicted_labels,output_df,'Optics')
    output_df.loc['Optics','n_clusters']=n_clustersLen
    
    print(output_df)
    
        
    saved_filename=os.path.join('image-results2',filename)
    np.save(saved_filename+'_optics_labels.npy', np.array(predicted_labels))
    
   
    

    
    
    
    return output_df





def runClustering(train_data,filename,dimensionality=None):
   
    
    output_df = pd.DataFrame(index=['Agglomerative clustering','DBSCAN','Mean-shift','Optics'],columns=['n_clusters','silhouette','calinski','davies'])
    
    if(dimensionality==None):
        output_df=runModels(train_data,output_df,filename)
    
    elif(dimensionality=='pca'):
        #train_data_transform=pca_transform(train_data)
        output_df=runModels(train_data_transform,output_df,filename)
    
    elif(dimensionality=='tsne'):
        #train_data_transform=tsne_transform(train_data)
        output_df=runModels(train_data_transform,output_df,filename)
   
    return output_df
        
    

if __name__ == "__main__":
    
    ideology_files_path,muslim_files_path=load_filepaths()
    
    
   # ideology_features,muslim_features=load_features(ideology_files_path,muslim_files_path,True)
    for file in os.listdir('Image_features/'):
        if file.endswith('.npy'):
           # if 'fcn.npy' in set(file.split('_')):
                print(file)
                features=loadFeatures(os.path.join('Image_features',file))
                output_df=runClustering(features,file)
                print("saving the model")
                saved_csv_filename=os.path.join('image-results2',file)
                output_df.to_csv(saved_csv_filename+'.csv')
        
