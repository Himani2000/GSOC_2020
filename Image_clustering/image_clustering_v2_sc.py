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

 # the function is used to calculate the features from the image 

    
def get_features(filelist): 
    filelist.sort()
    featurelist = []
    for i, imagepath in enumerate(filelist):
        print(" Status: %s / %s" %(i, len(filelist)), end="\r")
        img = image.load_img(imagepath, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = np.array(model.predict(img_data))
        featurelist.append(features.flatten())
    return featurelist

def load_features(ideology_files_path,muslim_files_path,saved=True):
    if(saved!=True):
        print("extracting the features ")
        ideology_features=get_features(ideology_files_path)
        muslim_features=get_features(muslim_files_path)

        ideology_features=np.array(ideology_features)
        muslim_features=np.array(muslim_features)
        with open('ideology.npy', 'wb') as f:
            np.save(f, np.array(ideology_features))

        with open('muslim.npy', 'wb') as f:
            np.save(f, np.array(muslim_features)) 
  
    else:
        print("loading the features ")

        ideology_features= load('ideology.npy')
        muslim_features= load('muslim.npy')

    return ideology_features,muslim_features

def cv_silhouette_scorer(estimator, X):
    print("Running the score on the estimator")
    estimator.fit(X)
    try:
        cluster_labels = estimator.labels_
    except Exception as e:
      #  print(e,estimator)
        cluster_labels=estimator.predict(X)
    num_labels = len(set(cluster_labels))
    num_samples = len(X)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return metrics.silhouette_score(X, cluster_labels)
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
    
def getEpsilon(train_data):
    neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(train_data)
    distances, indices = nbrs.kneighbors(train_data)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    y=distances
    x=list(np.arange(0,len(distances)))

    kneedle = KneeLocator(x,y,S=10, curve='convex', direction='increasing')
    epsilon=kneedle.all_elbows_y[0]
    epsilons=[]
    for i in range(3):
        epsilons.append(epsilon+i/10)
    return epsilons

def runGridSearch(estimator,params_dict,train_data):
    
    cv = [(slice(None), slice(None))]
    gs = GridSearchCV(estimator=estimator, param_grid=params_dict, scoring=cv_silhouette_scorer, cv=cv, n_jobs=-1)
    gs.fit(train_data)
    try:
        predicted_labels= gs.best_estimator_.labels_
    except:
        predicted_labels=gs.predict(train_data)
    
    
    return predicted_labels

def runModels(train_data,output_df):
   # output_df = pd.DataFrame(index=['K-Means','Agglomerative clustering','Birch','DBSCAN','Mean-shift','Optics','Gaussian-mixture'],columns=['n_clusters','ARI','MI','H','C','V','FM','A','R','P'])
 
    #n_clusters=[50,100,200]
    """
    print("K-means results")
   # params_dict={'n_clusters':n_clusters,'init':['k-means++'],'n_init':[200], 'max_iter':[1000000],'algorithm':['auto','full', 'elkan']}
    params_dict={'n_clusters':[2],'init':['k-means++'],'n_init':[200], 'max_iter':[1000000],'algorithm':['auto']}
    predicted_labels=runGridSearch(sklearn.cluster.KMeans(),params_dict,train_data)
    evaluation_Score(train_data,predicted_labels,output_df,'K-Means')
    output_df.loc['K-Means','n_clusters']=len(set(predicted_labels))
    
    print(output_df)
    """
    print("Agglomerative clustering results")
    params_dict={'affinity':['euclidean'], 'linkage':['ward','complete','average','single'],'distance_threshold':[500,1000,2000],'n_clusters':[None]}
   # params_dict={'affinity':['euclidean'], 'linkage':['ward']}
    predicted_labels=runGridSearch(sklearn.cluster.AgglomerativeClustering(),params_dict,train_data)
    evaluation_Score(train_data,predicted_labels,output_df,'Agglomerative clustering')
    output_df.loc['Agglomerative clustering','n_clusters']=len(set(predicted_labels))
    
    with open('image-results/ideology_agg_labels.npy', 'wb') as f:
        np.save(f, np.array(predicted_labels))
    print(output_df)
    
    
    """
    print("Birch")
   # params_dict={'threshold':[0.5,0.2,0.8], 'branching_factor':[50,100,200], 'n_clusters':n_clusters}
    params_dict={'threshold':[0.5], 'branching_factor':[50], 'n_clusters':[2]}
    predicted_labels=runGridSearch(sklearn.cluster.Birch(),params_dict,train_data)
    evaluation_Score(train_data,predicted_labels,output_df,'Birch')
    output_df.loc['Birch','n_clusters']=len(set(predicted_labels))
    
    print(output_df)
    """
    print("DBSCAN")
    epsilons=getEpsilon(train_data)
    params_dict = {'eps':epsilons,'min_samples':[20,30,40],'metric':['euclidean','manhattan','mahalanobis', 'minkowski']}

   # params_dict = {'eps':[0.5],'min_samples':[20]}
    predicted_labels=runGridSearch(sklearn.cluster.DBSCAN(),params_dict,train_data)
    evaluation_Score(train_data,predicted_labels,output_df,'DBSCAN')
    output_df.loc['DBSCAN','n_clusters']=len(set(predicted_labels))
    print(output_df)
    with open('image-results/ideology_dbscan_labels.npy', 'wb') as f:
        np.save(f, np.array(predicted_labels))
    
    print("Mean shift")
    quantiles=[0.2,0.5,0.8,1]
    #quantiles=[0.2]
    params_dict={}
    params_dict['bandwidth']=[]
    for quantile in quantiles:
        params_dict['bandwidth'].append(sklearn.cluster.estimate_bandwidth(train_data, quantile=quantile, n_samples=500))

    predicted_labels=runGridSearch(sklearn.cluster.MeanShift(),params_dict,train_data)
    evaluation_Score(train_data,predicted_labels,output_df,'Mean-shift')
    output_df.loc['Mean-shift','n_clusters']=len(set(predicted_labels))
    
    
    print(output_df)
    with open('image-results/ideology_mean_shift_labels.npy', 'wb') as f:
        np.save(f, np.array(predicted_labels))
            
    print("Optics")
   # params_dict={'eps':[0.5],'min_samples':[20]}
    params_dict = {'eps':[0.5,0.6,0.8],'min_samples':[20,30,40],'metric':['euclidean','manhattan','mahalanobis', 'minkowski']}
    predicted_labels=runGridSearch(sklearn.cluster.OPTICS(),params_dict,train_data)
    evaluation_Score(train_data,predicted_labels,output_df,'Optics')
    output_df.loc['Optics','n_clusters']=len(set(predicted_labels))
    
    print(output_df)
    
    with open('image-results/ideology_optics_labels.npy', 'wb') as f:
            np.save(f, np.array(predicted_labels))
    """
    print("Gaussian Mixture")
    params_dict={'covariance_type':['full'], 'max_iter':[100],'n_components':[2]}
   # params_dict={'covariance_type':['full','tied','diag','spherical'], 'max_iter':[1000000,1000,10000],'n_components':n_clusters}
    predicted_labels=runGridSearch(sklearn.mixture.GaussianMixture(),params_dict,train_data)
    evaluation_Score(train_data,predicted_labels,output_df,'Gaussian-mixture')
    output_df.loc['Gaussian-mixture','n_clusters']=len(set(predicted_labels))
    
    print(output_df)
    
    #print("Spectral")
    #yhat,clusters=spectral_model(train_data,2)
    #output_df=pred_cluster_label(yhat,clusters,cluster_df,output_df,'Spectral-clustering')
    """
    
    
    
    return output_df


def runClustering(train_data,dimensionality=None):
   
    
    output_df = pd.DataFrame(index=['Agglomerative clustering','DBSCAN','Mean-shift','Optics'],columns=['n_clusters','silhouette','calinski','davies'])
    
    if(dimensionality==None):
        output_df=runModels(train_data,output_df)
    
    elif(dimensionality=='pca'):
        #train_data_transform=pca_transform(train_data)
        output_df=runModels(train_data_transform,output_df)
    
    elif(dimensionality=='tsne'):
        #train_data_transform=tsne_transform(train_data)
        output_df=runModels(train_data_transform,output_df)
   
    return output_df
        
if __name__ == "__main__":
    
    ideology_files_path,muslim_files_path=load_filepaths()
    
    
    ideology_features,muslim_features=load_features(ideology_files_path,muslim_files_path,True)
    print(ideology_features.shape,muslim_features.shape)
    print("Running the models")
    output_df=runClustering(ideology_features)
    print("saving the model")
    output_df.to_csv('image-results/ideology_image_clustering_Results.csv')
        
    
    
   

