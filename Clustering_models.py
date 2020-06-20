#!/usr/bin/env python
# coding: utf-8

# In[9]:


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
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def get_muslim_labels(filename):
    df=pd.read_csv(filename)
    final_files=list(df['file_name'])
    
    df=pd.read_excel("spreadsheet_data/muslim_concordance_250_annotated.xls")
    df.dropna(inplace=True)
    
    
    final_updated_files=[]
    for file in final_files:
        for i in range(len(df)):
            if(df.loc[i,'File Name']==file):
                if(df['u'][i]==1):
                    final_updated_files.append('u')
    
                elif(df['a'][i]==1):
                    final_updated_files.append('a')
    
                elif(df['misaligned/error/etc.'][i]==1):
                    final_updated_files.append('misaligned')
    
                elif(df['some other problem'][i]==1):
                    final_updated_files.append('other')
    
                elif(df['can\'t decide'][i]==1):
                    final_updated_files.append('cantdecide')
    
    
    le = LabelEncoder()
    le.fit(np.array(final_updated_files))

    encoded_labels=le.transform(np.array(final_updated_files))
    print("classes",le.classes_)
    return encoded_labels


# In[ ]:


def load_dataset(filename,dataset_name):
    df=pd.read_csv(filename)
    train_data=df.iloc[:,0:-1]
    scaler =MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    train_data=pd.DataFrame(train_data)
    train_data=train_data.fillna(0)

    if(dataset_name=='ideology'):
        splited_df=df['file_name'].str.split("clip_",expand=True)
        labels=splited_df[1].str.split("_",expand=True)
        actual_labels=labels[1]

        le = LabelEncoder()
        le.fit(np.array(actual_labels))
        true_labels=le.transform(np.array(actual_labels))
        
    elif(dataset_name=='muslim'):
        true_labels=get_muslim_labels(filename)
        

   
    train_data=train_data.fillna(0)
    return train_data,true_labels


# In[ ]:


def cv_silhouette_scorer(estimator, X):
    estimator.fit(X)
    try:
        cluster_labels = estimator.labels_
    except Exception as e:
      #  print(e,estimator)
        cluster_labels=estimator.predict(X)
    num_labels = len(set(cluster_labels))
    num_samples = len(X.index)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return metrics.silhouette_score(X, cluster_labels)


# In[ ]:


def evaluation_Score(y_true,y_pred,output_df,model):
    try:
        output_df.loc[model,'ARI'] =metrics.adjusted_rand_score(y_true,y_pred)
        output_df.loc[model,'MI'] = metrics.adjusted_mutual_info_score(y_true,y_pred)
        output_df.loc[model,'H'] = metrics.homogeneity_score(y_true,y_pred)
        output_df.loc[model,'C'] = metrics.completeness_score(y_true,y_pred)
        output_df.loc[model,'V'] = metrics.v_measure_score(y_true,y_pred)
        output_df.loc[model,'FM'] =metrics.fowlkes_mallows_score(y_true,y_pred)
        output_df.loc[model,'A']=metrics.accuracy_score(y_true, y_pred)
        output_df.loc[model,'R']=metrics.recall_score(y_true,y_pred)
        output_df.loc[model,'P']=metrics.precision_score(y_true,y_pred)

    except ValueError as e:
        output_df.loc[model,'R']=metrics.recall_score(y_true,y_pred,average='micro')
        output_df.loc[model,'P']=metrics.precision_score(y_true,y_pred,average='micro')

        
    
    return output_df
    
    
    
    
   
    


# In[ ]:


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


# In[ ]:


def runGridSearch(estimator,params_dict,train_data):
    
    cv = [(slice(None), slice(None))]
    gs = GridSearchCV(estimator=estimator, param_grid=params_dict, scoring=cv_silhouette_scorer, cv=cv, n_jobs=-1)
    gs.fit(train_data)
    try:
        predicted_labels= gs.best_estimator_.labels_
    except:
        predicted_labels=gs.predict(train_data)
    
    
    return predicted_labels


# In[2]:


def runModels(train_data,true_labels,output_df,dataset_name):
   # output_df = pd.DataFrame(index=['K-Means','Agglomerative clustering','Birch','DBSCAN','Mean-shift','Optics','Gaussian-mixture'],columns=['n_clusters','ARI','MI','H','C','V','FM','A','R','P'])
    if(dataset_name=='ideology'):
        n_clusters=[2]
    elif(dataset_name=='muslim'):
        n_clusters=[5]
    
    print("K-means results")
    params_dict={'n_clusters':n_clusters,'init':['k-means++'],'n_init':[200], 'max_iter':[1000000],'algorithm':['auto','full', 'elkan']}
    #params_dict={'n_clusters':[2],'init':['k-means++'],'n_init':[200], 'max_iter':[1000000],'algorithm':['auto']}
    predicted_labels=runGridSearch(sklearn.cluster.KMeans(),params_dict,train_data)
    evaluation_Score(true_labels,predicted_labels,output_df,'K-Means')
    output_df.loc['K-Means','n_clusters']=len(set(predicted_labels))
    
    
    
    print("Agglomerative clustering results")
    params_dict={'affinity':['euclidean'], 'linkage':['ward','complete','average','single']}
    #params_dict={'affinity':['euclidean'], 'linkage':['ward']}
    predicted_labels=runGridSearch(sklearn.cluster.AgglomerativeClustering(),params_dict,train_data)
    evaluation_Score(true_labels,predicted_labels,output_df,'Agglomerative clustering')
    output_df.loc['Agglomerative clustering','n_clusters']=len(set(predicted_labels))
    
    
    
    
    print("Birch")
    params_dict={'threshold':[0.5,0.2,0.8], 'branching_factor':[50,100,200], 'n_clusters':n_clusters}
   # params_dict={'threshold':[0.5], 'branching_factor':[50], 'n_clusters':[2]}
    predicted_labels=runGridSearch(sklearn.cluster.Birch(),params_dict,train_data)
    evaluation_Score(true_labels,predicted_labels,output_df,'Birch')
    output_df.loc['Birch','n_clusters']=len(set(predicted_labels))
    
    
   
    print("DBSCAN")
    epsilons=getEpsilon(train_data)
    params_dict = {'eps':epsilons,'min_samples':[20,30,40],'metric':['euclidean','manhattan','mahalanobis', 'minkowski']}

   # params_dict = {'eps':[0.5],'min_samples':[20]}
    predicted_labels=runGridSearch(sklearn.cluster.DBSCAN(),params_dict,train_data)
    evaluation_Score(true_labels,predicted_labels,output_df,'DBSCAN')
    output_df.loc['DBSCAN','n_clusters']=len(set(predicted_labels))
    
    
    print("Mean shift")
    quantiles=[0.2,0.5,0.8,1]
    params_dict={}
    params_dict['bandwidth']=[]
    for quantile in quantiles:
        params_dict['bandwidth'].append(sklearn.cluster.estimate_bandwidth(train_data, quantile=quantile, n_samples=500))

    predicted_labels=runGridSearch(sklearn.cluster.MeanShift(),params_dict,train_data)
    evaluation_Score(true_labels,predicted_labels,output_df,'Mean-shift')
    output_df.loc['Mean-shift','n_clusters']=len(set(predicted_labels))
    
    
    
    print("Optics")
   # params_dict={'eps':[0.5],'min_samples':[20]}
    params_dict = {'eps':[0.5,0.6,0.8],'min_samples':[20,30,40],'metric':['euclidean','manhattan','mahalanobis', 'minkowski']}
    predicted_labels=runGridSearch(sklearn.cluster.OPTICS(),params_dict,train_data)
    evaluation_Score(true_labels,predicted_labels,output_df,'Optics')
    output_df.loc['Optics','n_clusters']=len(set(predicted_labels))
    
    
    print("Gaussian Mixture")
    #params_dict={'covariance_type':['full'], 'max_iter':[100],'n_components':[2]}
    params_dict={'covariance_type':['full','tied','diag','spherical'], 'max_iter':[1000000,1000,10000],'n_components':n_clusters}
    predicted_labels=runGridSearch(sklearn.mixture.GaussianMixture(),params_dict,train_data)
    evaluation_Score(true_labels,predicted_labels,output_df,'Gaussian-mixture')
    output_df.loc['Gaussian-mixture','n_clusters']=len(set(predicted_labels))
    
    
    
    #print("Spectral")
    #yhat,clusters=spectral_model(train_data,2)
    #output_df=pred_cluster_label(yhat,clusters,cluster_df,output_df,'Spectral-clustering')
    
    
    
    
    return output_df



# In[3]:


def pca_transform(train_data):
    pca = PCA(n_components = 2) 
    X_principal = pca.fit_transform(train_data) 
    X_principal = pd.DataFrame(X_principal) 
    X_principal.columns = ['P1', 'P2'] 
    return X_principal
    

def tsne_transform(train_data):
    tsne = TSNE(n_components=2)
    X_principal=tsne.fit_transform(train_data)
    X_principal = pd.DataFrame(X_principal) 
    X_principal.columns = ['P1', 'P2'] 
    return X_principal


# In[4]:


def runClustering(filename,dataset_name,dimensionality=None):
    train_data,true_labels=load_dataset(filename,dataset_name)
    
    
    output_df = pd.DataFrame(index=['K-Means','Agglomerative clustering','Birch','DBSCAN','Mean-shift','Optics','Gaussian-mixture'],columns=['n_clusters','ARI','MI','H','C','V','FM','A','R','P'])
    
    if(dimensionality==None):
        output_df=runModels(train_data,true_labels,output_df,dataset_name)
    
    elif(dimensionality=='pca'):
        train_data_transform=pca_transform(train_data)
        output_df=runModels(train_data_transform,true_labels,output_df,dataset_name)
    
    elif(dimensionality=='tsne'):
        train_data_transform=tsne_transform(train_data)
        output_df=runModels(train_data_transform,true_labels,output_df,dataset_name)
   
    return output_df
        
    


# In[6]:


if __name__=='__main__':
    files=list(os.listdir('Audio_features/'))
    for file in files:
        splited_file=file.split('_')
        if(splited_file[0]!='mfcc0'):
            print(file)
            output_df=runClustering(os.path.join('Audio_features',file),splited_file[2],None)
            
            filename="_".join(splited_file[0:-1])+'.csv'
            output_df.to_csv(os.path.join('results',filename))
            
            
            
            
            
            """
           
            filename=("_".join(splited_file[0:-1])+'pca'+'.csv'
            print(filename)
            output_df.to_csv(os.path.join('results',filename),index=False)
                      
           # output_df=runClustering(os.path.join('Audio_features',file),splited_file[2],'tsne')
            filename=("_".join(splited_file[0:-1])+'tsne'+'.csv'
            print(filename)
            output_df.to_csv(os.path.join('results',filename),index=False)
            
            """
        


# In[ ]:





# In[ ]:




