#importing the python standard libraries 
import pandas as pd 
import numpy as np
import os 
import hdbscan
import sklearn
from sklearn.cluster import DBSCAN,OPTICS,MeanShift

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from kneed import DataGenerator, KneeLocator
import umap
import argparse
#importing the python files 
import clustering_hyperparameter
import image_utils


    
def getEpsilon(train_data):
    neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=4)
    nbrs = neigh.fit(train_data)
    distances, indices = nbrs.kneighbors(train_data)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    y=distances
    x=list(np.arange(0,len(distances)))
    sensitivity = [1,3, 5, 10, 20,40,60,80, 100, 120,150,180,200,250,300,350,400]
    epsilons=[]
    for s in sensitivity:
        try:
            kneedle = KneeLocator(x,y,S=s, curve='convex', direction='increasing')
            epsilon=kneedle.all_elbows_y[0]
            if(len(epsilons)>=1 and epsilons[-1]-epsilon<=0.001):
                print("")
                
            else:    
                epsilons.append(epsilon)
        
        except Exception as e:
            print(e)
            if(len(epsilons)>=1):
                epsilons.append(epsilons[-1]+s/10)
            else:
                epsilons.append(s/10)
    
   # epsilons.append(0.3)
   # epsilons.append(0.5)
   # epsilons.append(0.8)
        
    print(epsilons)
 
    return epsilons




def runDensityModels(train_data,output_df,filename):
   
    print("HDBSCAN")
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2,cluster_selection_epsilon= 0.01,min_samples= 1)
    predicted_labels = clusterer.fit_predict(features)
    if(set(predicted_labels).issuperset({-1})):
        n_clustersLen=len(set(predicted_labels))-1
    else:
        n_clustersLen=len(set(predicted_labels))
        
    clustering_hyperparameter.evaluation_Score(train_data,predicted_labels,output_df,'HDBSCAN')
    output_df.loc['HDBSCAN','n_clusters']=n_clustersLen
          
    
    np.save(filename+'_hdbscan_labels.npy', np.array(predicted_labels))
    print(output_df)
    
   
    
    print("DBSCAN")
    epsilons=getEpsilon(train_data)
    params_dict = {'eps':epsilons,'min_samples':[2,3,4],'metric':['euclidean','manhattan','mahalanobis']}
    predicted_labels=clustering_hyperparameter.runGridSearch(DBSCAN(),params_dict,train_data)
    
    clustering_hyperparameter.evaluation_Score(train_data,predicted_labels,output_df,'DBSCAN')
    if(set(predicted_labels).issuperset({-1})):
        n_clustersLen=len(set(predicted_labels))-1
    else:
        n_clustersLen=len(set(predicted_labels))
    output_df.loc['DBSCAN','n_clusters']=n_clustersLen
    print(output_df)
    np.save(filename+'_dbscan_labels.npy', np.array(predicted_labels))
    
    
    print("Mean shift")
    quantiles=[0.2,0.5,0.8,1]
    params_dict={}
    params_dict['bandwidth']=[]
    for quantile in quantiles:
        params_dict['bandwidth'].append(sklearn.cluster.estimate_bandwidth(train_data, quantile=quantile, n_samples=500))

    params_dict['bandwidth'].append(0.2)
    predicted_labels=clustering_hyperparameter.runGridSearch(MeanShift(),params_dict,train_data)
    clustering_hyperparameter.evaluation_Score(train_data,predicted_labels,output_df,'Mean-shift')
    if(set(predicted_labels).issuperset({-1})):
        n_clustersLen=len(set(predicted_labels))-1
    else:
        n_clustersLen=len(set(predicted_labels))
    output_df.loc['Mean-shift','n_clusters']=n_clustersLen
    
    np.save(filename+'_mean_shift_labels.npy', np.array(predicted_labels))
            
    print("Optics")

    params_dict = {'min_samples':[3,4,5],'metric':['euclidean','manhattan','mahalanobis']}
    predicted_labels=clustering_hyperparameter.runGridSearch(OPTICS(),params_dict,train_data)
    if(set(predicted_labels).issuperset({-1})):
        n_clustersLen=len(set(predicted_labels))-1
    else:
        n_clustersLen=len(set(predicted_labels))
    clustering_hyperparameter.evaluation_Score(train_data,predicted_labels,output_df,'Optics')
    output_df.loc['Optics','n_clusters']=n_clustersLen
    
    print(output_df)
    
    np.save(filename+'_optics_labels.npy', np.array(predicted_labels))
            
    
    
    

    
   
    return output_df


def runDensityClustering(train_data,filename):
   
    output_df = pd.DataFrame(index=['HDBSCAN','DBSCAN','Mean-shift','Optics'],columns=['n_clusters','silhouette','calinski'])
   
    output_df=runDensityModels(train_data,output_df,filename)
    
 
   
    return output_df
        
    
    
if __name__ == "__main__":
    
    #image_directory from argparse 
    #feature directory from the argparse 
    #results saving directory from the argparse 
    
    
    my_parser = argparse.ArgumentParser(description='List the content of a folder')

    my_parser.add_argument('image_directory',
                           metavar='path',
                           type=str,
                           help='the path to the image directory')
    my_parser.add_argument('feature_directory',
                           metavar='path',
                           type=str,
                           help='the path to image features ')

    
    my_parser.add_argument('results_directory',
                           metavar='path',
                           type=str,
                           help='the path to save the image results ')

    args = my_parser.parse_args()

    image_directory=args.image_directory
    
    feature_directory=args.feature_directory
    results_directory=args.results_directory
    
    
    
    files_path=image_utils.loadFilePaths(image_directory)
    print("Total files_path",files_path[0])
    print(image_directory,feature_directory,results_directory)
    
    for file in list(os.listdir(feature_directory)):
        if file.endswith('.npy'):
            print(f'{feature_directory}/{file}')
            features=image_utils.loadFeatures(os.path.join(feature_directory,file))
            print("tsne....")
            tsne_transformed=TSNE(n_components=3, n_jobs=-1).fit_transform(features)
            output_df=runDensityClustering(tsne_transformed,f'{results_directory}/{file}-density-tsne')
            output_df.to_csv(f'{results_directory}/{file}-density-tsne.csv')
            
            
            
            print("pca....")
            pca_dims = PCA().fit(features)
            cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
            d = np.argmax(cumsum >= 0.95) + 1
            print(d)
            if(d==1):
                d=d+1
            pca_transformed=PCA(n_components=d).fit_transform(features)
            output_df=runDensityClustering(pca_transformed,f'{results_directory}/{file}-density-pca')
            output_df.to_csv(f'{results_directory}/{file}-density-pca.csv')
            
            print("umap....")
            reducer = umap.UMAP(random_state=42,n_components=2)
            umap_transformed = reducer.fit_transform(features)
            output_df=runDensityClustering(umap_transformed,f'{results_directory}/{file}-density-umap')
            output_df.to_csv(f'{results_directory}/{file}-density-umap.csv')
            
            
            pca_50 = PCA(n_components=50)
            pca_result_50 = pca_50.fit_transform(features)
            tsne_transformed = TSNE(random_state = 42, n_components=3,verbose=0, perplexity=40, n_iter=400).fit_transform(pca_result_50)
            output_df=runDensityClustering(tsne_transformed,f'{results_directory}/{file}-density-pca-tsne')
            output_df.to_csv(f'{results_directory}/{file}-density-pca-tsne.csv')    
           
    