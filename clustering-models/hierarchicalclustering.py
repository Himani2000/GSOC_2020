#importing the python standard libraries 
import pandas as pd 
import numpy as np
import os 
import sklearn
from sklearn.cluster import AgglomerativeClustering,Birch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
import umap
import argparse

#importing the python files 
import clustering_hyperparameter
import image_utils

def runHierarchicalModels(train_data,output_df,filename):
   
   
        
    print("Agglomerative clustering results")
    params_dict={'linkage':['ward','average'],'distance_threshold':[5,6,10,20,30,40,50,60,70],'n_clusters':[None]}
    predicted_labels=clustering_hyperparameter.runGridSearch(AgglomerativeClustering(),params_dict,train_data)
    
    clustering_hyperparameter.evaluation_Score(train_data,predicted_labels,output_df,'Agglomerative clustering')
    if(set(predicted_labels).issuperset({-1})):
        n_clustersLen=len(set(predicted_labels))-1
    else:
        n_clustersLen=len(set(predicted_labels))
    output_df.loc['Agglomerative clustering','n_clusters']=n_clustersLen
    
    np.save(filename+'_agg_labels.npy', np.array(predicted_labels))
    print(output_df)
    
    print("Birch")
    threshold_estimation=train_data.shape[1]/10
    if(threshold_estimation-10>=10):
        threshold_estimation-=10
    params_dict={'threshold':np.arange(threshold_estimation,threshold_estimation+5,0.3).tolist(),'n_clusters':[None]}
    predicted_labels=clustering_hyperparameter.runGridSearch(Birch(),params_dict,train_data)
    #print("number of distinct labels in the birch os")
    clustering_hyperparameter.evaluation_Score(train_data,predicted_labels,output_df,'Birch')
    if(set(predicted_labels).issuperset({-1})):
        n_clustersLen=len(set(predicted_labels))-1
    else:
        n_clustersLen=len(set(predicted_labels))
    output_df.loc['Birch','n_clusters']=n_clustersLen
    
    np.save(filename+'_birch_labels.npy', np.array(predicted_labels))
    print(output_df)
    
    return output_df

def runHierarchicalClustering(train_data,filename):
   
    output_df = pd.DataFrame(index=['Birch','Agglomerative clustering'],columns=['n_clusters','silhouette','calinski'])
   
    output_df=runHierarchicalModels(train_data,output_df,filename)
    
 
   
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
    for file in list(os.listdir(feature_directory)):
        if file.endswith('.npy'):
            print(f'{feature_directory}/{file}')
            features=image_utils.loadFeatures(os.path.join(feature_directory,file))
            print("tsne....")
            tsne_transformed=TSNE(n_components=3, n_jobs=-1).fit_transform(features)
            output_df=runHierarchicalClustering(tsne_transformed,f'{results_directory}/{file}-hierarchical-tsne')
            output_df.to_csv(f'{results_directory}/{file}-hierarchical-tsne.csv')
            
            
            
            print("pca....")
            pca_dims = PCA().fit(features)
            cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
            d = np.argmax(cumsum >= 0.95) + 1
            print(d)
            if(d==1):
                d=d+1
            pca_transformed=PCA(n_components=d).fit_transform(features)
            output_df=runHierarchicalClustering(pca_transformed,f'{results_directory}/{file}-hierarchical-pca')
            output_df.to_csv(f'{results_directory}/{file}-hierarchical-pca.csv')
            
            print("umap....")
            reducer = umap.UMAP(random_state=42,n_components=2)
            umap_transformed = reducer.fit_transform(features)
            output_df=runHierarchicalClustering(umap_transformed,f'{results_directory}/{file}-hierarchical-umap')
            output_df.to_csv(f'{results_directory}/{file}-hierarchical-umap.csv')
            
            
            pca_50 = PCA(n_components=50)
            pca_result_50 = pca_50.fit_transform(features)
            tsne_transformed = TSNE(random_state = 42, n_components=3,verbose=0, perplexity=40, n_iter=400).fit_transform(pca_result_50)
            output_df=runHierarchicalClustering(tsne_transformed,f'{results_directory}/{file}-hierarchical-pca-tsne')
            output_df.to_csv(f'{results_directory}/{file}-hierarchical-pca-tsne.csv')    
           
