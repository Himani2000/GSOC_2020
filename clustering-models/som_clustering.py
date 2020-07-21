import numpy as np
import os
from minisom import MiniSom
import numpy as np
import pandas as pd
import math
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 

import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
import umap
import argparse
import image_utils



def som(som_shape,data,sigma=0.5,learning_rate=0.5,neighborhood_function='gaussian',activation_distance='euclidean'):
    #print(neighborhood_function,activation_distance,sigma)
    print(som_shape)
    som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=.5, learning_rate=.5,
              neighborhood_function='gaussian',activation_distance='chebyshev',random_seed=10)

    som.train_batch(data, 500, verbose=True)

    # each neuron represents a cluster
    winner_coordinates = np.array([som.winner(x) for x in data]).T
    # with np.ravel_multi_index we convert the bidimensional
    # coordinates to a monodimensional index
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
    
    
    return cluster_index,som.quantization_error(data)


def runClustering(data,filename):
    output_df = pd.DataFrame(index=['SOM'],columns=['n_clusters','quantization_loss'])

    
    map_dimension=math.floor(math.sqrt(5*math.sqrt(data.shape[0])))
    som_shape=(map_dimension,map_dimension)
    
    predicted_labels,loss=som(som_shape,data)
    np.save(filename+'_som_.npy', np.array(predicted_labels))
    
    
    output_df.loc['SOM','n_clusters']=len(set(predicted_labels))
    output_df.loc['SOM','quantization_loss']=loss
    
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
    
    print(image_directory,feature_directory,results_directory)
    for file in list(os.listdir(feature_directory)):
        if file.endswith('.npy'):
            print(f'{feature_directory}/{file}')
            features=image_utils.loadFeatures(os.path.join(feature_directory,file))
            df=pd.DataFrame(features)
            df=df.fillna(0)
            features=df.to_numpy()
            """
            data=features
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
            output_df=runClustering(data,f'{results_directory}/{file}-som-tsne')
            output_df.to_csv(f'{results_directory}/{file}-som-tsne.csv')
            """
            
            print("tsne....")
            tsne_transformed=TSNE(n_components=3, n_jobs=-1).fit_transform(features)
            data=tsne_transformed
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
            
            output_df=runClustering(data,f'{results_directory}/{file}-som-tsne')
            output_df.to_csv(f'{results_directory}/{file}-som-tsne.csv')
            
            print("pca....")
            pca_transformed=PCA(n_components=3).fit_transform(features)
            data=pca_transformed
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
            output_df=runClustering(data,f'{results_directory}/{file}-som-pca')
            output_df.to_csv(f'{results_directory}/{file}-som-pca.csv')

            print("umap....")
            reducer = umap.UMAP(random_state=42,n_components=2)
            umap_transformed = reducer.fit_transform(features)
            data=umap_transformed
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
            output_df=runClustering(data,f'{results_directory}/{file}-som-umap')
            output_df.to_csv(f'{results_directory}/{file}-som-umap.csv')

            print("pca and tsne")
            pca_50 = PCA(n_components=50)
            pca_result_50 = pca_50.fit_transform(features)
            tsne_transformed = TSNE(random_state = 42, n_components=3,verbose=0, perplexity=40, n_iter=400).fit_transform(pca_result_50)
            data=tsne_transformed
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
            output_df=runClustering(data,f'{results_directory}/{file}-som-pca-tsne')
            output_df.to_csv(f'{results_directory}/{file}-som-pca-tsne.csv')
            
            
            
            
            
            
            
    
