import umap
import argparse
import os
import random as rn
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import umap
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from sklearn import metrics
from sklearn import mixture
from sklearn.cluster import KMeans, SpectralClustering,DBSCAN,OPTICS
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
#from sklearn.utils.linear_assignment_ import linear_assignment
from time import time


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def load_stl(data_path='/mnt/rds/redhen/gallina/home/hxn147/data/stl/stl10_binary/'):
    y1 = np.fromfile(data_path + '/train_y.bin', dtype=np.uint8) - 1
    y2 = np.fromfile(data_path + '/test_y.bin', dtype=np.uint8) - 1
    y = np.concatenate((y1, y2))
    x1 = np.fromfile(data_path + '/train_X.bin', dtype=np.uint8)
    x1 = x1.reshape((int(x1.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
    x2 = np.fromfile(data_path + '/test_X.bin', dtype=np.uint8)
    x2 = x2.reshape((int(x2.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
    x = np.concatenate((x1, x2)).astype(float)
    features=[]
    for image in x:
        gray = rgb2gray(image)    
        features.append(gray.flatten())
    features = np.asarray(features)
    return features,y

def cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster):
    # find manifold on autoencoded embedding
    if manifold_learner == 'UMAP':
        md = float(umap_min_dist)
        hle = umap.UMAP(random_state=0,metric=umap_metric,n_components=umap_dim,n_neighbors=umap_neighbors,min_dist=md).fit_transform(hl)
    elif manifold_learner == 'LLE':
        hle = LocallyLinearEmbedding(n_components=umap_dim,n_neighbors=umap_neighbors).fit_transform(hl)
    elif manifold_learner == 'tSNE':
        hle = TSNE(n_components=umap_dim,n_jobs=16,random_state=0,verbose=0).fit_transform(hl)
    elif manifold_learner == 'isomap':
        hle = Isomap(n_components=umap_dim,n_neighbors=5).fit_transform(hl)

    # clustering on new manifold of autoencoded embedding
    if cluster == 'GMM':
        gmm = mixture.GaussianMixture(covariance_type='full',n_components=n_clusters,random_state=0)
        gmm.fit(hle)
        y_pred_prob = gmm.predict_proba(hle)
        y_pred = y_pred_prob.argmax(1)
    elif cluster == 'KM':
        km = KMeans(init='k-means++',n_clusters=n_clusters,random_state=0,n_init=20)
        y_pred = km.fit_predict(hle)
    elif cluster == 'SC':
        sc = SpectralClustering(n_clusters=n_clusters,random_state=0,affinity='nearest_neighbors')
        y_pred = sc.fit_predict(hle)
        
    elif cluster=='DBSCAN':
        db=DBSCAN()
        y_pred=db.fit_predict(hle)
        
    elif cluster=='OPTICS':
        op=OPTICS()
        y_pred=op.fit_predict(hle)
   

        

    y_pred = np.asarray(y_pred)
    #y = np.asarray(y)
    # y = y.reshape(len(y), )

    #nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    #ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print('=='*80)
    #print("METRICS for the ",cluster,manifold_learner)
    #print(nmi)
    #print(ari)
    print('=' * 80)

  
    return y_pred

if __name__ == "__main__":

    start_time=time()
    n_clusters =10 # this quantity is unknown at real time 
    batch_size=256
    pretrain_epochs=1000
    save_dir=''
    umap_dim=2
    umap_neighbors=10
    umap_min_dist="0.00"
    umap_metric="euclidean"
    
    # these below are the parametes to be changed
    n_clusters=10
    eval_all=False
    
    
    #x,y=load_stl()
    #print("X->",x.shape)
    #print("Y->",y.shape)
    
    y=None
    
    hl=np.load("/mnt/rds/redhen/gallina/home/hxn147/original_color_features/ideology_crop_image_dataset.npy")
    print("Loaded features",hl.shape)
    
   
    #cluster_techniques=['GMM','KM','SC','DBSCAN','OPTICS']
    #dataset='stl'
    
    cluster_techniques=['DBSCAN','OPTICS']
    dataset='ideology_crop'
    manifold_learner='UMAP'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
    manifold_learner='LLE'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
    manifold_learner='tSNE'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
    manifold_learner='isomap'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
        
    
    hl=np.load("/mnt/rds/redhen/gallina/home/hxn147/original_color_features/ideology_image_dataset.npy")
    print("Loaded features",hl.shape)
    dataset='ideology_normal'
   
    #cluster_techniques=['GMM','KM','SC','DBSCAN','OPTICS']
    #dataset='stl'
    
    cluster_techniques=['DBSCAN','OPTICS']
    
    manifold_learner='UMAP'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
    manifold_learner='LLE'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
    manifold_learner='tSNE'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
    manifold_learner='isomap'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
        
        
        
    hl=np.load("/mnt/rds/redhen/gallina/home/hxn147/original_color_features/ideology_person_dataset.npy")
    print("Loaded features",hl.shape)
    dataset='ideology_person'
   
    #cluster_techniques=['GMM','KM','SC','DBSCAN','OPTICS']
    #dataset='stl'
    
    cluster_techniques=['DBSCAN','OPTICS']
    
    manifold_learner='UMAP'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
    manifold_learner='LLE'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
    manifold_learner='tSNE'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
    manifold_learner='isomap'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
        
   
    hl=np.load("/mnt/rds/redhen/gallina/home/hxn147/original_color_features/ideology_face_dataset.npy")
    print("Loaded features",hl.shape)
    dataset='ideology_face'
   
    #cluster_techniques=['GMM','KM','SC','DBSCAN','OPTICS']
    #dataset='stl'
    
    cluster_techniques=['DBSCAN','OPTICS']
    
    manifold_learner='UMAP'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
    manifold_learner='LLE'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
    manifold_learner='tSNE'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
        
    manifold_learner='isomap'
    for cluster in cluster_techniques:
        y_pred=cluster_manifold_in_embedding(hl, y, manifold_learner,umap_min_dist,umap_metric,umap_dim,umap_neighbors,n_clusters,cluster)
        filename=dataset+'_'+manifold_learner+"_"+cluster+".npy"
        np.save(os.path.join('results3',filename),y_pred)
    
    
        
    
    
    print("DONE")
        
        
     
        
    
        
        
    
    
    
    
    
    
    
    
    