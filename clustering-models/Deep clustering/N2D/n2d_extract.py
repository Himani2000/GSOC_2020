import argparse
import os
import random as rn
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
#import umap
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from sklearn import metrics
from sklearn import mixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.utils.linear_assignment_ import linear_assignment
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

def autoencoder(dims, act='relu'):
    n_stacks = len(dims) - 1
    x = Input(shape=(dims[0],), name='input')
    h = x
    for i in range(n_stacks - 1):
        h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)
    h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)
    for i in range(n_stacks - 1, 0, -1):
        h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)
    h = Dense(dims[0], name='decoder_0')(h)

    return Model(inputs=x, outputs=h)

if __name__ == "__main__":

    start_time=time()
    n_clusters =10 # this quantity is unknown at real time 
    batch_size=256
    pretrain_epochs=300 #actualy it is 1000
    save_dir=''
    umap_dim=2
    umap_neighbors=10
    umap_min_dist="0.00"
    umap_metric="euclidean"
    
    # these below are the parametes to be changed
    cluster="GMM"
    eval_all=False
    manifold_learner="UMAP"
    
    optimizer = 'adam'
    #x,y=load_stl()
    #print("X->",x.shape)
    #print("Y->",y.shape)
    
    # now inidtializing the autoencoder model 
    
    
    base_path='/mnt/rds/redhen/gallina/home/hxn147/original_color_features'
    for file in os.listdir(base_path):
        if(file!='ideology_face_dataset.npy'):
            print("[RUNNING] FOR THE FILE ",file)

            x=np.load(os.path.join(base_path,file))
            x=x.astype('float32') / 255.
            shape = [x.shape[-1], 500, 500, 2000,n_clusters]
            autoencoder = autoencoder(shape)

            hidden = autoencoder.get_layer(name='encoder_%d' % (len(shape) - 2)).output
            encoder = Model(inputs=autoencoder.input, outputs=hidden)

            autoencoder.compile(loss='mse', optimizer=optimizer)
            autoencoder.fit(x,x,batch_size=batch_size,epochs=pretrain_epochs,verbose=0)

            hl = encoder.predict(x)

            np.save(os.path.join('features3',file),hl)
            print("Saving the features")
        
        
    
        
        
        
        
        
    """
    print("RUNNING FOR THE NORMAL DATASET.....")
    x_normal=np.load('/mnt/rds/redhen/gallina/home/hxn147/ideology_dataset_normal_features_conventional2/ideology__grayscale_.npy')
    print(x_normal.shape)
    
    x=x_normal
    shape = [x.shape[-1], 500, 500, 2000,n_clusters]
    autoencoder = autoencoder(shape)

    hidden = autoencoder.get_layer(name='encoder_%d' % (len(shape) - 2)).output
    encoder = Model(inputs=autoencoder.input, outputs=hidden)
    
    autoencoder.compile(loss='mse', optimizer=optimizer)
    autoencoder.fit(x,x,batch_size=batch_size,epochs=pretrain_epochs,verbose=0)
    
    hl = encoder.predict(x)
    np.save("features2/normal_grayscale_features.npy",hl)
    print("Saving the features")
    
    """
    
    
    """
    try:
        print("RUNNING FOR THE CROP  DATASET.....")

        x_crop=np.load('/mnt/rds/redhen/gallina/home/hxn147/ideology_dataset_crop_features_conventional2/ideology_crop__grayscale_.npy')
        print(x_crop.shape)

        x=x_crop
        shape = [x.shape[-1], 500, 500, 2000,n_clusters]
        autoencoder = autoencoder(shape)

        hidden = autoencoder.get_layer(name='encoder_%d' % (len(shape) - 2)).output
        encoder = Model(inputs=autoencoder.input, outputs=hidden)

        autoencoder.compile(loss='mse', optimizer=optimizer)
        autoencoder.fit(x,x,batch_size=batch_size,epochs=pretrain_epochs,verbose=0)

        hl = encoder.predict(x)
        np.save("features2/crop_grayscale_features.npy",hl)
        print("Saving the features")
    except:
        pass
     
     """
    
    
    """
    try:
        print("RUNNING FOR THE PERSON DATASET.....")

        x_person=np.load('/mnt/rds/redhen/gallina/home/hxn147/ideology_person_dataset_features_conventional/ideology__grayscale_.npy')
        print(x_person.shape)

        x=x_person
        shape = [x.shape[-1], 500, 500, 2000,n_clusters]
        autoencoder = autoencoder(shape)

        hidden = autoencoder.get_layer(name='encoder_%d' % (len(shape) - 2)).output
        encoder = Model(inputs=autoencoder.input, outputs=hidden)

        autoencoder.compile(loss='mse', optimizer=optimizer)
        autoencoder.fit(x,x,batch_size=batch_size,epochs=pretrain_epochs,verbose=0)

        hl = encoder.predict(x)
        np.save("features2/person_grayscale_features.npy",hl)
        print("Saving the features")
    
    except Exception as e:
        print(e)
        pass
    
    
    
    try:
    
        print("RUNNING FOR THE FACE DATASET.....")


        x_face=np.load('/mnt/rds/redhen/gallina/home/hxn147/ideology_face_dataset_features_conventional/ideology__grayscale_.npy')
        print(x_face.shape)


        x=x_face
        shape = [x.shape[-1], 500, 500, 2000,n_clusters]
        autoencoder = autoencoder(shape)

        hidden = autoencoder.get_layer(name='encoder_%d' % (len(shape) - 2)).output
        encoder = Model(inputs=autoencoder.input, outputs=hidden)

        autoencoder.compile(loss='mse', optimizer=optimizer)
        autoencoder.fit(x,x,batch_size=batch_size,epochs=pretrain_epochs,verbose=0)

        hl = encoder.predict(x)
        np.save("features2/face_grayscale_features.npy",hl)
        print("Saving the features")
    
    except Exception as e:
        print(e)
        pass
    """
    
    
    
    
    end_time=time()
    
    print("COMPLETE ..... ")
    print("TOTAL TIME TAKEN ....... ",end_time-start_time)
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    