#import face_recognition

"""
This is the explanation of the code:
The Code in asks for the input file that is the csv file having the file id and the screenshots links (must) and output 
the csv file having the cluster labels as columns.

But if the column for the file id differs from the Text ID change it in the line 254.


"""
import warnings
warnings.filterwarnings('ignore')

import sys
import argparse
import pickle
import cv2
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
from kneed import DataGenerator, KneeLocator
import face_recognition
import sklearn
import argparse
import pickle
import cv2
import urllib.request
import time

def loadFilePaths(dirname):
    files=os.listdir(dirname)
    files_path=[os.path.join(dirname,file) for file in files ]
    return files_path

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def extractFeatures(files_path):
    data=[]
    #files_path.sort()
    for (i, imagePath) in enumerate(files_path):
        #print(" Status: %s / %s" %(i, len(files_path)), end="\r")
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        d=[{"image-path":imagePath,"face-location":box,"face-encodings":enc,"face-area":0,"clustering_label":None} for (box,enc) in zip(boxes,encodings)]
        data.extend(d)
        return d

    
def extractFeaturesUrl(image_urls,file_ids):
    data=[]
    for image_url,file_id in zip(image_urls,file_ids):
        #print(" Status: %s / %s" %(i, len(files_path)), end="\r")
        print(file_id)
        image=url_to_image(image_url)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        d=[{"image-path":file_id,"face-location":box,"face-encodings":enc,"face-area":0,"clustering_label":None} for (box,enc) in zip(boxes,encodings)]
        data.extend(d)
    return data 



def getEncodings(data):
    train_data=np.array(data)
    encodings = [d["face-encodings"] for d in train_data]
    return encodings


def updateArea(data):
    for i,data in enumerate(face_data):
        #print(" Status: %s / %s" %(i, len(face_data)), end="\r")
        #print(data['face-location'])
        coordinates=data['face-location']
        top=coordinates[0]
        right=coordinates[1]
        bottom=coordinates[2]
        left=coordinates[3]
        side1=abs(top-bottom+1)   
        side2=abs(left-right+1)
        area=side1*side2
        data['face-area']=area
    return face_data

def cv_silhouette_scorer(estimator, X):
    estimator.fit(X)
    try:
        cluster_labels = estimator.labels_
    except Exception as e:
      #  print(e,estimator)
        cluster_labels=estimator.predict(X)
    num_labels = len(set(cluster_labels))
    num_samples = len(X)
    if num_labels == 1 or num_labels == num_samples or num_labels<=2:
        return -1
    else:
        return metrics.silhouette_score(X, cluster_labels)


def evaluation_Score(features,y_pred,output_df,model):
    try:
        
        num_labels=len(set(y_pred))
        total_samples=len(y_pred)
        print("labels",num_labels)
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

def runGridSearch(estimator,params_dict,train_data):
    
    cv = [(slice(None), slice(None))]
    gs = GridSearchCV(estimator=estimator, param_grid=params_dict, scoring=cv_silhouette_scorer, cv=cv, n_jobs=-1)
    gs.fit(train_data)
  #  print("Grid search",gs.cv_results_)
    try:
        predicted_labels= gs.best_estimator_.labels_
    except:
        predicted_labels=gs.predict(train_data)
    
    print("best estimator is ",gs.best_estimator_)
    
    
    return predicted_labels


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
 
    return epsilons


def dbscan_model(encodings):
    output_df = pd.DataFrame(index=['DBSCAN-GridSearch'],columns=['n_clusters','silhouette','calinski','davies'])
    params_dict = {'eps':[0.3,0.4,0.5,0.6,0.7],'min_samples':[5],'metric':['euclidean','manhattan','mahalanobis']}
    predicted_labels=runGridSearch(sklearn.cluster.DBSCAN(),params_dict,encodings)
    evaluation_Score(encodings,predicted_labels,output_df,'DBSCAN-GridSearch')
    if(set(predicted_labels).issuperset({-1})):
        n_clustersLen=len(set(predicted_labels))-1
    else:
        n_clustersLen=len(set(predicted_labels))
    output_df.loc['DBSCAN-GridSearch','n_clusters']=n_clustersLen
    return output_df,predicted_labels

def updateClusterLabel(face_data,predicted_labels):
    for i,data in enumerate(face_data):
        data['clustering_label']=predicted_labels[i]
    return face_data   

def updateClusterDataframe(df):
    files_path=df['file-id']
    for i in range(len(files_path)):
        file_id=df['file-id'][i]
        area_cluster=[]
        for data in face_data:

            if(data['image-path']==file_id):
                sub_area_cluster=[]
                #print(data['face-area'])
                sub_area_cluster.append(data['face-area'])
                sub_area_cluster.append(data['clustering_label'])

                area_cluster.append(sub_area_cluster)

        #print("Before sorting",area_cluster)
        area_cluster=sorted(area_cluster, key = lambda x: x[0],reverse=True)  
        #print("After sorting",area_cluster)
        try:
            df['largest1'][i]=area_cluster[0][1]
            df['largest2'][i]=area_cluster[1][1]
            df['largest3'][i]=area_cluster[2][1]
           # print("done")
        except:
            #print("Exception")
            pass



    return df          


    
if __name__ == "__main__":
    start_time=time.time()
    
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument("-i","--input_file", required=True, help="path to  input file")
    #my_parser.add_argument("-f","--file_id_column", required=True, help="file id column ")
    my_parser.add_argument("-o","--output_file", required=True, help="path to  output_file")
    
    
        
    args = my_parser.parse_args()
    

    input_file=args.input_file
    output_file=args.output_file
    
  
    input_df=pd.read_excel(input_file)
    
    image_urls=list(input_df['Screenshot'])
    file_ids=list(input_df['Text ID'])
    
    
    df=pd.DataFrame(columns=['file-id','largest1','largest2','largest3'])
    df['file-id']=file_ids

    
    
    print("Total files ....{}".format(len(file_ids)))

    face_data=extractFeaturesUrl(image_urls,file_ids)
    print("Total data ",len(face_data))
    
    
    face_data=updateArea(face_data)
    encodings=getEncodings(face_data)
    
    

    output_df,predicted_labels=dbscan_model(encodings)
    print(Counter(predicted_labels))
    print(output_df)
    
    face_data=updateClusterLabel(face_data,predicted_labels)
    clustering_labels=[data['clustering_label'] for data in face_data]
    print("Total clustering and predicted labels",len(clustering_labels),len(predicted_labels))
   
    
    df=updateClusterDataframe(df)
    print(df.head(5))
    df.to_csv(output_file,index=False)
    
    
    
    
    
    
    
    
    

    
    
    
    """
    files_path=loadFilePaths('ideology_image_dataset')
    files_path.sort()
    
    print("Total file paths",len(files_path))
    df=pd.DataFrame(columns=['file-id','largest1','largest2','largest3'])
    df['file-id']=files_path
    print(df.head(5))
    face_data=extractFeatures(files_path[0:2])
    print(face_data[0])
    print(len(face_data))
   

    
    
    
    with open('ideology_data_face_Embeddings2.pickle','rb') as f:
        face_data=pickle.load(f)
        
    face_data=updateArea(face_data)
    encodings=getEncodings(face_data)


    print(len(encodings),len(face_data))
    output_df,predicted_labels=dbscan_model(encodings)
    print(Counter(predicted_labels))
    print(output_df)
    print(len(predicted_labels))
    face_data=updateClusterLabel(face_data,predicted_labels)
    clustering_labels=[data['clustering_label'] for data in face_data]
    print(len(clustering_labels),len(predicted_labels))
    print(clustering_labels[0:5])
    print(predicted_labels[0:5])
   
    
    df=updateClusterDataframe(df)
    print(df.head(5))
    """
    
    end_time=time.time()
    print("DONE>>>",end_time-start_time)
    
    
        
    
    
    