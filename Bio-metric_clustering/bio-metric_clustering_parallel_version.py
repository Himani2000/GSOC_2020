import cv2
import os
import numpy as np
import pandas as pd
import face_recognition
import argparse
import pickle
import cv2
import urllib.request
import time
import multiprocessing
from multiprocessing import Pool
from collections import Counter
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
from kneed import DataGenerator, KneeLocator
import sklearn
import json
import requests

#singularity pull --force --name bio-metric.img shub://Himani2000/GSOC_2020:clustering

#singularity exec -B `pwd` biometric_deploy/bio-metric.img python3 biometric_deploy/bio-metric_clustering2.py


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image



def extractFeaturesParallel(file_image_url):
    d=[]
    try:
        image_url=file_image_url[1]
        file_id=file_image_url[0]
        image=url_to_image(image_url)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)
        d=[{"image-url":image_url, "image-path":file_id,"face-location":box,"face-encodings":enc,"face-area":0,"clustering_label":None} for (box,enc) in zip(boxes,encodings)]
    except:
        pass
    
    return d

def printPrallel(file_image_url):
    print("The file id is ",file_image_url[0])
    print("The image url is ",file_image_url[1])
    return 1


def extractFeaturesUrl(image_urls,file_ids):
    data=[]
    for image_url,file_id in zip(image_urls,file_ids):
        #print(" Status: %s / %s" %(i, len(files_path)), end="\r")
        print(file_id)
        image=url_to_image(image_url)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        d=[{"image-path":file_id,"image-url":image_url,"face-location":box,"face-encodings":enc,"face-area":0,"clustering_label":None} for (box,enc) in zip(boxes,encodings)]
        data.extend(d)
    return data 
def getEncodings(data):
    train_data=np.array(data)
    encodings = [d[0]["face-encodings"] for d in train_data]
    return encodings


def updateArea(data):
    for i,data in enumerate(face_data):
        #print(" Status: %s / %s" %(i, len(face_data)), end="\r")
        #print(data['face-location'])
        coordinates=data[0]['face-location']
        top=coordinates[0]
        right=coordinates[1]
        bottom=coordinates[2]
        left=coordinates[3]
        side1=abs(bottom-top)+1   
        side2=abs(right-left)+1
        area=side1*side2
        data[0]['face-area']=area
    return face_data

def cv_silhouette_scorer(estimator, X):
    estimator.fit(X)
    try:
        cluster_labels = estimator.labels_
    except Exception as e:
      #  print(e,estimator)
        cluster_labels=estimator.predict(X)
    num_labels = len(set(cluster_labels))
    no_noise_labels=num_labels
    if(set(cluster_labels).issuperset({-1})):
        n_clustersLen=len(set(cluster_labels))-1
        no_noise_labels=n_clustersLen
    
    num_samples = len(X)
    if num_labels == 1 or num_labels == num_samples or no_noise_labels==1:
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
            
        else:
            output_df.loc[model,'silhouette'] =metrics.silhouette_score(features,y_pred)


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


def dbscan_model(encodings):
    output_df = pd.DataFrame(index=['DBSCAN-GridSearch'],columns=['n_clusters','silhouette','calinski','davies',
                                                                               'S_Dbw'])
    params_dict = {'eps':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],'min_samples':[1,2,3,4,5],'metric':['euclidean','manhattan','mahalanobis']}
    predicted_labels=runGridSearch(sklearn.cluster.DBSCAN(n_jobs=-1),params_dict,encodings)
    evaluation_Score(encodings,predicted_labels,output_df,'DBSCAN-GridSearch')
    if(set(predicted_labels).issuperset({-1})):
        n_clustersLen=len(set(predicted_labels))-1
    else:
        n_clustersLen=len(set(predicted_labels))
    output_df.loc['DBSCAN-GridSearch','n_clusters']=n_clustersLen
    return output_df,predicted_labels

    
def updateClusterLabel(face_data,predicted_labels):
    for i,data in enumerate(face_data):
        data[0]['clustering_label']=predicted_labels[i]
    return face_data   
  

def updateClusterDataframe(df):
    files_path=df['file-id']
    for i in range(len(files_path)):
        file_id=df['file-id'][i]
        area_cluster=[]
        for data in face_data:

            if(data[0]['image-path']==file_id):
                sub_area_cluster=[]
                #print(data['face-area'])
                sub_area_cluster.append(data[0]['face-area'])
                sub_area_cluster.append(data[0]['clustering_label'])

                area_cluster.append(sub_area_cluster)

        #print("Before sorting",area_cluster)
        area_cluster=sorted(area_cluster, key = lambda x: x[0],reverse=True)  
        #print("After sorting",area_cluster)
        try:
            df['largest1'][i]=area_cluster[0][1]
            df['largest2'][i]=area_cluster[1][1]
            df['largest3'][i]=area_cluster[2][1]
           # print("done")
        except Exception as e :
            #print("Exception",e)
            pass



    return df      
    


if __name__ == '__main__':
        start_time=time.time()
        #after get do exception handling 
        #service 
        while(True):
            try:
                
                
                #here the actual get request will be called
                #with open('getJobData.json','r') as f:
                #    full_data=json.loads(f.read())

                #full_data=requests.get('https://gallo.case.edu/rapidannotator/frontpage')
                full_data=requests.get("http://beta.rapidannotator.org:8010/frontpage/")
                print("dvhdw")
                for data in full_data['jobsData'][0:2]:
                    # here the post request 1 for processing 
                    start_time2=time.time()
                    experiment_id=data['experiment_id']
                    job_id=data['jobId']
                    file_ids=data['fileId']
                    image_urls=data['imageURLS']

                    file_image_url=[]
                    for file_id,image_url in zip(file_ids,image_urls):
                        file_image_url.append([file_id,image_url])



                    print("The cpu counts are {}".format(multiprocessing.cpu_count()))
                    cpu_counts=multiprocessing.cpu_count()


                    pool_number=cpu_counts-2
                    
                    print(pool_number)
                    p=Pool(pool_number)
                    face_data=p.map(extractFeaturesParallel, file_image_url)
                    face_data=[data for data in face_data if len(data)!=0]



                    face_data=updateArea(face_data)
                    encodings=getEncodings(face_data)
                    print("The length of the face data is {}".format(len(face_data)))

                    output_df,predicted_labels=dbscan_model(encodings)
                    print(Counter(predicted_labels))
                    print(output_df)


                    face_data=updateClusterLabel(face_data,predicted_labels)

                    df=pd.DataFrame(columns=['file-id','largest1','largest2','largest3'])
                    df['file-id']=file_ids
                    df=updateClusterDataframe(df)
                    df=df.where(df.notnull(), None)
                    print(df.head(5))
                    output_data={}
                    output_data['file_ids']=df['file-id'].to_list()
                    output_data['largest1']=df['largest1'].to_list()
                    output_data['largest2']=df['largest2'].to_list()
                    output_data['experiment_id']=experiment_id
                    output_data['job_id']=job_id
                    
                    end_time2=time.time()
                    print(output_data)
                    print("[TIME TAKEN]",end_time2-start_time2)
                    
                    #here post the data and mark completed 
           
            except Exception as e:
                print(e)
                pass

        
        end_time=time.time()
        print("[TOTAL TIME TAKEN]",end_time-start_time)
        print("DONE")
        p.close()      