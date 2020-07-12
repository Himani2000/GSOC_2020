import sklearn
from sklearn import metrics
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.model_selection import GridSearchCV
from kneed import DataGenerator, KneeLocator


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
    try:
        predicted_labels= gs.best_estimator_.labels_
    except:
        predicted_labels=gs.predict(train_data)
    
    print("best estimator is ",gs.best_estimator_)
    return predicted_labels



