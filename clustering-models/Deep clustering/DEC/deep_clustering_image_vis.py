import numpy as np
import os

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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


def showClustering(predicted_labels,label,pdf,algorithm,features):
    label_indexs= np.where(predicted_labels==label)[0]
    print("CLUSTER--> ",label,"TOTAL IMAGES--> ",len(label_indexs))
    print(label_indexs)
    
    

    if(len(label_indexs)>=500):
        fig=plt.figure(figsize=(10, 500))
        
        
    elif(len(label_indexs)>100 and len(label_indexs)<500):
        fig=plt.figure(figsize=(10, 40))
    elif(len(label_indexs)>=50 and len(label_indexs)<100):
        fig=plt.figure(figsize=(10, 10))
        
    elif(len(label_indexs)>=20 and len(label_indexs)<50):
        fig=plt.figure(figsize=(10, 3))
    
    elif(len(label_indexs)>=0 and len(label_indexs)<20):
        fig=plt.figure(figsize=(10, 2))
    
    else:
        fig=plt.figure(figsize=(10,50))
    
    plt.title(' f The cluster -> %s {label} and total images->{len(label_indexs)}')
   # plt.title('f the cluster %s and total images %i' %label %len(label_indexs))
    
    plt.gca().set_axis_off()
    plt.subplots_adjust(hspace = 0, wspace = 0)

    for i,index in enumerate(label_indexs):
        
       
        
        #image = cv2.imread(files_path[index])
        #image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image=features[index]
        
        columns = 10
        rows = np.ceil(len(label_indexs)/float(columns))
        
        fig.add_subplot(rows,columns, i+1)
        plt.axis("off")
       
        if(algorithm=="stl"):
            
            plt.imshow(image.reshape(96,96),cmap="gray")
        elif(algorithm=="original"):
            plt.imshow(image)
        
    
   
    pdf.savefig()
    plt.close(fig)
    plt.clf()
    plt.cla()


def makePdf(y_pred,algorithm,filename,features):
        
        predicted_labels=y_pred
        unique_labels=set(predicted_labels)
        with PdfPages(os.path.join('results',filename+'.pdf')) as pdf:
            for label in list(unique_labels):
                if(label!=-1):
                    showClustering(predicted_labels,label,pdf,algorithm,features)

if __name__ == "__main__":
    
    x,y=load_stl()
    print("X->",x.shape)
    print("Y->",y.shape)
    
    files=os.listdir('results')
    algorithm='stl'
    for file in files:
        if(file.endswith('.npy')):
            
            labels=np.load(os.path.join('results',file))
            print("Loading the file {} shape of the labels is {}".format(file,labels.shape))
            makePdf(labels,algorithm,file.split('.npy')[0],x)
            print("DONE PDF")
            
    
    
    
    
    