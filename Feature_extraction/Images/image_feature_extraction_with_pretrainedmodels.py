import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.densenet import DenseNet201

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.nasnet import NASNetMobile


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2


from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg16 import preprocess_input


import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
import argparse
import warnings
warnings.filterwarnings('ignore')


def loadPretrainedWeights():
    pretrained_weights={}

    pretrained_weights['vgg16']=VGG16(weights='imagenet', include_top=False,pooling='avg')
    pretrained_weights['vgg19']=VGG19(weights='imagenet', include_top=False,pooling='avg')

    pretrained_weights['resnet50']=ResNet50(weights='imagenet', include_top=False,pooling='avg')

    pretrained_weights['inceptionv3']=InceptionV3(weights='imagenet', include_top=False,pooling='avg')
    pretrained_weights['inception-resentv2']=InceptionResNetV2(weights='imagenet', include_top=False,pooling='avg')


    pretrained_weights['xception']=Xception(weights='imagenet', include_top=False,pooling='avg')

    pretrained_weights['densenet121']=DenseNet121(weights='imagenet', include_top=False,pooling='avg')
    pretrained_weights['densenet169']=DenseNet169(weights='imagenet', include_top=False,pooling='avg')
    pretrained_weights['densenet201']=DenseNet201(weights='imagenet', include_top=False,pooling='avg')
    pretrained_weights['mobilenet']=MobileNet(weights='imagenet', include_top=False,pooling='avg')


  #N retrained_weights['nasnetlarge']=NASNetLarge(weights='imagenet', include_top=False,pooling='avg',input_shape = (224, 224, 3))
  #N pretrained_weights['nasnetmobile']=NASNetMobile(weights='imagenet', include_top=False,pooling='avg')



    
  #N  pretrained_weights['mobilenetV2']=MobileNetV2(weights='imagenet', include_top=False,pooling='avg')
    
    return pretrained_weights

def loadFilePaths(image_directory):
    
    files=os.listdir(image_directory)
    files_path=[os.path.join(image_directory,file) for file in files ]
    return files_path


 # the function is used to calculate the features from the image 
def getFeatures(filelist,model): 
    filelist.sort()
    featurelist = []
    for i, imagepath in enumerate(filelist):
    #for i in tqdm(range(len(filelist))):
        print(" Status: %s / %s" %(i, len(filelist)), end="\r")
        img = image.load_img(filelist[i], target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = np.array(model.predict(img_data))
        featurelist.append(features.flatten())
    return featurelist



def saveFeatures(features,modelname,filename,dirname):
    saved_filename=filename+'_'+modelname
    saved_filename=os.path.join(dirname,saved_filename)

    print("saving",saved_filename+'.npy')
    np.save(saved_filename+'.npy',features)
    
    
if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='List the content of a folder')

    my_parser.add_argument('image_path',
                           metavar='path',
                           type=str,
                           help='the path to the image directory')
    my_parser.add_argument('image_feature_path',
                           metavar='path',
                           type=str,
                           help='the path to save the image extracted features')

    args = my_parser.parse_args()

    image_path=args.image_path
    
    feature_path=args.image_feature_path

    print("The paths are",image_path,feature_path)
    pretrained_weights=loadPretrainedWeights()
    files_path=loadFilePaths(image_path)
    print("length of files are ",len(files_path))
    
    
    #print(os.listdir(feature_path))
   
    
    for model in pretrained_weights.keys():
   # if(model!=vgg16 or model!=vgg19 or model!=resnet50 or model!=inceptionv3):
   # if(model not  in ['vgg16','vgg19','resnet50','inceptionv3']):    
        print("extracting features for -->",model)
        features=getFeatures(files_path,pretrained_weights[model])
        saveFeatures(features,f'model_{model}','ideology_crop',feature_path)
        
    

    


