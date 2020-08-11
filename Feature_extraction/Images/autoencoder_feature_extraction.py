#this is the code to extract the autoencoder feature representation 
import pandas as pd 
import numpy as np
import os 
import argparse
import image_utils
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Flatten,Reshape
from keras.models import Model
#from keras import backend as K

def saveFeatures(features,modelname,filename,dirname):
    saved_filename=filename+'_'+modelname
    saved_filename=os.path.join(dirname,saved_filename)

    print("saving",saved_filename+'.npy')
    np.save(saved_filename+'.npy',features)

def single_layer_autoencoder(encoding_dim,input_dimension):
        input_img=Input(shape=(input_dimension,))
        #encoder
        encoded=Dense(encoding_dim,activation="relu")(input_img)

        #decoder 
        decoded=Dense(input_dimension,activation="sigmoid")(encoded)
        autoencoder=Model(input_img,decoded)
        encoder=Model(input_img,encoded)

        #encoder and decoder models 
        return autoencoder,encoder

def sparse_autoencoder(encoding_dim,input_dimension):
        input_img=Input(shape=(input_dimension,))

        encoded=Dense(256,activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_img)
        encoded=Dense(128,activation='relu')(input_img)
        encoded=Dense(encoding_dim,activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_img)

        decoded=Dense(128,activation='sigmoid')(encoded)
        decoded=Dense(256,activation='sigmoid')(decoded)
        decoded=Dense(input_dimension,activation='sigmoid')(decoded)

        autoencoder=Model(input_img,decoded)
        encoder=Model(input_img,encoded)


        return autoencoder,encoder

def deep_autoencoder(encoding_dim,input_dimension):
        input_img=Input(shape=(input_dimension,))

        encoded=Dense(256,activation="relu")(input_img)
        encoded=Dense(128,activation="relu")(encoded)
        encoded=Dense(64,activation="relu")(encoded)
        encoded=Dense(32,activation="relu")(encoded)

        decoded=Dense(64,activation="relu")(encoded)
        decoded=Dense(128,activation="relu")(decoded)
        decoded=Dense(256,activation="relu")(decoded)

        decoded=Dense(input_dimension,activation="relu")(decoded)

        autoencoder=Model(input_img,decoded)
        return autoencoder,encoder


def convolutional_autoencoder(input_image_dimension):

        x=input_image_dimension[0]
        y=input_image_dimension[1]
        z=input_image_dimension[2]
        print(x,y,z)
        input_img = Input(shape=(x,y, z))  # adapt this if using `channels_first` image data format

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)



        x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu',padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        encoder=Model(input_img,encoded)
        return autoencoder,encoder


def denoising_convolutional_autoencoder(input_image_dimension):
  
        x=input_image_dimension[0]
        y=input_image_dimension[1]
        z=input_image_dimension[2]

        input_img = Input(shape=(x,y,z))  # adapt this if using `channels_first` image data format

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (7, 7, 32)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        encoder=Model(input_img,encoded)
        return autoencoder,encoder


def convolutional_autoencoder2(input_image_dimension):
        x=input_image_dimension[0]
        y=input_image_dimension[1]
        z=input_image_dimension[2]
        input_img = Input(shape=(x,y, z))

        #encoder
        #input = 28 x 28 x 1 (wide and thin)
        encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
        encoded = MaxPooling2D(pool_size=(2, 2))(encoded) #14 x 14 x 32
        encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded) #14 x 14 x 64
        encoded = MaxPooling2D(pool_size=(2, 2))(encoded) #7 x 7 x 64
        encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded) #7 x 7 x 128 (small and thick)
        encoded = MaxPooling2D(pool_size=(2, 2))(encoded)

        #decoder
        decoded = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded) #7 x 7 x 128
        decoded= UpSampling2D((2,2))(decoded) # 14 x 14 x 128
        decoded= Conv2D(128, (3, 3), activation='relu', padding='same')(decoded) # 14 x 14 x 64
        decoded= UpSampling2D((2,2))(decoded)
        decoded= Conv2D(64, (3, 3), activation='relu', padding='same')(decoded) # 14 x 14 x 64
        decoded= UpSampling2D((2,2))(decoded) # 28 x 28 x 64
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded) # 28 x 28 x 1

        autoencoder = Model(input_img, decoded)
        encoder=Model(input_img,encoded)
        return autoencoder,encoder

def autoencoder_fit(autoencoder,encoder,data):
        #autoencoder=single_layer_autoencoder(encoding_dim,input_dimension)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        autoencoder.fit(data,data,epochs=100,batch_size=128,shuffle=True)
        decoded_imgs = autoencoder.predict(data)
        encoded_embedding=encoder.predict(data)

        #visualize(data,decoded_imgs)
        return encoded_embedding,decoded_imgs


def autoencoder_fit_noisy(autoencoder,encoder,data,data_noisy):
        #autoencoder=single_layer_autoencoder(encoding_dim,input_dimension)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(data_noisy, data,epochs=100,batch_size=128,shuffle=True)
        decoded_imgs = autoencoder.predict(data)
        encoded_embedding=encoder.predict(data)
        #visualize(data,decoded_imgs)
        return encoded_embedding,decoded_imgs



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
        my_parser.add_argument("save_filename",type=str,help="the save filename ")

        args = my_parser.parse_args()

        image_directory=args.image_directory

        feature_directory=args.feature_directory
        results_directory=args.results_directory
        save_filename=args.save_filename


        args = my_parser.parse_args()

        

        files_path=image_utils.loadFilePaths(image_directory)
        print("Total files_path",len(files_path))
        print(image_directory,feature_directory,results_directory)

        print("save filename is ",save_filename)
        for file in list(os.listdir(feature_directory)):
            if file.endswith('.npy'):
                #print(f'{feature_directory}/{file}')
                print(os.path.join(feature_directory,file))
                if(save_filename=="ideology"):
                    fileword=file.split('_')[2]
                    print(fileword)
                elif(save_filename=="ideology_crop"):
                    fileword=file.split('_')[3]
                    print(fileword)
                    
                save_filename2=fileword+'_'+save_filename
                print("saving the name as",save_filename2)
                
                features=image_utils.loadFeatures(os.path.join(feature_directory,file))
                data=features
                data = data.astype('float32') / 255.
                print(data.shape)
                encoding_dim = 128
                input_dimension=data.shape[1]
                
                if(fileword=='hog'):
                    data_image=data.reshape(-1,90,90,1)
                    
                else:
                    data_image=data.reshape(-1,128,128,1)
                
                noise_factor = 0.5
                data_image_noisy = data_image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data_image.shape) 

                data_image_noisy = np.clip(data_image_noisy, 0., 1.)
                
                print("the data shape is",data.shape)
                print("the image shape is",data_image.shape)
                
                
                
                
                
                
                
                
                
                print("Calling the single layer")
                autoencoder,encoder=single_layer_autoencoder(encoding_dim,input_dimension)
                encoded_embedding,decoded_imgs=autoencoder_fit(autoencoder,encoder,data)
                print("shape for single is ",encoded_embedding.shape)
                print(encoded_embedding)

                saveFeatures(encoded_embedding,'single_layer',save_filename2,results_directory)
                
                
                print("for the sparse autoencoder:")
                autoencoder,encoder=sparse_autoencoder(encoding_dim,input_dimension)
                encoded_embedding,decoded_imgs=autoencoder_fit(autoencoder,encoder,data)
                print("shape for sparse is ",encoded_embedding.shape)
                print(encoded_embedding)
                saveFeatures(encoded_embedding,'sparse_layer',save_filename2,results_directory)
                
                
                
                
                
                print("for the deep autoencoder:")
                autoencoder,encoder=deep_autoencoder(encoding_dim,input_dimension)     
                encoded_embedding,decoded_imgs=autoencoder_fit(autoencoder,encoder,data)
                print("shape for sparse is ",encoded_embedding.shape)
                print(encoded_embedding)
                saveFeatures(encoded_embedding,'deep_layer',save_filename2,results_directory)
                
                print("Convolutional1")
                autoencoder,encoder=convolutional_autoencoder(data_image.shape[1:])
                
                encoded_embedding,decoded_imgs=autoencoder_fit(autoencoder,encoder,data_image)
                print("shape for convolutional 1  is ",encoded_embedding.shape)
                print(encoded_embedding)
                embedding=[]
                for i in encoded_embedding:
                    embedding.append(i.flatten())
                embedding=np.array(embedding)
                print(embedding.shape)
                saveFeatures(embedding,'convolutional1',save_filename2,results_directory)
                
                print("Convolutional2")
                autoencoder,encoder=convolutional_autoencoder2(data_image.shape[1:])
                encoded_embedding,decoded_imgs=autoencoder_fit(autoencoder,encoder,data_image)
                print("shape for convolutional 2  is ",encoded_embedding.shape)
                print(encoded_embedding)
                embedding=[]
                for i in encoded_embedding:
                    embedding.append(i.flatten())
                embedding=np.array(embedding)
                print(embedding.shape)
                saveFeatures(embedding,'convolutional2',save_filename2,results_directory)
                
                print("Convolutional3")
                autoencoder,encoder=denoising_convolutional_autoencoder(data_image.shape[1:])
                encoded_embedding,decoded_imgs=autoencoder_fit(autoencoder,encoder,data_image)
                print("shape for convolutional 3  is ",encoded_embedding.shape)
                print(encoded_embedding)
                embedding=[]
                for i in encoded_embedding:
                    embedding.append(i.flatten())
                embedding=np.array(embedding)
                print(embedding.shape)
                saveFeatures(embedding,'convolutional3',save_filename2,results_directory)
                
                
                print("For the denoising autoencoder")
                
                autoencoder,encoder=denoising_convolutional_autoencoder(data_image_noisy.shape[1:])

                encoded_embedding,decoded_imgs=autoencoder_fit_noisy(autoencoder,encoder,data_image,data_image_noisy)
                print("shape for denoising is   is ",encoded_embedding.shape)
                print(encoded_embedding)
                embedding=[]
                for i in encoded_embedding:
                    embedding.append(i.flatten())
                embedding=np.array(embedding)
                print(embedding.shape)
                saveFeatures(embedding,'denoisingConv',save_filename2,results_directory)
                

                #print("shape for ")
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
