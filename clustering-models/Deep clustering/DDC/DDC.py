  
from ConvDEC import ConvDEC
import os, csv
from datasets import load_data_conv
from keras.initializers import VarianceScaling
import numpy as np
from time import time

# the code is built on the top of the existing code with some my functionalities also a huge shoutout to->https://github.com/Yazhou-Ren/DDC


def run_exp(dbs, aug_ae,expdir, ae_weights_dir, trials=5, verbose=0):

    for db in dbs:
        if(db=='normal'):
            x=np.load('/mnt/rds/redhen/gallina/home/hxn147/original_color_features/ideology_image_dataset.npy')
            x=x.reshape(-1,128,128,3)
            x=x.astype('float32') / 255.
            print(x.shape)
            y=None
            
        elif(db=='crop'):
            x=np.load('/mnt/rds/redhen/gallina/home/hxn147/original_color_features/ideology_crop_image_dataset.npy')
            x=x.reshape(-1,128,128,3)
            x=x.astype('float32') / 255.
            print(x.shape)
            y=None
            
        
        
        elif(db=='person'):
            x=np.load('/mnt/rds/redhen/gallina/home/hxn147/original_color_features/ideology_person_dataset.npy')
            x=x.reshape(-1,128,128,3)
            x=x.astype('float32') / 255.
            print(x.shape)
            y=None
            
        elif(db=='face'):
            x=np.load('/mnt/rds/redhen/gallina/home/hxn147/original_color_features/ideology_face_dataset.npy')
            x=x.reshape(-1,128,128,3)
            x=x.astype('float32') / 255.
            print(x.shape)
            
            y=None
            
        init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')  # sqrt(1./fan_in)

        # Training
        
        results = np.zeros(shape=[2, trials, 5], dtype=float)  # init metrics before finetuning
        for i in range(trials):  # base
            t0 = time()

            model = ConvDEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10],
                            init=init)
            # model.compile(optimizer='adam', loss='kld')

            # whether to use pretrained ae weights
            if ae_weights_dir is None:
                model.pretrain(x, y, optimizer='adam', epochs=500,
                               save_dir='/mnt/rds/redhen/gallina/home/hxn147', verbose=verbose, aug_ae=aug_ae)
            else:
                model.autoencoder.load_weights(os.path.join(ae_weights_dir, db, 'trial%d' % i, 'ae_weights.h5'))
            t1 = time()

            # training
            y_pred=model.fit(x, y=y, save_dir='/mnt/rds/redhen/gallina/home/hxn147')

    print("")
    print("Y pred",len(y_pred))
    return y_pred


if __name__=="__main__":
    # Global experiment settings
    trials = 5
    verbose = 0
    
    
    print("RUNNING FOR THE NORMAL GRAYSCALE")
    dbs=['normal']
    
    y_pred=run_exp(dbs, aug_ae=False,
            expdir='',
            ae_weights_dir=None,
            verbose=verbose, trials=trials)
    
    np.save('results2/normal_grayscale.npy',y_pred)
    
    
    print("RUNNING FOR THE CROP GRAYSCALE")
    dbs=['crop']
    
    y_pred=run_exp(dbs, aug_ae=False,
            expdir='',
            ae_weights_dir=None,
            verbose=verbose, trials=trials)
    
    np.save('results2/crop_grayscale.npy',y_pred)
    
    
    
    
    
    print("RUNNING FOR THE PERSON GRAYSCALE")
    dbs=['person']
    y_pred=run_exp(dbs, aug_ae=False,
            expdir='',
            ae_weights_dir=None,
            verbose=verbose, trials=trials)
    
    np.save('results2/person_grayscale.npy',y_pred)
    
    
    print("RUNNING FOR THE FACE GRAYSCALE")
    dbs=['face']
    
    y_pred=run_exp(dbs, aug_ae=False,
            expdir='',
            ae_weights_dir=None,
            verbose=verbose, trials=trials)
    
    np.save('results2/face_grayscale.npy',y_pred)
    
    
    
    
    
    
    