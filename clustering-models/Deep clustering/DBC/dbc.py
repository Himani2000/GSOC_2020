from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, InputLayer
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.manifold import TSNE
from keras.datasets import mnist
import numpy as np
from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import os
import metrics
#import image_utils
from keras.datasets import mnist
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.image import img_to_array, array_to_img
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Model

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE



#from dec import DEC

def load_stl(data_path='/mnt/rds/redhen/gallina/home/hxn147/data/stl/stl10_binary/'):
    y1 = np.fromfile(data_path + '/train_y.bin', dtype=np.uint8) - 1
    y2 = np.fromfile(data_path + '/test_y.bin', dtype=np.uint8) - 1
    y = np.concatenate((y1, y2))
    x1 = np.fromfile(data_path + '/train_X.bin', dtype=np.uint8)
    x1 = x1.reshape((int(x1.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
    x2 = np.fromfile(data_path + '/test_X.bin', dtype=np.uint8)
    x2 = x2.reshape((int(x2.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
    x = np.concatenate((x1, x2)).astype(float)
    return x/255.0,y



def CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    model.add(InputLayer(input_shape))
    model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1'))

    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(Dense(units=filters[3], name='embedding'))
    model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))

    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    encoder = Model(inputs=model.input, outputs=model.get_layer('embedding').output)
    return model, encoder

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ConvDEC(object):
    def __init__(self,
                 input_shape,
                 filters=[32, 64, 128, 10],n_clusters=10,alpha=1.0,
                 init='glorot_uniform'):
        super(ConvDEC, self).__init__()

        # self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=10)
        self.datagenx = ImageDataGenerator()
        self.autoencoder, self.encoder = CAE(input_shape, filters)
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

        
        
        
        
    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, save_dir='/mnt/rds/redhen/gallina/home/hxn147/deep_clustering_results'):
        print('...Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')
        csv_logger = callbacks.CSVLogger('/mnt/rds/redhen/gallina/home/hxn147/deep_clustering_results/pretrain_log.csv')
        cb = [csv_logger]

        # begin pretraining
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs,callbacks=cb)
        print('Pretraining time: %ds' % round(time() - t0))
        #self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        #print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='./results/temp'):

        print('Update interval', update_interval)
        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        print('Save interval', save_interval)

        # Step 1: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 2: deep clustering
        # logging file
        import csv
        #logfile = open(save_dir + '/dec_log.csv', 'w')
        #logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        #logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    #logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=loss)
                    #logwriter.writerow(logdict)
                    #print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    #logfile.close()
                    break

            # train on batch
            # if index == 0:
            #     np.random.shuffle(index_array)
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            # save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                #self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        #logfile.close()
        #print('saving model to:', save_dir + '/DEC_model_final.h5')
        #self.model.save_weights(save_dir + '/DEC_model_final.h5')

        return y_pred
def autoencoder_fit(autoencoder,encoder,data):
        #autoencoder=single_layer_autoencoder(encoding_dim,input_dimension)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        autoencoder.fit(data,data,epochs=100,batch_size=128,shuffle=True)
        decoded_imgs = autoencoder.predict(data)
        encoded_embedding=encoder.predict(data)

        #visualize(data,decoded_imgs)
        return encoded_embedding,decoded_imgs


if __name__ == "__main__":
    init = 'glorot_uniform'
    pretrain_optimizer = 'adam'
  

    update_interval = 140
    #pretrain_epochs = 300
    maxiter=2e4
    #to be removed 
    pretrain_epochs=100
    init = VarianceScaling(scale=1. / 3., mode='fan_in',distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
    pretrain_optimizer = SGD(lr=1, momentum=0.9)
    
    update_interval = 30
    pretrain_epochs = 100
    #n_clusters=10
    y=None
    
    
    
    #here the code for the dimensionality reduction will be written 
    
    base_path='/mnt/rds/redhen/gallina/home/hxn147/original_color_features'
    for file in os.listdir(base_path):
        print("[RUNNING] FOR THE FILE",file)
        
        x=np.load(os.path.join(base_path,file))
        x=x.reshape(-1,128,128,3)
        x=x.astype('float32') / 255.

        model,encoder= CAE(input_shape=x.shape[1:], filters=[32, 64, 128, 10])
        encoded_embedding,decoded_imgs=autoencoder_fit(model,encoder,x)
        print("THE SHAPE OF THE ENCODED EMBEDDING IS  ",encoded_embedding.shape)
        tsne_transformed = TSNE(n_components=2).fit_transform(encoded_embedding)
        db=DBSCAN(min_samples=2)
        y_pred=db.fit_predict(tsne_transformed)
        n_clusters=len(set(y_pred))

        print("THE NUMBER OF CLUSTERS ARE ",n_clusters)

        model = ConvDEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10],n_clusters=n_clusters,init=init)
        model.pretrain(x=x, y=y, optimizer=pretrain_optimizer,epochs=pretrain_epochs, batch_size=256,save_dir='')
        print("DONE pretraining")
        model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
        y_pred = model.fit(x, y=y, tol=0.001, maxiter=2, batch_size=256,update_interval=update_interval, save_dir='')
        np.save(os.path.join('results3',file),y_pred)
        print("[SAVING] the results")
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
   
    #x,y=load_stl()
    #print("X shape",x.shape)
    #print("Y shape",y.shape)
    """
    y=None
    print("RUNNING FOR THE NORMAL GRAYSCALE")
    x=np.load('/mnt/rds/redhen/gallina/home/hxn147/ideology_dataset_normal_features_conventional2/ideology__grayscale_.npy')
    x=x.reshape(-1,128,128,1)
    x=x.astype('float32') / 255.
    print(x.shape)
    
    
    
    n_clusters=444
    model = ConvDEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10],n_clusters=n_clusters,init=init)
    model.pretrain(x=x, y=y, optimizer=pretrain_optimizer,epochs=pretrain_epochs, batch_size=256,save_dir='')
    print("DONE pretraining")
    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    y_pred = model.fit(x, y=y, tol=0.001, maxiter=2, batch_size=256,update_interval=update_interval, save_dir='')
    np.save("results2/normal_grayscale_optics_tsne.npy",y_pred)
    print("Saving the results ...")
    
    
    
    
    y=None
    
    print("RUNNING FOR THE CROP GRAYSCALE")
    x=np.load('/mnt/rds/redhen/gallina/home/hxn147/ideology_dataset_crop_features_conventional2/ideology_crop__grayscale_.npy')
    x=x.reshape(-1,128,128,1)
    x=x.astype('float32') / 255.
    print(x.shape)
    
    n_clusters=447
    model = ConvDEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10],n_clusters=n_clusters,init=init)
    model.pretrain(x=x, y=y, optimizer=pretrain_optimizer,epochs=pretrain_epochs, batch_size=256,save_dir='')
    print("DONE pretraining")
    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    y_pred = model.fit(x, y=y, tol=0.001, maxiter=2, batch_size=256,update_interval=update_interval, save_dir='')
    np.save("results2/crop_grayscale_optics_tsne.npy",y_pred)
    print("Saving the results ...")
    
    
    
    print("RUNNING FOR THE PERSON GRAYSCALE")
    x=np.load('/mnt/rds/redhen/gallina/home/hxn147/ideology_person_dataset_features_conventional/ideology__grayscale_.npy')
    x=x.reshape(-1,128,128,1)
    x=x.astype('float32') / 255.
    
    
    n_clusters=403
    model = ConvDEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10],n_clusters=n_clusters,init=init)
    model.pretrain(x=x, y=y, optimizer=pretrain_optimizer,epochs=pretrain_epochs, batch_size=256,save_dir='')
    print("DONE pretraining")
    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    y_pred = model.fit(x, y=y, tol=0.001, maxiter=2, batch_size=256,update_interval=update_interval, save_dir='')
    np.save("results2/person_grayscale_optics_tsne.npy",y_pred)
    print("Saving the results ...")
    
    
    
    
    
    print("RUNNING FOR THE FACE GRAYSCALE")
    x=np.load('/mnt/rds/redhen/gallina/home/hxn147/ideology_face_dataset_features_conventional/ideology__grayscale_.npy')
    x=x.reshape(-1,128,128,1)
    x=x.astype('float32') / 255.
    
    
    
    n_clusters=426
    model = ConvDEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10],n_clusters=n_clusters,init=init)
    model.pretrain(x=x, y=y, optimizer=pretrain_optimizer,epochs=pretrain_epochs, batch_size=256,save_dir='')
    print("DONE pretraining")
    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    y_pred = model.fit(x, y=y, tol=0.001, maxiter=2, batch_size=256,update_interval=update_interval, save_dir='')
    np.save("results2/face_grayscale_optics_tsne.npy",y_pred)
    print("Saving the results ...")
    """
    
    
    
    
    
    
   

    
    
    
    
    
    
    
    
    
    