
import numpy as np
import keras
import tensorflow as tf
import traj_pred.utils as utils
import json
import pickle
import os
import matplotlib.pyplot as plt
import time

class BatchLSTM(keras.utils.Sequence):
    """ Creates mini-batches for an LSTM for trajectories

    Given a sequence of pairs (time, X) and a particular deltaT and length, returns a sequence
    of tensors ((X,Xobs),(Y,Yobs)) with the same time length and deltaT.
    """

    def __init__(self, batch_sampler, length, deltaT):
        self.batch_sampler = batch_sampler
        self.length = length
        self.deltaT = deltaT
        self.on_epoch_end()

    def on_epoch_end(self):
        self.batch_sampler.on_epoch_end()

    def __data_generation(self, times, X):
        assert( len(times) == len(X) )
        n_times, n_x = utils.shift_time(times, X, self.length)
        Y, Yobs = utils.encode_fixed_dt(n_times, n_x, self.length, self.deltaT)
        return Y, Yobs

    def __len__(self):
        return len(self.batch_sampler)

    def __getitem__(self, index):
        times, X = self.batch_sampler[index]
        Y, Yobs = self.__data_generation(times, X)
        Ymask = np.concatenate((Y,Yobs),axis=-1)
        return [ [Y[:,0:-1,:], Yobs[:,0:-1,:]], Ymask[:,1:,:] ]

def lstm_loss(log_sig_y=None):
    def loss(y_mask, y_decoded_mean):
        y_mask_shape = keras.backend.shape(y_mask)
        batch_size = y_mask_shape[0]
        y = y_mask[:,:,0:-1]
        mask = y_mask[:,:,-1]

        # Compute Exp Likelihood
        d = y - y_decoded_mean
        d_sq = keras.backend.square(d)
        d_sum = keras.backend.sum(d_sq, axis=-1 )
        d_sq_masked = d_sum * mask
        rec_loss = 0.5 * keras.backend.sum( d_sq_masked )
        tot_loss = rec_loss
        return tot_loss / tf.to_float(batch_size)
    return loss

class TrajectoryLSTM:
    """ Trajectory modeling with LSTM

    We assume that the model was already trained. This class can only 
    be used to make predictions.
    """

    def __build_graph(self, model):
        self.model = model
        self.model.compile(optimizer='adam', loss=lstm_loss())

    def __init__(self, model, normalizer=None, samples=30, length=200, deltaT=1.0/180.0,
            default_Sigma_y=1e1, Sigma_y=None):
        self.__build_graph(model)
        self.normalizer = normalizer
        self.samples = samples
        self.deltaT = deltaT
        self.length = length
        self.default_Sigma_y = default_Sigma_y
        self.Sigma_y = Sigma_y

    def traj_dist(self, prev_times, prev_obs, pred_times, recursive=True):
        batch_size = 1
        D = prev_obs[0].shape[-1]
        #1) First normalize and then encode. Very important!
        curr_ix = int( round((prev_times[-1]-prev_times[0]) / self.deltaT) )
        t1 = time.time()
        pred_time = 0.0

        Sigma_y = self.Sigma_y if self.Sigma_y is not None else 0.01*np.eye(D)
        noise = np.random.multivariate_normal(mean=np.zeros(D), cov=Sigma_y, size=(self.samples,self.length-1))
        Xn, Xobs = utils.encode_fixed_dt([prev_times],[self.normalizer.transform(prev_obs)], 
            self.length-1, self.deltaT)
        if recursive:
            Xsamples = np.repeat(Xn, repeats=self.samples, axis=0)
            Xobs_samples = np.repeat(Xobs, repeats=self.samples, axis=0)
            y_n = None
            for i in range(curr_ix, self.length-1):
                t2 = time.time()
                y_n = self.model.predict([Xsamples,Xobs_samples]) + noise
                t3 = time.time()
                pred_time += t3-t2
                if i+1 < self.length-1:
                    Xsamples[:,i+1] = y_n[:,i]
                    Xobs_samples[:,i+1] = 1.0
            if y_n is None:
                y_n = self.model.predict([Xsamples,Xobs_samples]) + noise
            y = utils.apply_scaler(self.normalizer.inverse_transform, y_n)
        else:
            y_n = self.model.predict([Xn, Xobs]) + noise
            y = utils.apply_scaler(self.normalizer.inverse_transform, y_n)

        ixs = [int(round((x - prev_times[0])/self.deltaT))-1 for x in pred_times]
        means = [] 
        covs = []
        for i in ixs:
            if i < self.length-1:
                y_mu = np.mean(y[:,i,:], axis=0)
                y_Sigma = np.cov(y[:,i,:], rowvar=False)
                bias_scale = self.samples/(self.samples-1)
                y_Sigma = bias_scale*y_Sigma
            else:
                y_mu = np.mean(y[:,-1,:], axis=0)
                y_Sigma = self.default_Sigma_y*np.eye(y.shape[-1])
            means.append(y_mu)
            covs.append(y_Sigma)
        t4 = time.time()
        print("Total pred time: {}, Computing the LSTM prediction: {}, Avg. LSTM pred: {}".format(
            t4-t1, pred_time, pred_time / (self.length-curr_ix-1)))
        return np.array(means), np.array(covs)


    def fit_generator(self, generator, validation_data, epochs, use_multiprocessing, workers, callbacks):
        self.model.fit_generator(generator=generator, validation_data=validation_data, epochs=epochs,
                use_multiprocessing=use_multiprocessing, workers=workers, callbacks=callbacks)

        #Compute now Sigma_y
        x,y_masked = generator[0]
        y = y_masked[:,:,0:-1]
        mask = y_masked[:,:,-1]

        y_pred = self.model.predict(x)
        d = y-y_pred
        d_flat = d.reshape((-1,3))
        #TODO: Need to use mask for Sigma_y
        self.Sigma_y = np.cov(d_flat.T)

    def save(self, path):
        """ Saves to the file passed as argument
        """
        extra = {'model': 'lstm',
                'deltaT': self.deltaT, 
                'samples': self.samples, 
                'default_Sigma_y': self.default_Sigma_y,
                'Sigma_y': self.Sigma_y.tolist(),
                #'z_size': self.z_size,
                'length': self.length}
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, 'lstm.h5'))
        pickle.dump({'xscaler': self.normalizer}, open(os.path.join(path, 'norm.pickle'),'wb'))
        json.dump(extra, open(os.path.join(path, 'conf.json'), 'w'))

def load_traj_model(path):
    """ Loads trained trajectory model
    """
    extra = json.load( open(os.path.join(path,'conf.json'), 'r') )
    extra.setdefault('deltaT', 1.0/180.0)
    extra.setdefault('samples', 30)
    extra.setdefault('default_Sigma_y', 1e1)
    extra.setdefault('Sigma_y', None)
    model = keras.models.load_model( os.path.join(path,'lstm.h5'), custom_objects={'loss': lstm_loss()} )
    norm = pickle.load(open(os.path.join(path,'norm.pickle'), 'rb'))
    return TrajectoryLSTM(model, norm['xscaler'], samples=extra['samples'], 
            length=extra['length'], 
            deltaT=extra['deltaT'],
            default_Sigma_y=extra['default_Sigma_y'],
            Sigma_y=extra['Sigma_y'])
            #partial_encoder=partial_encoder)
