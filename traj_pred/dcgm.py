
import numpy as np
import keras
import tensorflow as tf
import traj_pred.utils as utils
import json
import pickle
import os
import matplotlib.pyplot as plt
import time
import bisect
import logging

class Trajectory:
    """ Trajectory modeling with Deep Conditional Generative Model

    We assume that the model was already trained. This class can only 
    be used to make predictions.
    """

    def __init__(self, encoder, decoder, normalizer=None, samples=30, z_size=16, 
            length=200, deltaT=1.0/180.0, default_Sigma_y=1e2, partial_encoder=None):
        self.encoder = encoder
        self.partial_encoder = partial_encoder
        self.decoder = decoder
        self.normalizer = normalizer
        self.deltaT = deltaT
        self.length = length
        self.z_size = z_size
        self.samples = samples
        self.default_Sigma_y = default_Sigma_y

    def save(self, path):
        """ Saves to the file passed as argument
        """
        extra = {'model': 'dcgm',
                'deltaT': self.deltaT, 
                'samples': self.samples, 
                'default_Sigma_y': self.default_Sigma_y,
                'z_size': self.z_size,
                'length': self.length}
        if not os.path.exists(path):
            os.makedirs(path)
        self.encoder.save(os.path.join(path, 'encoder.h5'))
        self.decoder.save(os.path.join(path, 'decoder.h5'))
        if self.partial_encoder is not None:
            self.partial_encoder.save(os.path.join(path, 'partial_encoder.h5'))
        pickle.dump({'xscaler': self.normalizer}, open(os.path.join(path, 'norm.pickle'),'wb'))
        json.dump(extra, open(os.path.join(path, 'conf.json'), 'w'))

    def traj_llh(self, times, obs):
        #TODO: Not implemented yet, think well the math first
        #X, Xobs = utils.encode_fixed_dt([times],[obs], self.length,self.deltaT)
        return 0.0

    def traj_dist(self, prev_times, prev_obs, pred_times):
        batch_size = 1
        #1) First normalize and then encode. Very important!
        Xn, Xobs = utils.encode_fixed_dt([prev_times],[self.normalizer.transform(prev_obs)], 
                self.length, self.deltaT)
        #z = np.random.normal(loc=0.0, scale=1.0, size=(self.samples,batch_size,self.z_size))
        z = np.random.normal(loc=0.0, scale=1.0, size=(self.samples, self.z_size))
        if self.partial_encoder is not None:
            t1 = time.time()
            mu_z, log_sig_z = self.partial_encoder.predict([Xn, Xobs])
            t2 = time.time()
            logging.info("DCGM Encoding time: {}".format(t2-t1))
            sig_z = np.sqrt( np.exp(log_sig_z) )
            z = mu_z + z*sig_z
        Xn_rep = np.tile(Xn, (self.samples,1,1))
        Xobs_rep = np.tile(Xobs, (self.samples,1,1))
        t1 = time.time()
        y_n = self.decoder.predict([Xn_rep, Xobs_rep, z])
        t2 = time.time()
        logging.info("DCGM Decoding time: {}".format(t2-t1))
        y = utils.apply_scaler(self.normalizer.inverse_transform, y_n)
        ixs = [int(round((x - prev_times[0])/self.deltaT)) for x in pred_times]
        t1 = time.time()
        means_model, covs_model = utils.empirical_traj_dist(y)
        limit = bisect.bisect_left(ixs, self.length)
        means = means_model[ixs[0:limit]]
        covs = covs_model[ixs[0:limit]]
        if limit < len(ixs):
            missing = len(ixs) - limit
            means = np.concatenate( ( means, np.tile(means[-1], (missing, 1)) ), axis=0)
            covs = np.concatenate( ( covs, np.tile(self.default_Sigma_y*np.eye(y.shape[-1]), (missing,1,1)) ), axis=0)
        t2 = time.time()
        logging.info("DCGM Comp. distribution time: {}".format(t2-t1))
        return np.array(means), np.array(covs)

def load_traj_model(path):
    """ Loads trained trajectory model
    """
    extra = json.load( open(os.path.join(path,'conf.json'), 'r') )
    extra.setdefault('deltaT', 1.0/180.0)
    extra.setdefault('samples', 30)
    extra.setdefault('default_Sigma_y', 1e2)
    decoder = keras.models.load_model( os.path.join(path,'decoder.h5') )
    encoder = keras.models.load_model( os.path.join(path,'encoder.h5') )
    penc_fn = os.path.join(path,'partial_encoder.h5')
    if os.path.exists(penc_fn):
        partial_encoder = keras.models.load_model(penc_fn)
    else:
        partial_encoder = None
    norm = pickle.load(open(os.path.join(path,'norm.pickle'), 'rb'))
    return Trajectory(encoder, decoder, norm['xscaler'], samples=extra['samples'], 
            z_size=extra['z_size'], length=extra['length'], 
            deltaT=extra['deltaT'], 
            default_Sigma_y=extra['default_Sigma_y'],
            partial_encoder=partial_encoder)


class BatchDCGM(keras.utils.Sequence):
    """ Creates mini-batches for a deep conditional generative model for trajectories

    Given a sequence of pairs (time, X) and a particular deltaT and length, returns a sequence
    of tensors (X,Xobs,Y,Yobs) with the same time length and deltaT.
    """

    def __init__(self, batch_sampler, length, deltaT, fake_missing_p=0.0):
        self.batch_sampler = batch_sampler
        self.length = length
        self.deltaT = deltaT
        self.duration = length*deltaT
        self.fake_missing_p = fake_missing_p
        self.on_epoch_end()

    def on_epoch_end(self):
        self.batch_sampler.on_epoch_end()

    def __data_generation(self, times, X):
        assert( len(times) == len(X) )
        n_times, n_x = utils.shift_time_duration(times, X, self.duration) # arbitrary start point
        Y, Yobs = utils.encode_fixed_dt(n_times, n_x, self.length, self.deltaT)
       
        N,T,K = Y.shape        
        ts_lens = np.random.randint(low=0, high=T, size=N)
        is_obs = np.array([np.logical_and(np.arange(T) < x, np.random.rand(T)>=self.fake_missing_p) for x in ts_lens])
        Xobs = Yobs*is_obs.reshape((-1,T,1))
        X = Xobs*Y

        return X, Xobs, Y, Yobs

    def __len__(self):
        return len(self.batch_sampler)

    def __getitem__(self, index):
        times, X = self.batch_sampler[index]
        X, Xobs, Y, Yobs = self.__data_generation(times, X)
        Ymask = np.concatenate((Y,Yobs),axis=-1)
        return [X,Xobs,Y,Yobs], [Ymask]

def dcgm_loss(mu_z_c, log_sig_z_c, mu_z_p, log_sig_z_p, log_sig_y):
    def loss(y_mask, y_decoded_mean):
        y_mask_shape = keras.backend.shape(y_mask)
        batch_size = y_mask_shape[0]
        y = y_mask[:,:,0:-1]
        mask = y_mask[:,:,-1]

        # Compute Exp Likelihood
        sig_y = keras.backend.exp(log_sig_y)
        d = y - y_decoded_mean
        d_sq = keras.backend.square(d)
        d_mah = tf.divide(d_sq, sig_y)
        log_det_sig_y = keras.backend.sum(log_sig_y) #has to be a scalar
        d_mah_sum = keras.backend.sum(d_mah, axis=-1 ) + log_det_sig_y
        d_sq_masked = d_mah_sum * mask

        # Compute KL divergence
        sig_z = keras.backend.exp(log_sig_z_c)
        sig_z_p = keras.backend.exp(log_sig_z_p)
        g_dist = (sig_z + keras.backend.square(mu_z_c - mu_z_p))/sig_z_p
        kl_terms = log_sig_z_p - log_sig_z_c + g_dist - 1
        kl = 0.5 * keras.backend.sum(kl_terms)
        rec_loss = 0.5 * keras.backend.sum( d_sq_masked )
        tot_loss = rec_loss + kl
        return tot_loss / tf.to_float(batch_size)
    return loss

class TrajDCGM:
    """ A deep conditional generative model for trajectory generation
    """

    def __build_graph(self, encoder, partial_encoder, cond_generator, 
            log_sig_y, length, D, z_size):
        self.encoder = encoder
        self.partial_encoder = partial_encoder
        self.cond_generator = cond_generator
        self.log_sig_y = log_sig_y
        x = keras.layers.Input(shape=(length,D), name='x_in')
        x_obs = keras.layers.Input(shape=(length,1), name='x_obs_in')
        y = keras.layers.Input(shape=(length,D), name='y_in')
        y_obs = keras.layers.Input(shape=(length,1), name='y_obs_in')
        
        #self.z_in = keras.layers.Input(shape=(z_size,))
        mu_z, log_sig_z = encoder([y,y_obs])
        mu_z_partial, log_sig_z_partial = partial_encoder([x,x_obs])

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = keras.backend.random_normal(shape=keras.backend.shape(z_mean))
            return z_mean + tf.sqrt(tf.exp(z_log_sigma)) * epsilon

        z_sampler = keras.layers.Lambda(sampling, output_shape=(z_size,))
        z = z_sampler([mu_z, log_sig_z])
        y_pred = cond_generator([x,x_obs,z])

        self.full_tree = keras.models.Model(inputs=[x,x_obs,y,y_obs], outputs=[y_pred])
        my_loss = dcgm_loss(mu_z, log_sig_z, mu_z_partial, log_sig_z_partial, log_sig_y)
        self.full_tree.compile(optimizer='adam', loss=my_loss)

    def fit_generator(self, generator, validation_data, epochs, use_multiprocessing, workers, callbacks):
        self.full_tree.fit_generator(generator=generator, validation_data=validation_data, epochs=epochs,
                use_multiprocessing=use_multiprocessing, workers=workers, callbacks=callbacks)

    def __init__(self, encoder, partial_encoder, cond_generator, log_sig_y, length, D, z_size):
        """ Constructs a Trajectory Deep Conditional Model

        Parameters
        ----------

        encoder : Model with inputs=[y,y_obs] and outputs=[mu_z, log_sig_z]
        partial_encoder : Model with inputs=[x,x_obs] and outputs=[mu_z, log_sig_z]
        cond_generator : Model with inputs=[x,x_obs,z] and output=[y]
        log_sig_y : Variable representing the sensor noise. If it is trainable, it will be optimized.
        length : Maximum number of time samples of each trajectory
        D : Dimensionality of the observations
        z_size : Dimensionality of the hidden state
        """
        self.__build_graph(encoder, partial_encoder, cond_generator, log_sig_y, length, D, z_size)



