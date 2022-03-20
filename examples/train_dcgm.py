""" Trains deep conditional generative model
"""

import argparse
import os
import json
import time
import math

import numpy as np
import sklearn.preprocessing as prep
import traj_pred.utils as utils
import traj_pred.dcgm as dcgm
from tensorflow import keras


def dense_network(input_shape, output_shape, 
        hidden_layers=[(512, 'relu')], 
        output_activation = 'relu'):
    input_size = np.prod(input_shape)
    output_size = np.prod(output_shape)
    encoder = keras.models.Sequential()
    encoder.add( keras.layers.Reshape( (input_size,) ) )
    for hl in hidden_layers:
        encoder.add( keras.layers.Dense(hl[0], activation=hl[1]) )
    encoder.add( keras.layers.Dense(output_size, activation=output_activation) )
    encoder.add( keras.layers.Reshape( output_shape ) )
    return encoder

def create_encoder(length, D, z_size):
    y = keras.layers.Input(shape=(length,D), name='y_input')
    y_obs = keras.layers.Input(shape=(length,1), name='y_obs_input')
    
    h_size = 128
    y_conc = keras.layers.concatenate([y, y_obs])
    fn = dense_network((length,D+1), (h_size,), hidden_layers=[(512,'relu')], output_activation='relu')
    h = fn(y_conc)
    mu = keras.layers.Dense(z_size)(h)
    log_sig = keras.layers.Dense(z_size)(h)
    return keras.models.Model(inputs=[y,y_obs], outputs=[mu, log_sig])

def create_cond_gen(length, D, z_size):
    x = keras.layers.Input(shape=(length,D), name='x_input')
    x_obs = keras.layers.Input(shape=(length,1), name='x_obs_input')
    z = keras.layers.Input(shape=(z_size,), name='z_input')
    
    h_size = 128
    x_conc = keras.layers.concatenate([x, x_obs])
    fn = dense_network((length,D+1), (h_size,), hidden_layers=[(512,'relu')], output_activation='relu')
    h = fn(x_conc)

    tmp = keras.layers.concatenate([h, z])
    fn_2  = dense_network((h_size+z_size,), (length,D), hidden_layers=[(1024,'relu')], output_activation=None)
    mu = fn_2(tmp)
    log_sig = fn_2.layers[-1].add_weight(name='noise', shape=(D,), initializer='zeros', trainable=True)
    return keras.models.Model(inputs=[x,x_obs,z], outputs=[mu]), log_sig

def train_ball_dcgm(X, Times, Xval, Time_v, length, deltaT, z_size=64, batch_size=64, epochs=100):
    x_scaler = utils.train_std_scaler(X)
    x_transform = lambda x: x_scaler.transform( utils.transform_ball_traj(x,(-45,45),((-0.6,-0.8,0.0),(0.6,0.8,0.0))) )
    N = len(X)
    D = len(X[0][0])

    my_mb = utils.TrajMiniBatch(Times, X, batch_size=batch_size, ds_mult=8,
            shuffle=True, x_transform=x_transform)
    dcgm_mb = dcgm.BatchDCGM(my_mb, length, deltaT, fake_missing_p=args.fake_missing_p)

    my_mb_val = utils.TrajMiniBatch(Time_v, Xval, batch_size=batch_size, ds_mult=16,
            shuffle=True, x_transform=x_transform)
    dcgm_mb_val = dcgm.BatchDCGM(my_mb_val, length, deltaT, fake_missing_p=args.fake_missing_p)

    
    encoder = create_encoder(length, D, z_size)
    partial_encoder = encoder
    cond_generator, log_sig_y_trainable = create_cond_gen(length, D, z_size)
    log_sig_y_no_trainable = keras.backend.variable(np.zeros(D))
    log_sig_y = log_sig_y_trainable
    model = dcgm.TrajDCGM(encoder=encoder, partial_encoder=partial_encoder, cond_generator=cond_generator, log_sig_y=log_sig_y, length=length, D=D, z_size=z_size)
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=5, verbose=1, min_lr=1e-7)
    early_st = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, verbose=1, patience=10, mode='auto')
    #model_cp = keras.callbacks.ModelCheckpoint("/tmp/weights-{epoch:02d}.h5", monitor='val_loss', save_best_only=True, save_weights_only=True, mode="auto")
    callback_list = [reduce_lr, early_st]
    model.fit_generator(dcgm_mb, dcgm_mb_val, epochs=epochs, use_multiprocessing=True, workers=8,
            callbacks=callback_list)
    print("Finishing training")

    model_pred = dcgm.Trajectory(encoder, cond_generator, normalizer=x_scaler, samples=30,
            z_size=z_size, length=length, deltaT=deltaT, default_Sigma_y=1.0, partial_encoder=partial_encoder)

    return model, model_pred, log_sig_y


def main(args):
    with open(args.training_data,'rb') as f:
        data = np.load(f, encoding='latin1')
        Times = data['Times']
        X = data['X']
    N = len(Times)
    
    # Permute
    perm = np.random.permutation(N)
    X = X[perm]
    Times = Times[perm]

    nval = int(round(N*args.p))
    ntrain = N-nval
    Xt = X[0:ntrain]
    Time_t = Times[0:ntrain]
    Xval = X[ntrain:]
    Time_val = Times[ntrain:]

    model, t_pred, log_sig_y = train_ball_dcgm(Xt, Time_t, Xval, Time_val, length=args.length, deltaT=args.dt, batch_size=args.batch_size, epochs=args.epochs)
    print("Model trained")
    t_pred.save(args.model)
    print("Model saved")

    np_l_sy = keras.backend.eval(log_sig_y)
    noise_y = np.exp(np_l_sy)
    print(noise_y)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('training_data', help="File with the stored training trajectories")
    parser.add_argument('model', help="Path where the resulting model is saved")
    parser.add_argument('--p', type=float, default=0.1, help="Percentage of the trajectories used for validation")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--dt', type=float, default=1.0/180.0, help="Delta Time")
    parser.add_argument('--length', type=int, default=200, help="Length of the time series to model")
    parser.add_argument('--batch_size', type=int, default=256, help="Size of each minibatch")
    parser.add_argument('--fake_missing_p', type=float, default=0.0, help="Fake probability of missing observations")
    args = parser.parse_args()
    main(args)
