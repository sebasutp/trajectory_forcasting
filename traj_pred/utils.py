
import numpy as np
from tensorflow import keras
import sklearn.preprocessing as prep
import math
import bisect

def rotate_z(x, th, center):
    rm = np.array([[math.cos(th), -math.sin(th), 0],[math.sin(th), math.cos(th),0],[0,0,1]])
    xt = [np.dot(rm, (y - center)) + center for y in x]
    return np.array(xt)

def transform_ball_traj(x, rotation_range, translation_range):
    rot = np.random.uniform(low=rotation_range[0], high=rotation_range[1])
    trans = np.random.uniform(low=translation_range[0], high=translation_range[1])
    th = rot*math.acos(-1)/180.0

    center = x[len(x) // 2]
    #print("Center: {}, Theta: {}, Trans: {}".format(center,th,trans))
    xt = rotate_z(x, th, center) + trans
    #print(xt)
    return xt

def shift_time_length(Times, X, length):
    """ Shifts time based on length
    """
    nTimes = []
    nX = []
    for i, t in enumerate(Times):
        offset = len(t) - length
        if offset > 0:
            start_ix = np.random.randint(low=0, high=offset)
        else:
            start_ix = 0
        nTimes.append(t[start_ix:])
        nX.append(X[i][start_ix:])
    return nTimes, nX

def shift_time_duration(Times, X, duration):
    """ Shifts time based on duration
    """
    nTimes = []
    nX = []
    for i, t in enumerate(Times):
        T = t[-1] - t[0]
        t0_offset = T - duration
        if t0_offset > 0:
            t0 = np.random.rand() * t0_offset
            start_ix = bisect.bisect_left(t - t[0], t0)
        else:
            start_ix = 0
        nTimes.append(t[start_ix:])
        nX.append(X[i][start_ix:])
    return nTimes, nX


def encode_fixed_dt(Times, Xreal, length, deltaT=1.0/180.0):
    """ Encode time series as with fixed time and missing observations

    Assume a time series with an appoximate fixed deltaT is measured, 
    but we have observations with their time stamps and possibly some 
    observations are missing. This function returns an array of size
    (N, length, D), where N is the number of trajectories, length is
    an argument passed representing the length of each trajectory, and
    D is the dimensionality of each observation.
    """

    N = len(Times)
    assert(N>0 and len(Xreal) == N)
    if len(Xreal[0].shape) == 1:
        D = 1
    else:
        D = len(Xreal[0][0])

    X = np.zeros((N,length,D))
    Xobs = np.zeros((N,length,1))
    for n in range(N):
        assert(len(Xreal[n]) == len(Times[n]))
        X[n,0] = Xreal[n][0]
        Xobs[n,0] = 1
        Tn = len(Xreal[n])
        t_ix = -1
        for i in range(1,Tn):
            if Times[n][i] < Times[n][i-1]:
                continue
            t = Times[n][i] - Times[n][0]
            t_ix = int(round(t / deltaT))
            if t_ix >= length:
                break
            X[n,t_ix] = Xreal[n][i]
            Xobs[n,t_ix] = 1
    return X, Xobs

def cov_to_std(covs):
    """ Changes a tensor of covariance matrices to standard deviations

    Given an array X[...,d,d] representing covariance matrices, returns an array
    Y[...,d] representing the standard deviations (diagonal) corresponding to X.
    """
    var = np.diagonal(covs, axis1=-2,axis2=-1)
    std = np.sqrt(var)
    return std

def empirical_traj_dist(samples):
    """ Creates a mean and cov array for the Gaussian trajectory distribution from traj samples

    samples : A (N,T,D) array with N samples, T measurements per time series, and D dimensions 
    per observation

    returns a (T,D) mean array and (T,D,D) covariance array
    """
    means = np.mean(samples, axis=0)
    diff = samples - means
    covs = np.matmul(
            np.transpose(diff, (1, 2, 0)),
            np.transpose(diff, (1, 0, 2))
            ) / len(samples)
    return means, covs

def apply_scaler(f, tensor):
    """ Apply the scaler function to the tensor
    """
    shape = np.shape(tensor)
    tr = tensor.reshape(-1,shape[-1])
    st = f(tr)
    return st.reshape(shape)

def train_std_scaler(X):
    xscaler = prep.StandardScaler()
    fX = []
    for x in X: 
        fX.extend(x)
    xscaler.fit(fX)
    return xscaler

class TrajMiniBatch(keras.utils.Sequence):
    """ Generates random mini-batches from the trajectory training set

    Given a set of times and observations, this class produces random mini-batches
    applying possibly a transformation to the data passed as parameter.
    """

    def __init__(self, Times, X, batch_size=32, ds_mult=16, shuffle=True, x_transform=lambda x: x,
            t_transform=lambda t: t - t[0]):
        self.X = X
        self.Times = Times
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ds_mult = ds_mult
        self.x_transform = x_transform
        self.t_transform = t_transform
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.repeat(np.arange(len(self.X)), self.ds_mult)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        x = np.floor(len(self.indexes) / self.batch_size)
        return int(x)

    def __getitem__(self, index):
        list_ids = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = [self.x_transform(self.X[i]) for i in list_ids]
        Times = [self.t_transform(self.Times[i]) for i in list_ids]
        return (Times, X)

class BayesianLinearReg:
    """ Linear regression with Gaussian Distributed weights
    """

    def __init__(self, init_mean, init_cov, noise):
        self.mean = init_mean
        self.cov = init_cov
        self.noise = noise
        self.n_features = len(init_mean)

    def fit(self, Phi, y):
        """ Fit weights so Phi*w = y
        """
        inv_Sw = np.linalg.inv(self.cov)
        Sig_w = np.linalg.inv(inv_Sw + np.dot(Phi.T, Phi)/self.noise**2 )
        mu_w = np.dot( Sig_w, (
            np.dot(Phi.T,y)/self.noise**2 + np.dot(inv_Sw, self.mean)
            ))
        self.mean = mu_w
        self.cov = Sig_w

    def get_params(self):
        return {'mean': self.mean, 'cov': self.cov, 'noise': self.noise}
