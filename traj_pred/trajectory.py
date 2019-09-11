
import traj_pred.dcgm as dcgm
import traj_pred.lstm
import traj_pred.ball_model.diff_eq as ball_deq
import numpy as np
import os
import json
import scipy.stats as stats
import traj_pred.utils as utils
import time

traj_loaders = {'dcgm': dcgm.load_traj_model, 
        "lstm": traj_pred.lstm.load_traj_model, 
        "ball/diff_eq": ball_deq.load_traj_model}

def load_model(path):
    conf = json.load( open(os.path.join(path,'conf.json'), 'r') )
    if not 'model' in conf:
        raise Exception('The keyword "model" must be present in the configuration JSON file')
    model = conf['model']
    if not model in traj_loaders:
        raise Exception('Model {} not recognized'.format(model))
    return traj_loaders[model](path)

def traj_pred_error(model, Times, X, nobs, noise=0.0):
    """ Evaluate the prediction error given nobs observations

    For each of the trajectories passed, it computes the predictive distribution
    assuming only nobs obsverations are given, and evaluates a set of metrics 
    for each trajectory, returning a dictionary with following the keys:
    * distances: The distance between the ground truth and the prediction
    * c_marg: The log-likelihood of each observation marginally, i.e, P(y_t | y_{0:nobs})
      for all t.
    """
    distances = []
    c_marg = []
    latencies = []
    Sigma_y = noise**2 * np.eye(X[0].shape[1])
    for n, tabs in enumerate(Times):
        t = tabs - tabs[0]
        ts1 = time.time()
        pred_mean, pred_cov = model.traj_dist(t[0:nobs], X[n][0:nobs], t)
        ts2 = time.time()
        latencies.append(ts2-ts1)
        d = X[n] - pred_mean
        distances.append( np.linalg.norm(d,axis=1) )
        llh = [stats.multivariate_normal.logpdf(d[i], cov=(pred_cov[i] + Sigma_y)) for i in range(len(t))]
        c_marg.append(np.array(llh))
    return {'distances': distances, 'c_marg': c_marg, 'latencies': latencies}

def comp_traj_dist(Times, Xreal, length, deltaT=1.0/180.0):
    """ Computes the empirical distribution of a set of trajectories

    Given a set of trajectories, it computes its empirical mean and standard deviation.
    This method is useful for example to plot the error distribution of different
    prediction models. All the trajectories are cut to the specified length.
    """
    X, Xobs = utils.encode_fixed_dt(Times, Xreal, length, deltaT)
    sX = np.sum(X, axis=0)
    ssX = np.sum(X**2, axis=0)
    N = np.sum(Xobs, axis=0)
    means = sX / N
    var = ssX/N - means**2
    stds = np.sqrt(var)
    return means, stds
