
import numpy as np
import traj_pred.utils as utils
from sklearn.preprocessing import PolynomialFeatures
import json
import os
import copy

class BallInitState:
    """ Estimates the initial state of the ball trajectory
    """

    def __init__(self, poly_deg, sig_w=1e2, noise=1e-2):
        self.poly_deg = poly_deg
        mu_w = np.zeros(poly_deg+1)
        Sig_w = np.eye(poly_deg+1)*sig_w
        self.state = []
        for d in range(3):
            st_d = utils.BayesianLinearReg(mu_w, Sig_w, noise)
            self.state.append(st_d)

    def fit(self, time, obs):
        poly = PolynomialFeatures(degree=self.poly_deg)
        phi = poly.fit_transform(time.reshape(-1,1))
        for d in range(3):
            self.state[d].fit(phi, obs[:,d])

    def sample(self, n):
        w = np.array([np.random.multivariate_normal(x.get_params()['mean'], 
            x.get_params()['cov'], size=n) for x in self.state]) # (D,N,K), K poly deg
        pos = w[:,:,0].T # (N,D)
        vel = w[:,:,1].T # (N,D)
        return pos, vel


class BallTrajectory:
    """ Ball trajectory modeling with differential equations

    We assume that the model was already trained. This class can only 
    be used to make predictions.
    """

    def __init__(self, air_drag, bounce_fac, max_bounces, 
            init_state_dist, deltaT, bounce_z=0.0,
            filter_window=50, n_samples=30):
        self.air_drag = air_drag
        self.bounce_fac = bounce_fac
        self.max_bounces = max_bounces
        self.filter_window = filter_window
        self.init_state_dist = init_state_dist
        self.deltaT = deltaT
        self.n_samples = 30
        self.bounce_z = bounce_z

    def get_traj_sample(self, x0, xd0, init_time, pred_times, deltaT=None):
        if deltaT is None: deltaT = self.deltaT
        x = x0
        xd = xd0
        t = init_time
        n = 0
        ans = []
        n_bounces = 0
        while n < len(pred_times):
            assert(t - self.deltaT < pred_times[n])
            if 2*abs(t - pred_times[n]) < self.deltaT:
                ans.append(x)
                n += 1
            t += self.deltaT
            xdd = -self.air_drag*np.linalg.norm(xd)*xd + np.array([0,0,-9.8])
            x = x + xd*self.deltaT + 0.5*xdd*self.deltaT**2
            xd = xd + self.deltaT*xdd
            if x[2] < self.bounce_z and n_bounces<self.max_bounces:
                xd[2] *= -1
                xd *= self.bounce_fac
                x[2] = 2*self.bounce_z - x[2]
                n_bounces += 1
        return ans

    def traj_dist(self, prev_times, prev_obs, pred_times):
        """ Find the initial state distribution and sample from it
        """
        if len(prev_times) > self.filter_window:
            #N = len(prev_times)
            prev_times = prev_times[:self.filter_window]
            prev_obs = prev_obs[:self.filter_window]
        t0 = prev_times[0]
        pt = prev_times - t0
        init_state = copy.deepcopy(self.init_state_dist)
        init_state.fit(pt, prev_obs)
        pos, vel = init_state.sample(self.n_samples)
        samples = np.array([self.get_traj_sample(pos[n], vel[n], 0.0, pred_times - t0) for n in range(self.n_samples)])
        return utils.empirical_traj_dist(samples)

def load_traj_model(path):
    """ Loads trained trajectory model
    """
    extra = json.load( open(os.path.join(path,'conf.json'), 'r') )
    extra.setdefault('deltaT', 1.0/180.0)
    extra.setdefault('samples', 30)
    return BallTrajectory(air_drag=extra['air_drag'], 
            bounce_fac=extra['bounce_fac'], 
            max_bounces=extra['max_bounces'], 
            init_state_dist = BallInitState(
                poly_deg = extra['init_state']['poly_deg'],
                sig_w = extra['init_state']['sig_w'],
                noise = extra['init_state']['noise']
                ), 
            deltaT = extra['deltaT'], 
            bounce_z = extra['bounce_z'],
            filter_window = extra['filter_window'], 
            n_samples = extra['samples'])
