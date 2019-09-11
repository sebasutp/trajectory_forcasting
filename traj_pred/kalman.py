""" My own implementation of Kalman Filters
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sum_all_elems(L, dims):
    if (dims == 1):
        return sum(L)
    return sum(map(lambda x: sum_all_elems(x,dims-1),L))

def my_num_grad(myobj, params, eps):
    n_params = len(params)
    ans = np.zeros(n_params)
    for d in xrange(n_params):
        new_params = np.copy(params)
        new_params[d] += eps
        ans[d] = (myobj(new_params) - myobj(params)) / eps
    return ans

def get_cov_prior(std_diag, v):
    N = v + len(std_diag) + 1
    return {'v': v, 'invS0': N*np.diag(map(lambda x: x**2, std_diag))}


class ConstMat:
    def __init__(self, mat):
        self.__mat = mat
        self.__params = 0

    @property
    def params(self):
        return self.__params

    @params.setter
    def params(self, params):
        self.__params = params

    def mat(self, state, params=None, **opt_pars):
        return self.__mat

    def deriv(self, state, params=None, **opt_pars):
        raise NotImplementedError

    def shape(self):
        return np.shape(self.__mat)

    def n_params(self):
        return 1

class LinearKF:
    #Sigma: Covariance of the observations
    #Gamma: Covariance of the hidden state
    #mu0: Initial state mean
    #P0: Initial state Covariance
    #A,B,C,D: Constant matrices (See Murphy's book)

    def _get_dims(self):
        obs_dim, hid_dim = self.C.shape()
        action_dim = self.B.shape()[1]
        return [obs_dim, hid_dim, action_dim]

    def __init__(self, A, C, **params):
        obs_dim,hid_dim = C.shape()
        params.setdefault('action_dim', 1)
        params.setdefault('B',ConstMat(np.zeros((hid_dim, params['action_dim']))))
        params.setdefault('D',ConstMat(np.zeros((obs_dim, params['action_dim']))))
        self.A = A
        self.B = params['B']
        self.C = C
        self.D = params['D']

    def init_params(self, **params):
        obs_dim,hid_dim,action_dim = self._get_dims()
        params.setdefault('init_state_mean', np.zeros(hid_dim))
        params.setdefault('init_state_cov', 100*np.eye(hid_dim))
        params.setdefault('init_Sigma', 100*np.eye(obs_dim))
        params.setdefault('init_Gamma', 0.01*np.eye(hid_dim))
        self.mu0 = params['init_state_mean']
        self.P0 = params['init_state_cov']
        self.Sigma = params['init_Sigma']
        self.Gamma = params['init_Gamma']

    def generate_sample(self, T, **params):
        obs_dim,hid_dim,action_dim = self._get_dims()
        params.setdefault('actions', np.zeros((T,action_dim)))
        actions = params['actions']
        sample = []
        z = np.random.multivariate_normal(self.mu0, self.P0)
        hidden = []
        for t in xrange(T):
            hidden.append(z)
            state = {'t': t, 'prev_z': z, 'action': actions[t,:], 'A': self.A, 'B': self.B, 'C': self.C, 'D': self.D}
            noiseless_x = np.dot(self.C.mat(state),z) + np.dot(self.D.mat(state),actions[t,:])
            x = np.random.multivariate_normal(noiseless_x, self.Sigma)
            z = np.random.multivariate_normal(np.dot(self.A.mat(state),z) + np.dot(self.B.mat(state),actions[t,:]), self.Gamma)
            sample.append(np.array([x]))
        return np.array(sample), np.array(hidden)

    def _numerical_stability(self):
        #Improve numerical stability by ensuring some matrix properties
        self.Gamma = (self.Gamma + self.Gamma.T) / 2 #Ensure matrix is symmetric
        self.Sigma = (self.Sigma + self.Sigma.T) / 2 #Same for Sigma (Probably unnecessary)
        self.P0 = (self.P0 + self.P0.T) / 2 #Same for P0 (Probably unnecessary)

    def get_delta_Ts(self, times):
        return map(lambda ix: times[ix+1]-times[ix], range(len(times)-1))

    def forward_recursion(self, Xn, Un, **params):
        Tn = np.shape(Un)[0]
        On = np.shape(Xn)[0]
        hidden = [False for i in xrange(On)] if not 'hidden' in params else params['hidden']
        P = self.P0
        means = []
        covs = []
        P_vals = []
        state = {'t': 0, 'prev_z': self.mu0, 'action': Un[0,:], 'A': self.A, 'B': self.B, 'C': self.C, 'D': self.D}
        if 'n' in params: state['n'] = params['n']
        As = [-1]
        Bs = [-1]
        Cs = [self.C.mat(state)]
        Ds = [self.D.mat(state)]
        Ct = Cs[0]
        Dt = Ds[0]
        for t in xrange(Tn):
            #tmp = np.linalg.inv(np.dot(Ct,np.dot(P,Ct.T)) + self.Sigma)
            #K = np.dot(P,np.dot(Ct.T,tmp))
            hid_t = t>=On or hidden[t]
            if t==0:
                #pred = np.dot(Ct, self.mu0) + np.dot(Dt, Un[t,:])
                mu = self.mu0 #+ ( 0 if hid_t else np.dot(K, Xn[t,:] - pred) )
            else:
                #pred = np.dot(Ct,np.dot(At,mu)) + np.dot(Ct,np.dot(Bt,Un[t,:])) + np.dot(Dt, Un[t,:])
                mu = np.dot(At,mu) + np.dot(Bt,Un[t,:]) #+ (0 if hid_t else np.dot(K,Xn[t,:] - pred))
            V = P #- ( 0 if hid_t else np.dot(K,np.dot(Ct,P)) )
            if not hid_t:
                for obs in Xn[t]:
                    pred = np.dot(Ct, mu) + np.dot(Dt, Un[t,:])
                    tmp = np.linalg.inv(np.dot(Ct,np.dot(V,Ct.T)) + self.Sigma)
                    K = np.dot(V, np.dot(Ct.T, tmp))
                    mu += np.dot(K, obs - pred)
                    V = V - np.dot(K,np.dot(Ct,V))
            V = (V + V.T) / 2 #For numerical reasons
            state = {'t': t+1, 'prev_z': mu, 'action': Un[t,:], 'A': self.A, 'B': self.B, 'C': self.C, 'D': self.D}
            if 'n' in params: state['n'] = params['n']
            opt_mat_pars = {}
            #if 'deltaTs' in params and t<len(params['deltaTs']):
            #    opt_mat_pars['deltaT'] = params['deltaTs'][t]
            At = self.A.mat(state, None, **opt_mat_pars)
            Bt = self.B.mat(state, None, **opt_mat_pars)
            Ct = self.C.mat(state, None, **opt_mat_pars)
            Dt = self.D.mat(state, None, **opt_mat_pars)
            P = self.Gamma + np.dot(At,np.dot(V,At.T))
            P = (P + P.T) / 2 #For numerical reasons
            means.append(mu)
            covs.append(V)
            P_vals.append(P)
            As.append(At)
            Bs.append(Bt)
            Cs.append(Ct)
            Ds.append(Dt)
        return means, covs, P_vals, As, Bs, Cs, Ds

    def backward_recursion(self, fw_means, fw_covs, fw_P_vals, Un, As, Bs):
        Tn = np.shape(Un)[0]
        bw_means = list(fw_means)
        bw_covs = list(fw_covs)
        J_vals = [None] * (Tn-1)
        for t in xrange(Tn-2,-1,-1):
            Jt = np.dot(fw_covs[t],np.dot(As[t+1].T,np.linalg.inv(fw_P_vals[t])))
            Vt = fw_covs[t] + np.dot(Jt,np.dot(bw_covs[t+1] - fw_P_vals[t], Jt.T))
            bw_surprise = bw_means[t+1] - (np.dot(As[t+1],fw_means[t]) + np.dot(Bs[t+1], Un[t+1,:]))
            mu = fw_means[t] + np.dot(Jt, bw_surprise)
            Vt = (Vt + Vt.T) / 2 #For numerical reasons
            bw_means[t] = mu
            bw_covs[t] = Vt
            J_vals[t] = Jt
        return bw_means, bw_covs, J_vals

    def __recomp_As(self, fwd_means, U):
        As = []
        for n,fmeans in enumerate(fwd_means):
            An = [-1]
            for t in range(len(fmeans)):
                if t!= 0:
                    state = {'t': t, 'n': n, 'prev_z': fmeans[t-1], 'action': U[n][t,:], 'A': self.A, 'B': self.B, 'C': self.C, 'D': self.D}
                    A = self.A.mat(state)
                    An.append(A)
            As.append(An)
        return As

    def _EM_lowerbound(self, means, covs, Js, As, Bs, Cs, Ds, X, U, fwd_means, **params):
        ans = 0.0
        N = len(X)
        obs_dim, hid_dim, action_dim = self._get_dims()
        tmp, log_det_P0 = np.linalg.slogdet(self.P0)
        tmp, log_det_Sigma =  np.linalg.slogdet(self.Sigma)
        tmp, log_det_Gamma =  np.linalg.slogdet(self.Gamma)
        inv_P0 = np.linalg.inv(self.P0)
        inv_Sigma = np.linalg.inv(self.Sigma)
        inv_Gamma = np.linalg.inv(self.Gamma)
        gauss_exp = lambda x_t,mu_t,inv_Sigma_t: np.dot(x_t-mu_t, np.dot(inv_Sigma_t, x_t-mu_t))
        for n in xrange(N):
            ans += log_det_P0 + np.trace(np.dot(inv_P0,covs[n][0])) + gauss_exp(means[n][0], self.mu0, inv_P0)
            Tn = len(X[n])
            hidden = [False for i in xrange(Tn)] if not 'hidden' in params else params['hidden'][n]
            for t in xrange(Tn):
                C = Cs[n][t]
                D = Ds[n][t]
                if t!=0:
                    if 'A_params' in params:
                        state = {'t': t, 'n': n, 'prev_z': fwd_means[n][t-1], 'action': U[n][t,:], 'A': self.A, 'B': self.B, 'C': self.C, 'D': self.D}
                        #if 'n' in params: state['n'] = params['n']
                        #assert(not 'n' in params)
                        A = self.A.mat(state, params['A_params'])
                    else:
                        A = As[n][t]
                    B = Bs[n][t]
                    Mnt = np.dot(covs[n][t], np.dot(Js[n][t-1].T, A.T))
                    tmp_cov = covs[n][t] - Mnt - Mnt.T + np.dot(A, np.dot(covs[n][t-1], A.T))
                    tmp_mean = np.dot(A, means[n][t-1]) + np.dot(B, U[n][t,:])
                    ans += log_det_Gamma + np.trace(np.dot(inv_Gamma, tmp_cov)) + gauss_exp(means[n][t], tmp_mean, inv_Gamma)
                tmp_cov = np.dot(C, np.dot(covs[n][t], C.T))
                tmp_mean = np.dot(C, means[n][t]) + np.dot(D, U[n][t,:])
                if not hidden[t]:
                    Xnt = X[n][t]
                    for obs in Xnt:
                        ans += log_det_Sigma + np.trace(np.dot(inv_Sigma,tmp_cov)) + gauss_exp(obs,tmp_mean,inv_Sigma)
        if 'prior_P0' in params:
            P0_params = params['prior_P0']
            ans += (P0_params['v']+hid_dim+1)*log_det_P0 + np.trace(np.dot(P0_params['invS0'], inv_P0))
        if 'prior_Sigma' in params:
            Sig_params = params['prior_Sigma']
            ans += (Sig_params['v']+obs_dim+1)*log_det_Sigma + np.trace(np.dot(Sig_params['invS0'], inv_Sigma))
        if 'prior_Gamma' in params:
            Gam_params = params['prior_Gamma']
            ans += (Gam_params['v']+hid_dim+1)*log_det_Gamma + np.trace(np.dot(Gam_params['invS0'], inv_Gamma))
        return -0.5*ans

    def stoch_A_err(self, a_params, pairs, means, covs, Js, X, U, log_det_Gamma, inv_Gamma, Bs, fwd_means):
        ans = 0.0
        gauss_exp = lambda x_t,mu_t,inv_Sigma_t: np.dot(x_t-mu_t, np.dot(inv_Sigma_t, x_t-mu_t))
        for (n,t) in pairs:
            if t != 0:
                state = {'t': t, 'n': n, 'prev_z': fwd_means[n][t-1], 'action': U[n][t,:], 'A': self.A, 'B': self.B, 'C': self.C, 'D': self.D}
                #if 'n' in params: state['n'] = params['n']
                A = self.A.mat(state, a_params)
                B = Bs[n][t]
                Mnt = np.dot(covs[n][t], np.dot(Js[n][t-1].T, A.T))
                tmp_cov = covs[n][t] - Mnt - Mnt.T + np.dot(A, np.dot(covs[n][t-1], A.T))
                tmp_mean = np.dot(A, means[n][t-1]) + np.dot(B, U[n][t,:])
                ans += np.trace(np.dot(inv_Gamma, tmp_cov)) + gauss_exp(means[n][t], tmp_mean, inv_Gamma)
        return ans/len(pairs)

    def stoch_A_deriv(self, a_params, pairs, means, covs, Js, X, U, log_det_Gamma, inv_Gamma, Bs, fwd_means):
        ans = np.zeros(self.A.n_params())
        obs_dim,hid_dim,action_dim = self._get_dims()
        N = len(U)
        for (n,t) in pairs:
            if t != 0:
                state = {'t': t, 'prev_z': fwd_means[n][t-1], 'action': U[n][t,:], 'A': self.A, 'B': self.B, 'C': self.C, 'D': self.D}
                state['n'] = n
                A = self.A.mat(state, a_params)
                B = Bs[n][t]
                term1 = np.dot(covs[n][t],Js[n][t-1])
                term2 = np.outer(means[n][t] - np.dot(B,U[n][t,:]),means[n][t-1])
                term3 = np.dot(A,np.outer(means[n][t-1], means[n][t-1]) + covs[n][t-1])
                preMat = np.dot(inv_Gamma, term1+term2-term3).T
                for d in xrange(self.A.n_params()):
                    ans[d] += np.trace( np.dot(preMat,self.A.deriv(state,d,a_params)) )
        return -2*ans / len(pairs)

    def __deriv_lowerbound_theta_A(self, U, means, covs, Js, As, Bs, Cs, Ds, Gamma_inv, params):
        ans = np.zeros(self.A.n_params())
        obs_dim,hid_dim,action_dim = self._get_dims()
        N = len(U)
        for n in xrange(N):
            Tn = len(U[n])
            for t in xrange(1,Tn):
                state = {'t': t, 'prev_z': means[n][t-1], 'action': U[n][t,:], 'A': self.A, 'B': self.B, 'C': self.C, 'D': self.D}
                state['n'] = n
                term1 = np.dot(covs[n][t],Js[n][t-1])
                term2 = np.outer(means[n][t] - np.dot(Bs[n][t],U[n][t,:]),means[n][t-1])
                term3 = np.dot(As[n][t],np.outer(means[n][t-1], means[n][t-1]) + covs[n][t-1])
                preMat = np.dot(Gamma_inv, term1+term2-term3).T
                for d in xrange(self.A.n_params()):
                    ans[d] += np.trace( np.dot(preMat,self.A.deriv(state,d,params)) )
        return ans

    def _E_step(self, X, U, **params):
        N = len(X)
        fwd_means = []
        means = []
        covs = []
        Js = []
        As = []
        Bs = []
        Cs = []
        Ds = []
        for n in xrange(N):
            opt_fwd_params = {}
            if 'hidden' in params:
                opt_fwd_params['hidden'] = params['hidden'][n]
            if 'Time' in params:
                deltaTs = self.get_delta_Ts(params['Time'][n])
                opt_fwd_params['deltaTs'] = deltaTs
            fw_means, fw_covs, P_vals, An, Bn, Cn, Dn = self.forward_recursion(X[n], U[n], n=n, **opt_fwd_params)
            bw_means, bw_covs, J_vals = self.backward_recursion(fw_means, fw_covs, P_vals, U[n], An, Bn)
            means.append(bw_means)
            covs.append(bw_covs)
            Js.append(J_vals)
            As.append(An)
            Bs.append(Bn)
            Cs.append(Cn)
            Ds.append(Dn)
            fwd_means.append(fw_means)
        return means, covs, Js, As, Bs, Cs, Ds, fwd_means

    def _M_step(self, means, covs, Js, As, Bs, Cs, Ds, X, U, fwd_means, **params):
        obs_dim, hid_dim, action_dim = self._get_dims()
        N = len(X)
        mu0 = (1.0/N)*reduce(lambda a,b: a+b, map(lambda x: x[0], means))
        P0_N = reduce(lambda a,b: a+b, map(lambda n: covs[n][0] + np.outer(means[n][0]-mu0, means[n][0]-mu0), xrange(N)))
        if 'prior_P0' in params:
            P0_params = params['prior_P0']
            P0 = (1.0/(N+P0_params['v']+hid_dim+1)) * (P0_params['invS0'] + P0_N)
        else:
            P0 = (1.0/N)*P0_N
        if (params['debug_LB']): 
            print "Debug. Initial LB=", self._EM_lowerbound(means,covs,Js,As,Bs,Cs,Ds,X,U,fwd_means,**params)
        self.mu0 = mu0
        if (params['debug_LB']): print "Debug. After updating mu0=", self._EM_lowerbound(means,covs,Js,As,Bs,Cs,Ds,X,U,fwd_means,**params)
        if (params['em_print_params']): print "mu0=", self.mu0
        self.P0 = P0
        if (params['debug_LB']): 
            print "Debug. After updating P0=", self._EM_lowerbound(means,covs,Js,As,Bs,Cs,Ds,X,U,fwd_means,**params)
        if (params['em_print_params']): print "P0=",self.P0

        Avalid = True
        if params['optimize_A']:
            tmp, log_det_Gamma =  np.linalg.slogdet(self.Gamma)
            inv_Gamma = np.linalg.inv(self.Gamma)
            pairs = [(n,t) for n in range(N) for t in range(len(X[n]))]
            np.random.shuffle(pairs)
            x = self.A.params
            for epoch in range(self.__a_epochs):
                for i in range(len(pairs)/self.__a_bs):
                    pairs_i = pairs[i*self.__a_bs:(i+1)*self.__a_bs]
                    grad = self.stoch_A_deriv(x, pairs_i, means, covs, Js, X, U, log_det_Gamma, inv_Gamma, Bs, fwd_means)
                    obj = self.stoch_A_err(x, pairs_i, means, covs, Js, X, U, log_det_Gamma, inv_Gamma, Bs, fwd_means)
                    if params['debug_LB'] and np.random.randint(0,10000)==0:
                        #Check gradients
                        my_obj = lambda x: self.stoch_A_err(x, pairs_i, means, covs, Js, X, U, log_det_Gamma, inv_Gamma, Bs, fwd_means)
                        print("error: {}".format(my_obj(self.A.params)))
                        print("error grad: {}".format(grad))
                        num_grad = my_num_grad(my_obj, self.A.params, 1e-5)
                        print("num grad: {}".format(num_grad))
                    #print(x,obj,grad)
                    grad[3:6] = grad[3:6]*0.001
                    x = x - self.__a_lr*grad
                if (params['debug_LB']): print("error: {}".format(self.stoch_A_err(x, pairs, means, covs, Js, X, U, log_det_Gamma, inv_Gamma, Bs, fwd_means)))
            self.A.params = x
            if (params['debug_LB']):
                print "A_params: ", self.A.params
                print "Debug. After updating A=", self._EM_lowerbound(means,covs,Js,As,Bs,Cs,Ds,X,U,fwd_means,A_params=self.A.params, **params)
            Avalid = False

        if False:
            my_obj = lambda x: -self._EM_lowerbound(means,covs, Js, As, Bs, Cs, Ds, X, U, A_params = x, **params)
            my_grad = lambda x: -self.__deriv_lowerbound_theta_A(U, means, covs, Js, As, Bs, Cs, Ds, np.linalg.inv(self.Gamma), x)
            #my_grad = lambda params: my_num_grad(my_obj, params, 1e-8)
            if (params['check_grad']):
                num_grad = my_num_grad(my_obj, self.A.params, 1e-4)
                an_grad = my_grad(self.A.params)
                print "Num_grad=",num_grad," An_grad=", an_grad, " Diff=", num_grad-an_grad
                print "Grad_check: ", opt.check_grad(my_obj, my_grad, self.A.params)
            #res = opt.minimize(my_obj, self.A.params, method='CG', jac=my_grad)
            res = opt.minimize(my_obj, self.A.params, method='Powell', options={'maxfev': 30, 'maxiter': 5})
            self.A.params = res.x
            if (params['em_print_params']): print "A_params: ", self.A.params
            if (params['debug_LB']):
                print "A_params: ", self.A.params
                print "Debug. After updating A=", self._EM_lowerbound(means,covs,Js,As,Bs,Cs,Ds,X,U,A_params=self.A.params, **params)
            Avalid = False

        Ntotal = 0
        ObsTotal = 0
        Gamma = np.zeros((hid_dim,hid_dim))
        Sigma = np.zeros((obs_dim,obs_dim))
        for n in xrange(N):
            Tn = len(X[n])
            hidden = [False for i in xrange(Tn)] if not 'hidden' in params else params['hidden'][n]
            Ntotal += Tn
            for t in xrange(Tn):
                assert np.allclose(covs[n][t],covs[n][t].T)
                state = {'t': t, 'n': n, 'prev_z': fwd_means[n][t-1], 'action': U[n][t,:], 'A': self.A, 'B': self.B, 'C': self.C, 'D': self.D}
                if not Avalid: As[n][t] = self.A.mat(state)
                A = As[n][t]
                B = Bs[n][t]
                C = Cs[n][t]
                D = Ds[n][t]
                if t != 0:
                    Mnt = np.dot(covs[n][t],np.dot(Js[n][t-1].T, A.T))
                    dif = means[n][t] - (np.dot(A,means[n][t-1]) + np.dot(B, U[n][t,:]))
                    tmp_cov = covs[n][t] - Mnt - Mnt.T + np.dot(A, np.dot(covs[n][t-1], A.T))
                    Gamma += tmp_cov + np.outer(dif,dif)
                if not hidden[t]:
                    Xnt = X[n][t]
                    for obs in Xnt:
                        dif = obs - (np.dot(C,means[n][t]) + np.dot(D, U[n][t,:]))
                        Sigma += np.dot(C, np.dot(covs[n][t], C.T)) + np.outer(dif,dif)
                        ObsTotal += 1
        if 'prior_Gamma' in params:
            Gam_params = params['prior_Gamma']
            self.Gamma = (Gamma + Gam_params['invS0']) / (Ntotal - N + Gam_params['v'] + hid_dim + 1)
        else:
            self.Gamma = Gamma / (Ntotal - N)
        if not 'full_gamma' in params: self.Gamma = np.diag(np.diag(self.Gamma))
        if (params['debug_LB']): print "Debug. After updating Gamma=", self._EM_lowerbound(means,covs,Js,As,Bs,Cs,Ds,X,U,fwd_means,**params)
        if (params['em_print_params']): print "Gamma=",self.Gamma
        if 'prior_Sigma' in params:
            Sig_params = params['prior_Sigma']
            self.Sigma = (Sigma + Sig_params['invS0']) / (ObsTotal + Sig_params['v'] + obs_dim + 1)
        else:
            self.Sigma = Sigma / ObsTotal
        if not 'full_sigma' in params: self.Sigma = np.diag(np.diag(self.Sigma))
        if (params['debug_LB']): print "Debug. After updating Sigma=", self._EM_lowerbound(means,covs,Js,As,Bs,Cs,Ds,X,U,fwd_means,**params)
        if (params['em_print_params']): print "Sigma=",self.Sigma
        self._numerical_stability()
    
    def loglikelihood(self, X, U, **params):
        means, covs, Js, As, Bs, Cs, Ds, fwd_means = self._E_step(X, U, **params)
        return self._EM_lowerbound(means, covs, Js, As, Bs, Cs, Ds, X, U, fwd_means, **params)

    def EM_train(self, X, U, **params):
        self.init_params(**params)
        params.setdefault('max_iter',20)
        params.setdefault('debug_LB', False)
        params.setdefault('em_print_params', False)
        params.setdefault('print_lowerbound', False)
        params.setdefault('check_grad', False)
        params.setdefault('verbose', False)
        params.setdefault('optimize_A', False)
        params.setdefault('A_bounds', None)
        max_iter = params['max_iter']
        if params['optimize_A']: 
            self.__a_lr = 1e-2 #learning rate
            self.__a_bs = 128 #batch size
            self.__a_epochs = 10
        for i in xrange(max_iter):
            means, covs, Js, As, Bs, Cs, Ds, fwd_means = self._E_step(X, U, **params)
            if (params['print_lowerbound']): print "LB E-Step:", self._EM_lowerbound(means, covs, Js, As, Bs, Cs, Ds, X, U, fwd_means, **params)
            if ('callback_after_E_step' in params): params['callback_after_E_step'](self, X, U, means, covs)
            self._M_step(means, covs, Js, As, Bs, Cs, Ds, X, U, fwd_means, **params)
            if (params['print_lowerbound']): print "LB M-Step:", self._EM_lowerbound(means, covs, Js, As, Bs, Cs, Ds, X, U, fwd_means, **params)
