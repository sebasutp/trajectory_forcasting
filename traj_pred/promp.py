
import tensorflow as tf
import numpy as np

def promp_poly_basis(inputs, num_outputs, degree):
    curr_pow = tf.ones(tf.shape(inputs), dtype=tf.float64)
    poly_basis = []
    for i in range(degree):
        poly_basis.append(curr_pow)
        curr_pow = tf.multiply(curr_pow, inputs)
    poly_basis.append(curr_pow)
    tensor_basis = tf.concat(poly_basis, axis = 2)
    #For the moment, this only works with num_outputs=1
    return tf.expand_dims(tensor_basis, 2)

def rbf(inputs, init_centers, init_scale):
    N = tf.shape(inputs)[0]
    T = tf.shape(inputs)[1]
    k,d = init_centers.shape
    with tf.variable_scope('rbf') as vs:
        centers = tf.Variable(init_centers, name="centers", dtype=tf.float64)
        scale = tf.Variable(init_scale, name="scale", dtype=tf.float64)
        diff = tf.expand_dims(inputs,-2) - tf.reshape(centers,[1,1,k,d])
        sq_dist = tf.reduce_sum(diff*diff, axis=-1)
        rbfs = tf.exp(-(tf.reshape(scale*scale,[1,1,-1]) * sq_dist))
        bias_rbfs = tf.concat((tf.ones((N,T,1),dtype=tf.float64), rbfs), axis=2)
    return bias_rbfs

class DeepProMP:

    def __create_em_functions(self, prior_mu_w, prior_Sigma_w):
        # numerical stability matrices
        eps_sw = 1e-6*tf.eye(self.feature_dims, dtype=tf.float64)
        eps_sy = 1e-6*tf.eye(self.out_dims, dtype=tf.float64)

        self.inv_Sig_w = tf.matrix_inverse(self.Sigma_w + eps_sw)
        self.inv_Sig_y = tf.matrix_inverse(self.Sigma_y + eps_sy)
        self.log_det_sig_w = tf.linalg.logdet(self.Sigma_w)
        self.log_det_sig_y = tf.linalg.logdet(self.Sigma_y)
        N = tf.shape(self.x)[0] # batch size (number of functions)
        T = tf.shape(self.x)[1] # size of the support set for each function instance

        inv_Sig_y_nt = tf.tile(tf.reshape(self.inv_Sig_y, shape=(1,1,self.out_dims,self.out_dims)), [N,T,1,1])
        e_pmul = tf.matmul(a=self.phi, b=inv_Sig_y_nt, transpose_a=True, name="e_pmul")
        #e_pmul = tf.tensordot(tf.transpose(self.phi, perm=(0,1,3,2)), self.inv_Sig_y, axes=1)
        # To compute the E step
        e_cov_sum = tf.reduce_sum(tf.matmul(e_pmul,self.phi), axis = 1)
        comp_w_covs = tf.linalg.inv( tf.expand_dims(self.inv_Sig_w + eps_sw, 0) + e_cov_sum ) 
        self.comp_w_covs = (tf.transpose(comp_w_covs, perm=(0,2,1)) + comp_w_covs)/2.0

        e_mean_sum = tf.reduce_sum(tf.matmul(e_pmul, tf.expand_dims(self.y, -1)), axis=1)
        w_mean_helper = tf.expand_dims(tf.matmul(self.inv_Sig_w,self.mu_w),0) + e_mean_sum
        self.comp_w_means = tf.matmul(self.comp_w_covs, w_mean_helper)

        # Average lower bound
        m_pred_y = tf.matmul(self.phi, tf.tile(tf.expand_dims(self.w_means, 1),[1,T,1,1]))
        m_err_y = tf.expand_dims(self.y,-1) - m_pred_y
        m_mah_dist_y = tf.matmul(a=m_err_y, b=tf.matmul(inv_Sig_y_nt, m_err_y), transpose_a = True)
        m_uncert_y = tf.matmul(self.phi, tf.matmul(a=tf.tile(tf.expand_dims(self.w_covs,1),[1,T,1,1]), b=self.phi, transpose_b=True))
        m_tr_y = tf.trace( tf.matmul(inv_Sig_y_nt, m_uncert_y) )
        m_lh_y = self.log_det_sig_y + m_tr_y + tf.reshape(m_mah_dist_y,[N,T])
        m_avg_lh_y = tf.reshape(tf.reduce_mean(m_lh_y, axis=1), [N])

        m_w_err = self.w_means - tf.reshape(self.mu_w,shape=(1,self.feature_dims,1))
        m_mah_dist_w = tf.matmul(a=m_w_err, b=tf.matmul(tf.tile(tf.expand_dims(self.inv_Sig_w,0),[N,1,1]), m_w_err), transpose_a=True)
        m_tr_w = tf.trace(tf.matmul(tf.tile(tf.expand_dims(self.inv_Sig_w,0),[N,1,1]), self.w_covs))
        lh_w = m_tr_w + tf.reshape(m_mah_dist_w,[N]) + self.log_det_sig_w
        #m_tr_w = tf.trace(tf.tensordot(self.w_covs, self.inv_Sig_w, axes=1))
        self.mean_y_pred = m_pred_y
        self.cov_y_pred = m_uncert_y

        # M-Step MLE updates
        mu_w_mle = tf.reduce_mean(self.w_means, axis=0) # K*1
        cov_nw = self.w_covs + tf.matmul(a=m_w_err, b=m_w_err, transpose_b=True) # N*K*K
        m_step_Sigma_w = tf.reduce_mean(cov_nw, axis=0) # K*K
        Sigma_w_mle = (tf.transpose(m_step_Sigma_w) + m_step_Sigma_w)/2.0

        nlpriors = 0.0
        if prior_mu_w is not None:
            m0 = prior_mu_w['m0']
            inv_V0 = prior_mu_w['k0']*self.inv_Sig_w #Normal-Inverse-Wishart prior
            dist_w_prior = self.mu_w - m0 #K*1
            nlpriors += tf.squeeze(tf.matmul(a=dist_w_prior, b=tf.matmul(inv_V0,dist_w_prior), transpose_a=True))
        if prior_Sigma_w is not None:
            v0 = prior_Sigma_w['v0']
            if 'mean_cov_mle' in prior_Sigma_w:
                S0 = prior_Sigma_w['mean_cov_mle'](v0, Sigma_w_mle) * (v0 + self.feature_dims + 1)
            else:
                S0 = prior_Sigma_w['invS0']
            nlpriors += (v0 + self.feature_dims + 1)*self.log_det_sig_w + tf.trace(tf.matmul(S0, self.inv_Sig_w))

        self.avg_loss = tf.reduce_mean(lh_w + m_avg_lh_y, axis = 0) + nlpriors/tf.to_double(N)
        self.avg_elbo = -0.5*self.avg_loss
        self.learning_rate = tf.placeholder(tf.float64, name="learning_rate")

        # M-Step updates
        if prior_mu_w is None:
            self.m_step_mu_w = mu_w_mle 
        else:
            self.m_step_mu_w = (tf.to_double(N)*mu_w_mle + prior_mu_w['k0']*m0)/(tf.to_double(N)+prior_mu_w['k0'])
        if prior_Sigma_w is None:
            self.m_step_Sigma_w = Sigma_w_mle 
        else:
            sw_map = (S0 + tf.to_double(N)*Sigma_w_mle) / (v0 + tf.to_double(N + self.feature_dims + 1))
            self.m_step_Sigma_w = (sw_map + tf.transpose(sw_map)) / 2.0 
        cov_ny = tf.matmul(a=m_err_y, b=m_err_y, transpose_b=True) + m_uncert_y # N*T*D*D
        m_step_Sigma_y = tf.reduce_mean(cov_ny, axis=[0,1])
        if self.__diag_sy:
            self.m_step_Sigma_y = tf.diag(tf.diag_part(m_step_Sigma_y)) #+ eps_sy
        else:
            self.m_step_Sigma_y = (tf.transpose(m_step_Sigma_y) + m_step_Sigma_y) / 2.0
        self.closed_m_steps = [self.mu_w.assign(self.m_step_mu_w), 
                self.Sigma_w.assign(self.m_step_Sigma_w),
                self.Sigma_y.assign(self.m_step_Sigma_y)]
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            basis_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='basis_fun')
            print("Basis function variables: {}".format(basis_params))
            if len(basis_params) > 0:
                self.opt_basis = tf.train.AdamOptimizer(self.learning_rate).minimize(self.avg_loss, name="train_step",
                        var_list=basis_params)
            else:
                self.opt_basis = None


    def __init__(self, in_dims, out_dims, feature_dims, phi,
            init_Sigma_y = None, 
            init_mu_w = None, 
            init_Sigma_w=None,
            prior_mu_w=None,
            prior_Sigma_w=None):
        if init_Sigma_y is None: init_Sigma_y = 1e-2*np.eye(out_dims)
        if init_mu_w is None: init_mu_w = np.zeros((feature_dims,1))
        if init_Sigma_w is None: init_Sigma_w = 1e2*np.eye(feature_dims)
        self.__diag_sy = True
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.feature_dims = feature_dims
        self.x = tf.placeholder(tf.float64, shape=(None, None, in_dims))
        self.y = tf.placeholder(tf.float64, shape=(None, None, out_dims))
        with tf.variable_scope('promp') as vs:
            self.Sigma_y = tf.Variable(init_Sigma_y, name="Sigma_y")
            self.mu_w = tf.Variable(init_mu_w, name="mu_w")
            self.Sigma_w = tf.Variable(init_Sigma_w, name="Sigma_w")
        self.w_means = tf.placeholder(tf.float64, shape=(None,feature_dims,1))
        self.w_covs = tf.placeholder(tf.float64, shape=(None,feature_dims,feature_dims))
        with tf.variable_scope('basis_fun') as vs:
            self.phi = phi(self.x, self.is_training)

        self.__create_em_functions(prior_mu_w, prior_Sigma_w)

    def comp_w_dist(self, sess, inputs, outputs):
        """ Compute the probability distribution of the hidden representation

        Using the current model parameters, computes the probability distribution of the representation
        variables w for each of the (input,output) pairs. The inputs and outputs are expected to have sizes
        (N,T,d) and (N,T,D) respectively. Where d and D are the input and output dimensions.

        Note that this function corresponds to the implementation of the E step in EM.
        """
        [w_means, w_covs] = sess.run([self.comp_w_means, self.comp_w_covs], feed_dict={self.x: inputs, 
            self.y: outputs, self.is_training: False})
        return w_means, w_covs

    def elbo(self, sess, inputs, outputs, w_means, w_covs):
        """ Average of the EM lower bound of the marginal likelihood

        Computes the lower bound of the marginal likelihood divided by the number of training data. Opposed
        to the actual EM lower bound, this function can be optimized with Stochastic gradient decent.
        """
        return sess.run(self.avg_elbo, feed_dict={self.x: inputs, self.y: outputs, self.w_means: w_means, 
            self.w_covs: w_covs, self.is_training: False})

    def output_dist(self, sess, inputs, w_dist=None):
        """ Computes the distribution of the output marginalizing the hidden representation

        If w_dist is None, only the prior is used. Otherwise, for an input of size (N,T,D), w_dist should
        contain two tensors (w_means, w_covs) with shapes (N,K,1) and (N,K,K) respectively, where K is the
        feature dimensions. If w_dist is provided, the predictions are made with the provided distribution
        instead of the prior parameters.
        """
        N,T,in_dims = np.shape(inputs)
        if w_dist is None:
            mu_w, Sigma_w = sess.run([self.mu_w, self.Sigma_w])
            w_means = np.reshape(mu_w, shape=(1,self.feature_dims,1))
            w_covs = np.reshape(Sigma_w, shape=(1,self.feature_dims,self.feature_dims))
        else:
            w_means, w_covs = w_dist

        mean_y_pred, cov_y_pred = sess.run([self.mean_y_pred, self.cov_y_pred], 
                feed_dict={self.x: inputs, self.w_means: w_means, self.w_covs: w_covs, self.is_training: False})
        return mean_y_pred, cov_y_pred

    @staticmethod
    def __embatch(inputs, outputs, batch_size, sample_size, it, noise_x=None, noise_y=None):
        x = []
        y = []
        N = len(inputs)
        start_n = (batch_size*it) % N
        for i in range(batch_size):
            n = (start_n + i) % N
            Tn = len(inputs[n])
            assert(Tn >= sample_size)
            ix_n = np.random.permutation(Tn)[0:sample_size]
            x_n = inputs[n][ix_n,:]
            y_n = outputs[n][ix_n,:]
            if noise_x is not None: x_n += np.random.normal(loc=0, scale=noise_x, size=x_n.shape)
            if noise_y is not None: y_n += np.random.normal(loc=0, scale=noise_y, size=y_n.shape)
            x.append(x_n)
            y.append(y_n)
        return np.array(x),np.array(y)

    def __init_basis_fun(self, sess, inputs, outputs, em_batch_size, sample_size, lr=1e-3, init_iter=30, 
            noise_x=None, noise_y=None):
        """ Initialize the basis functions in a sensible way

        Assume that we have a one-layer larger NN. The weights of the last layer are independent
        per instance to predict, and we want to train several instances at a time.
        """
        x,y = self.__embatch(inputs, outputs, em_batch_size, sample_size, 0)
        w_init = np.random.multivariate_normal(mean=np.zeros(self.feature_dims), cov=np.eye(self.feature_dims), size=em_batch_size)
        with tf.variable_scope('init_phi') as vs:
            w = tf.Variable(np.reshape(w_init,(em_batch_size,self.feature_dims,1)), name="init_phi_w")
            y_pred = tf.matmul(self.phi, tf.tile(tf.expand_dims(w, 1),[1,sample_size,1,1]))
            y_err = tf.expand_dims(self.y,-1) - y_pred
            init_loss = tf.reduce_mean(tf.matmul(a=y_err,b=y_err,transpose_a=True))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                init_basis_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='basis_fun')
                init_basis_params.append(w)
                init_opt_basis = tf.train.AdamOptimizer(self.learning_rate).minimize(init_loss, name="init_phi_train_step",
                        var_list=init_basis_params)
        losses = []
        sess.run(tf.global_variables_initializer())
        for it in range(init_iter):
            x_n = x
            y_n = y
            if noise_x is not None: x_n += np.random.normal(loc=0,scale=noise_x,size=x.shape)
            if noise_y is not None: y_n += np.random.normal(loc=0,scale=noise_y,size=y.shape)
            loss, void_obj = sess.run([init_loss, init_opt_basis], feed_dict={self.x: x_n, 
                self.y: y_n, 
                self.learning_rate: lr, self.is_training: True})
            losses.append(loss)
        return losses


    def fit(self, sess, inputs, outputs, lr=1e-3, em_batch_size=None, basis_batch_size=None, sample_size=None, 
            opt_basis_steps=0, iterations=30, init_phi_iter=0, init_phi_lr=1e-2, noise_x=None, noise_y=None):
        """ Train the function prior

        The number of functions to take at each step is defined in batch_size, and the number
        of samples per function in sample_size. The inputs and outputs have shapes (N,T,d) and (N,T,D)
        respectively, with N>batch_size and T>sample_size.
        """
        N = len(inputs)
        if em_batch_size is None: em_batch_size = N
        if basis_batch_size is None: basis_batch_size = 1
        if sample_size is None: sample_size = min([len(x) for x in inputs])
        
        ans = {}
        if init_phi_iter>0:
            ans['init_phi_loss'] = self.__init_basis_fun(sess, inputs, outputs, em_batch_size, sample_size, 
                    init_phi_lr, init_phi_iter, noise_x, noise_y)
        
        loss = []
        ob_losses = []
        for it in range(iterations):
            x,y = self.__embatch(inputs, outputs, em_batch_size, sample_size, it, noise_x, noise_y)
            w_means, w_covs = self.comp_w_dist(sess, x, y)
            #w_covs += np.expand_dims(1e-6*np.eye(self.feature_dims), axis=0) # For numerical stability
            e_loss = sess.run([self.avg_loss], feed_dict={self.x: x, self.y: y, 
                    self.w_means: w_means, self.w_covs: w_covs, self.is_training: False, self.learning_rate: lr})
            loss.append(e_loss)
            sess.run(self.closed_m_steps, feed_dict={self.x: x, self.y: y, 
                    self.w_means: w_means, self.w_covs: w_covs, self.is_training: False, self.learning_rate: lr})
            m_loss = sess.run([self.avg_loss], feed_dict={self.x: x, self.y: y, 
                    self.w_means: w_means, self.w_covs: w_covs, self.is_training: False, self.learning_rate: lr})
            loss.append(m_loss)
            mu_w,Sigma_w,Sigma_y = sess.run([self.mu_w, self.Sigma_w, self.Sigma_y])
            print('E-loss: {}, M-loss: {}\nmu_w: {}\nSigma_w: {}\n,Sigma_y: {}\n'.format(e_loss, m_loss, mu_w, Sigma_w, Sigma_y))
            opt_basis_loss = []
            for j in range(opt_basis_steps):
                start_ix = (j*basis_batch_size) % em_batch_size
                end_ix = min(em_batch_size, start_ix+basis_batch_size)
                curr_loss, void_obj = sess.run([self.avg_loss, self.opt_basis], feed_dict={
                    self.x: x[start_ix:end_ix], self.y: y[start_ix:end_ix], 
                    self.w_means: w_means[start_ix:end_ix], 
                    self.w_covs: w_covs[start_ix:end_ix], 
                    self.is_training: True, self.learning_rate: lr})
                opt_basis_loss.append(curr_loss)
            ob_losses.append(opt_basis_loss)
        ans.update( {'loss': loss, 'opt_basis_losses': ob_losses} )
        return ans
