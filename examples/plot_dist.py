
import traj_pred.trajectory as traj
import traj_pred.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import time

import argparse
import logging

def main(args):
    logging.basicConfig(level=logging.INFO)
    with open(args.data,'rb') as f:
        data = np.load(f, encoding='latin1')
        X = data['X']
        Times = data['Times']
    pred = traj.load_model(args.model)
    D = 3

    pred_dur = int(round(args.t/args.dt))
    pred_times = np.arange(0,pred_dur)*args.dt
    for i,tabs in enumerate(Times):
        t = tabs - tabs[0]
        if i > args.n: break
        T = len(t)
        x = np.array(X[i])
        given_ix = np.random.randint(low=0, high=T-1)
        prev_times = t[0:given_ix+1]
        prev_obs = x[0:given_ix+1]

        t1 = time.time()
        pred_mean, pred_cov = pred.traj_dist(prev_times, prev_obs, pred_times)
        t2 = time.time()
        logging.info("Prediction latency: {}".format(t2-t1))
        pred_std = utils.cov_to_std(pred_cov)
        plt.plot(prev_times, prev_obs, 'bo')
        plt.plot(t[given_ix+1:], x[given_ix+1:], 'ro')
        plt.axvline(x=t[given_ix])

        plt.figure(1)
        for d in range(D):
            plt.subplot(D,1,d+1)
            plt.plot(prev_times,prev_obs[:,d],'g.')
            plt.plot(t[given_ix+1:], x[given_ix+1:,d], 'r.')
            plt.axvline(x=t[given_ix])
            y_test_mean = pred_mean[:,d]
            y_test_std = pred_std[:,d] 
            plt.plot(pred_times, y_test_mean, 'b')
            plt.fill_between(pred_times, y_test_mean - 2*y_test_std, y_test_mean + 2*y_test_std,
                    color='b', alpha=0.5)
        plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', help="File with the normalized training/validation data")
    parser.add_argument('model', help="Path where the model is stored")
    parser.add_argument('--n', type=int, default=5, help="Number of instances to plot")
    parser.add_argument('--t', type=float, default=1.2, help="Time to predict in the future")
    parser.add_argument('--dt', type=float, default=1.0/180.0, help="Delta Time")
    args = parser.parse_args()
    main(args)
