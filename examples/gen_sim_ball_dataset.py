
import numpy as np
import argparse

def sim_ball_traj(init_pos=np.zeros(3), init_vel=np.array([-1.3,4.5,2.2]), lin_air_drag=np.array([0.0,0.0,0.0]), 
        quad_air_drag=0.0, bounce_fac=np.array([0.9,0.9,0.8]), deltaT=0.005, T=120, max_bounces=None):
    x = init_pos
    xd = init_vel
    obs = []
    vel = []
    time = []
    is_bounce = []
    bounce_time = []
    bounces_left = max_bounces if max_bounces is not None else 1000000
    for i in range(T):
        t = deltaT*i
        obs.append(x)
        vel.append(xd)
        time.append(t)
        a = -lin_air_drag*xd - quad_air_drag*np.linalg.norm(xd)*xd + np.array([0,0,-9.8])
        x = x + xd*deltaT + 0.5*(deltaT**2)*a
        xd = xd + deltaT*a
        if x[2] < 0.0 and bounces_left > 0:
            is_bounce.append(True)
            bounce_time.append(t)
            x[2] *= -1.0
            xd[2] *= -1.0
            xd = xd*bounce_fac
            bounces_left -= 1
        else:
            is_bounce.append(False)
    return np.array(time), np.array(obs), np.array(vel), is_bounce, bounce_time


def sim_data(N=100, T=120, 
        init_state = {
            'pos': {'mean': np.array([0.0,0.0,0.3]), 'cov': np.diag(np.array([0.5,1,0.01])**2)},
            'vel': {'mean': np.array([-1.4,4.5,2.3]), 'cov': np.eye(3)}
        }, deltaT=0.005, max_bounces=None, bounce_fac=np.array([0.9,0.9,0.8]),
        lin_air_drag=np.array([0,0,0]), quad_air_drag=0.15):
    X = []
    Xd = []
    times = []
    is_bounce = []
    bounce_times = []
    s = lambda x: np.random.multivariate_normal(x['mean'], x['cov'])
    for i in range(N):
        time, x, xd, is_bounce_n, bounce_time_n = sim_ball_traj(init_pos=s(init_state['pos']), 
                init_vel=s(init_state['vel']), 
                lin_air_drag=lin_air_drag, 
                quad_air_drag=quad_air_drag,
                bounce_fac=bounce_fac,
                deltaT=deltaT, T=T, max_bounces=max_bounces)
        X.append(x)
        Xd.append(xd)
        times.append(time)
        is_bounce.append(is_bounce_n)
        bounce_times.append(bounce_time_n)
    return np.array(times), np.array(X), np.array(Xd), is_bounce, bounce_times

def main(args):
    times, X, Xd, is_bounce, bounce_times = sim_data(N=args.N, T=args.T, max_bounces=args.max_bounces,
            bounce_fac=args.bounce_fac, quad_air_drag=args.air_drag, deltaT=args.dt)
    noisyX = X + np.random.normal(loc=0, scale=args.noise, size=X.shape)
    with open(args.data,'wb') as f:
        np.savez(f, Times=times, X=noisyX, Xd=Xd, Xsim=X, is_bounce=is_bounce, 
            bounce_times=bounce_times)
        print("{} trajectories saved to file".format(len(times)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', help="Path to file with data")
    parser.add_argument('N', type=int, help="Number of trajectories to generate")
    parser.add_argument('--T', type=int, default=200, help="Number of time steps per trajectory")
    parser.add_argument('--max_bounces', type=int, help="Maximum number of bounces")
    parser.add_argument('--noise', type=float, default=0.01, help="Scale of the gaussian noise to add to the positions")
    parser.add_argument('--air_drag', type=float, default=0.15, help="Quadratic factor for the air drag")
    parser.add_argument('--bounce_fac', type=float, default=0.9, help="Energy kept by the ball after bouncing")
    parser.add_argument('--dt', type=float, default=1.0/180.0, help="Energy kept by the ball after bouncing")
    args = parser.parse_args()
    main(args)
