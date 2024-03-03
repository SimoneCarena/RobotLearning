"""
    Robot Learning
    Exercise 2

    Linear Quadratic Regulator

    Polito A-Y 2023-2024
"""
import gym
import numpy as np
from scipy import linalg     # get riccati solver
import argparse
import matplotlib.pyplot as plt
import sys
from utils import get_space_dim, set_seed
import pdb 
import time
import matplotlib.pyplot as plt
import scipy

# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--time_sleep", action='store_true',
                        help="Add timer for visualizing rendering with a slower frame rate")
    parser.add_argument("--mode", type=str, default="control",
                        help="Type of test ['control', 'multiple_R']")
    return parser.parse_args(args)

def linerized_cartpole_system(mp, mk, lp, g=9.81):
    mt=mp+mk
    a = g/(lp*(4.0/3 - mp/(mp+mk)))
    # state matrix
    A = np.array([[0, 1, 0, 0],
                [0, 0, a, 0],
                [0, 0, 0, 1],
                [0, 0, a, 0]])

    # input matrix
    b = -1/(lp*(4.0/3 - mp/(mp+mk)))
    B = np.array([[0], [1/mt], [0], [b]])
    return A, B

def optimal_controller(A, B, R_value=1):
    R = R_value*np.eye(1, dtype=int)  # choose R (weight for input)
    Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)
   # solve ricatti equation
    P = linalg.solve_continuous_are(A, B, Q, R)

    # calculate optimal controller gain
    K = np.dot(np.linalg.inv(R),
            np.dot(B.T, P))
    return K

def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left

def multiple_R(env, mp, mk, l, g, time_sleep=False, terminate=True):
    """
    Vary the value of R within the range [0.01, 0.1, 10, 100] and plot the forces 
    """

    U = np.zeros((1000,4),dtype=float)
    AbsU = np.zeros((1000,4),dtype=float)

    for (idx,R) in enumerate([0.01, 0.1, 10, 100]):
        obs = env.reset() 

        A, B = linerized_cartpole_system(mp, mk, l, g)
        K = optimal_controller(A, B, R)
        T = [i for i in range(400)]

        for i in range(1000):

            env.render()
            if time_sleep:
                time.sleep(.1)
        
            # get force direction (action) and force value (force)
            action, force = apply_state_controller(K, obs)
            U[i,idx] = force
        
            # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
            abs_force = abs(float(np.clip(force, -10, 10)))
            AbsU[i,idx] = abs_force

            # change magnitute of the applied force in CartPole
            env.env.force_mag = abs_force

            # apply action
            obs, reward, done, _ = env.step(action)
        
            if i>399:
                break

    for i in range(4):
        if U[0,i]>0:
            U[:,i]*=-1

    plt.plot(T,U[:400,0])
    plt.plot(T,U[:400,1])
    plt.plot(T,U[:400,2])
    plt.plot(T,U[:400,3])
    plt.legend([r'$u_{R=0.01}$',r'$u_{R=0.1}$',r'$u_{R=10}$',r'$u_{R=100}$'])
    plt.show()

    return

def control(env, mp, mk, l, g, time_sleep=False, terminate=True):
    """
    Control using LQR
    """

    obs = env.reset()    # Reset the environment for a new episode
    
    A, B = linerized_cartpole_system(mp, mk, l, g)
    K = optimal_controller(A, B)    # Re-compute the optimal controller for the current R value
    X = np.zeros((1000,4),dtype=float)
    T = [i for i in range(400)]

    for i in range(1000):

        env.render()
        if time_sleep:
            time.sleep(.1)
        
        # get force direction (action) and force value (force)
        action, force = apply_state_controller(K, obs)
        
        # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
        abs_force = abs(float(np.clip(force, -10, 10)))
        
        # change magnitute of the applied force in CartPole
        env.env.force_mag = abs_force

        # apply action
        obs, reward, done, _ = env.step(action)

        X[i,0] = obs[0]
        X[i,1] = obs[1]
        X[i,2] = obs[2]
        X[i,3] = obs[3]
        
        if i>399:
            break

    t_min = None
    flag = False
    for t in T:
        if all(np.abs(X[t,:])<=0.05) and not flag:
            t_min = t
            flag = True
        if any(np.abs(X[t,:])>0.05):
            t_min = None
            flag = False

    print('System states converge within 0.05 at time ',t_min)

    _, axs = plt.subplots(4)
    axs[0].plot(T,X[:400,0])
    axs[0].plot((0,400),(0,0),'k--')
    axs[0].legend([r'$x$'])
    axs[1].plot(T,X[:400,1])
    axs[1].plot((0,400),(0,0),'k--')
    axs[1].legend([r'$\dot{x}$'])
    axs[2].plot(T,X[:400,2])
    axs[2].plot((0,400),(0,0),'k--')
    axs[2].legend([r'$\theta$'])
    axs[3].plot(T,X[:400,3])
    axs[3].plot((0,400),(0,0),'k--')
    axs[3].legend([r'$\dot{\theta}$'])
    axs[3].set_xlabel('Iterations')
    plt.show()

# The main function
def main(args):
    # Create a Gym environment with the argument CartPole-v0 (already embedded in)
    env = gym.make(args.env)

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Print some stuff
    print("Environment:", args.env)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    set_seed(args.seed)    # seed for reproducibility
    env.env.seed(args.seed)
    
    mp, mk, l, g = env.masspole, env.masscart, env.length, env.gravity

    if args.mode == "control":
        control(env, mp, mk, l, g, args.time_sleep, terminate=True)
    elif args.mode == "multiple_R":
        multiple_R(env, mp, mk, l, g, args.time_sleep, terminate=True)

    env.close()

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

