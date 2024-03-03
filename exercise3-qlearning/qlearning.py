import gym
import numpy as np
from matplotlib import animation, pyplot as plt
from time import sleep
import random
import seaborn as sns
import pandas as pd

import sys

np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

# GLIE
glie = True

# Whether to perform training or use the stored .npy file
MODE = 'TRAINING' # TRAINING, TEST

episodes = 20000
test_episodes = 100
num_of_actions = 2  # 2 discrete actions for Cartpole

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Parameters
gamma = 0.98
alpha = 0.1
constant_eps = 0.2
b = 2222

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

# Initialize Q values
q_grid = np.zeros((discr, discr, discr, discr, num_of_actions))

if MODE == 'TEST':
    q_grid = np.load('q_values.npy')

def find_nearest(array, value):
    return np.argmin(np.abs(array - value))

def get_cell_index(state):
    """Returns discrete state from continuous state"""
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av


def get_action(state, q_values, greedy=False):
    x, v, th, av = get_cell_index(state)

    if greedy: # TEST -> greedy policy
        best_action_estimated = np.argmax(q_values[x,v,th,av]) #greedy w.r.t. q_grid
        # print('best estimaed action greedy: ',best_action_estimated)

        return int(best_action_estimated)

    else: # TRAINING -> epsilon-greedy policy
        if np.random.rand() < epsilon:
            # Random action
            action_chosen = np.random.choice([0,1]) #choose random action with equal probability among all actions
            
            return int(action_chosen)
        else:
            # Greedy action
            best_action_estimated = np.argmax(q_values[x,v,th,av]) #greedy w.r.t. q_grid

            return int(best_action_estimated)


def update_q_value(old_state, action, new_state, reward, done, q_array):
    old_cell_index = get_cell_index(old_state)
    x_old = old_cell_index[0]
    v_old = old_cell_index[1]
    th_old = old_cell_index[2]
    av_old = old_cell_index[3]

    new_cell_index = get_cell_index(new_state)
    x_new = new_cell_index[0]
    v_new = new_cell_index[1]
    th_new = new_cell_index[2]
    av_new = new_cell_index[3]

    # Target value used for updating our current Q-function estimate at Q(old_state, action)
    if done is True:
        target_value = reward  # HINT: if the episode is finished, there is not next_state. Hence, the target value is simply the current reward.
    else:
        target_value = reward + gamma*np.max(q_array[x_new,v_new,th_new,av_new])

    # Update Q value
    error = target_value - q_array[x_old,v_old,th_old,av_old,action]
    q_grid[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] = q_array[x_old,v_old,th_old,av_old,action] + alpha*error

    return


# Training loop
ep_lengths, epl_avg = [], []
returns = []
epsilons = []
value_functions = []
i=0 #Episode Counter
for ep in range(episodes+test_episodes):
    test = ep > episodes

    if MODE == 'TEST':
        test = True

    state, done, steps = env.reset(), False, 0

    if glie:
        epsilon = b/(b+i)
        epsilons.append(epsilon)
        i+=1
    else:
        epsilon = constant_eps

    return_value = 0

    while not done:
        action = get_action(state, q_grid, greedy=test)
        new_state, reward, done, _ = env.step(action)
        return_value+=reward
        if not test:
            update_q_value(state, action, new_state, reward, done, q_grid)
        else:
            pass
            #env.render()

        state = new_state
        steps += 1

    returns.append(return_value)
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        # print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))
        # print('Epsilon:', epsilon)
        pass

    #Value Function
    value_function = np.max(q_grid,axis=4)
    value_function = np.mean(value_function, axis=(1,3))
    value_functions.append(value_function)
    
if glie:
    #Return Plot
    plt.plot(returns)
    plt.plot(epl_avg) #Same as return/ep_len
    plt.title(r'Training with GLIE $\epsilon$')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.show()

    #Epsilon Evolution
    plt.plot(epsilons)
    plt.title(r'Varation of $\epsilon$ with GLIE')
    plt.xlabel('Episodes')
    plt.ylabel(r'$\epsilon$')
    plt.show()
else:
    #Return Plot
    plt.plot(returns)
    plt.plot(epl_avg) #Same as return/ep_len
    plt.title(r'Training with Constant $\epsilon=0.2$')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.show()

#Value Function
value_function = np.max(q_grid,axis=4)
value_function = np.mean(value_function, axis=(1,3))

#Heatmap 
plt.imshow(value_function)
plt.title(r'Optimal Value Function $\nu^{*}$ Heatmap')
plt.xlabel(r'$x$')
plt.ylabel(r'$\theta$')
plt.colorbar()
plt.show()

if MODE == 'TEST':
    sys.exit()

# Save the Q-value array
np.save("q_values.npy", q_grid)