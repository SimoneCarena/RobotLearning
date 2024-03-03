import gym
import numpy as np
from DQN import Policy, Agent
import torch
import matplotlib.pyplot as plt

def train(train_len):
    env = gym.make("CartPole-v0")
    state_size = 4
    action_space_size = env.action_space.n
    policy = Policy(state_size,action_space_size)
    batch_size = 32
    returns = []
    agent = Agent(policy, 256, action_space_size, gamma)
    for t in range(train_len):
        epsilon = b/(b+t)
        state, done = env.reset(), False
        return_value = 0
        while not done:
            action = agent.take_action(state, epsilon)
            new_state, reward, done, _ = env.step(action)
            return_value+=reward
            agent.store_transition((state, action, new_state, reward, done))
            state = new_state
        print(f'Episode {t} -> Return={return_value}')
        if len(agent.memory.memory)>=batch_size:
            agent.update_policy(batch_size)
        returns.append(return_value)

    plt.plot(returns)
    plt.plot([np.mean(returns[max(0,i-100):i]) for i in range(len(returns))])
    plt.show()

    torch.save(agent.policy,'policy.pth')

def test(test_len):
    env = gym.make("CartPole-v0")
    action_space_size = env.action_space.n
    policy = torch.load('policy.pth')
    returns = []
    agent = Agent(policy, 256, action_space_size, gamma)

    for t in range(test_len):
        epsilon = b/(b+t)
        state, done = env.reset(), False
        return_value = 0
        while not done:
            action = agent.take_action(state, epsilon,False)
            new_state, reward, done, _ = env.step(action)
            return_value+=reward
            state = new_state
        returns.append(return_value)

    plt.plot(returns)
    plt.plot([np.mean(returns[max(0,i-100):i]) for i in range(len(returns))])
    plt.legend(['Return','100-Episodes Average Return'])
    plt.show()


if __name__ == "__main__":

    train_episodes = 20000
    test_episodes = 5000
    b = 2222
    gamma = 0.98
    training = False

    if training:
        train(train_episodes)
    test(test_episodes)