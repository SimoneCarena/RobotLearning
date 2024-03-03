import torch
import torch.nn.functional as F
import numpy as np
import random

class Policy(torch.nn.Module):
    def __init__(self, state_size, action_space_size):
        super().__init__()
        self.action_space = action_space_size
        self.lin1 = torch.nn.Linear(state_size,64)
        self.lin2 = torch.nn.Linear(64,64)
        self.lin3 = torch.nn.Linear(64,action_space_size)
    
    #Inputs the state and outputs the action-value for each action possible from that state
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
    
class ReplayBuffer:
    def __init__(self, max_size):
        self.memory = []
        self.max_size = max_size

    def push(self, transition):
        '''
        transition is composed of: (state, action, new_state, reward, done)
        '''
        self.memory.append(transition)
        if len(self.memory)>self.max_size:
            self.memory.pop(0)

    def sample(self, batch_size):
        return random.sample(self.memory,batch_size)

class Agent:
    def __init__(self, policy, max_mem_size, action_space_size, gamma):
        self.policy = policy
        self.memory = ReplayBuffer(max_mem_size)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.action_space_size = action_space_size
        self.gamma = gamma
    
    def take_action(self, state, eps, train=True):
        #Get q-values associated to current state
        with torch.no_grad():
            q_values = self.policy(torch.from_numpy(state))
        if train:
            #Choose random action
            if np.random.rand()<eps:
                action = np.random.choice(self.action_space_size)
            #Choose greedy action
            else:
                action = int(torch.argmax(q_values))
            return action
        else:
            return int(torch.argmax(q_values))
        
    def store_transition(self,transition):
        self.memory.push(transition)

    def update_policy(self, batch_size):
        batch = self.memory.sample(batch_size)
        loss = torch.zeros(1)
        for transition in batch:
            state = torch.from_numpy(transition[0])
            action = torch.tensor(transition[1])
            new_state = torch.from_numpy(transition[2])
            reward = torch.tensor(transition[3])
            done = torch.tensor(transition[4])
            with torch.no_grad():
                target = reward
                if done == False:
                    target += self.gamma*torch.max(self.policy(new_state))
            q_values = self.policy(state)
            loss += F.mse_loss(q_values[action],target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        

            