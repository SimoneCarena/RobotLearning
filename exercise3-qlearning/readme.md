# Exercise 3

The file ```function_approx_q.py``` implements function approximation via neural network. The network parameters are updated using the following cost function
$$
\mathcal{L}=\mathbb{E}_{m\sim\mathcal{D}}\left[\left( r+\max_{a'}Q(s',a')-Q(s,a) \right)  \right]
$$
where $m$ is the transition $(s,a,r,s')$ sampled from the replay memory $\mathcal{D}$.

The file ```function_approx_q2.py``` implements function approximation via neural network. The network parameters are updated using the following cost function
$$
\mathcal{L}=\mathbb{E}_{m\sim\mathcal{D}}\left[\left( r+\max_{a'}Q(s',a',\boldsymbol{\theta'})-Q(s,a,\boldsymbol{\theta}) \right)  \right]
$$
where $\boldsymbol{\theta'}$ referes to the parameters of the old policy and $\boldsymbol{\theta}$ refers to the parameters of the policy currently in use.

The agent for the first file is implemented in ```DQN.py```, and the agent for the second is ```DQN2.py```