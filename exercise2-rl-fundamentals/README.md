[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/rmdLebzk)
# Exercise2a
Exercise 2a - Reinforcement Learning Foundamentals

## Reinforcement Learning

All of the paramets on which to conduct the various test are stored in the `models` folder, in particular:
- `models/Cartpole-V0_x0_0.ai` contains the model trained to keep the cartpole around $x_{0}=0$
- `models/Cartpole-V0_x0_1_5.a1` contains the model trained to keep the cartpole around $x_{0}=1.5$
- `Cartpole-V0_reward3_1.ai` containts the model trained to make the cartpole move as fast as possible from left to right using as reward function
$$\tanh\left(\left|\dot{x}\right|^{4}\right)+0.5$$
- `Cartpole-V0_reward3_2.ai` containts the model trained to make the cartpole move as fast as possible from left to right using as reward function
$$ \left|\dot{x}\right|e^{0.001x^{10}}+0.5 $$

Currenlty, the file `cartpole_rl.py` is set to use the second reward function. The two reward functions are defined in the file `cartpole_rl.py`, respectively, with the functions `new_reward_1` and `new_reward_2`. To test the first reward function it is possible to change the line
```{python}
reward = new_reward_2(observation, x0)
```
to
```{python}
reward = new_reward_1(observation, x0)
```
inside the `test` function in the `cartpole_rl.py` file.