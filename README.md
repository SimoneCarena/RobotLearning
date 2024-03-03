# Robot Learning Homeworks

In this repository are contained the homeworks developed for the course "Robot Learning" of the Master Degree in Computer Engineering @ Politecnico di Torino.

## EKF
The first project, contained in `exercise1-ekf`, is an implemenation of and Extended Kalman Filter for a single and a double pendulum, and their respective implementaion in ROS

## RL Fundamentals
The second folder, `exercise2-rl-fundamentals`, contains introductory exercises to Reinforcement Learning performed on the cartpole evironment of `gym`

## Q-Learning and DQN
The third folder, `exercise3-qlearning`, contains and implementation of the `Q-Learning` algorithm for the cartpole environment. It aslo contains the implementation of a `DQN` algorithm for the same environment

## Policy Gradient
The last project, contained in `exercise4-policygradient`, is an implementation of a policy gradient method, in particular using a Gaussian Policy parametrization, to deal with the catrpole environment in a continuos action space. It is also present a version that uses `stable-baseline3` to train the RL agent.