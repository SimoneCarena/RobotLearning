# Exercises 1 -  Extended Kalman Filter

Every part of the assignement is contained in its own folder. Folders ending in `ros` contain the ROS ssources needed to run the ROS simulation

## Single Pendulum 

The python simulation for the single pendulum runs for 400 iterations. The plot of the ekf is presented when running the `single_pendulum/main.py` file.

## Single Pendulum ROS

This folder contains the files and directories needed to run the single pendulum's simulation on ROS. To properly run it then following steps are to be followed:
- Execute the `roscore` command to start the master's node
- Run the `rosrun single_pendulum pendulum.py` command to start the real data generation via the pendulum node
- Run the `rosrun single_pendulum sensor.py` command to start the sensor node, which reads the data sent by the first node and outputs the noisy measurements
- Run the `rosrun single_pendulum ekf.py` command to start the ekf node, which reads the noisy measurements from the sensor and outputs the filtered states
- Run a plot command (e.g. `rqt_plot`) to visulize the various states

All the source files are contained inside the `scripts` folder.

The Simulation runs contiously.

## Double Pendulum

The python simulation for the double pendulum runs for 4000 iterations. An animation of the pendulum evolving and the plot of the ekf is presented when running the `double_pendulum/main.py` file.

## Double Pendulum ROS

This folder contains the files and directories needed to run the double pendulum's simulation on ROS. To properly run it then following steps are to be followed:
- Execute the `roscore` command to start the master's node
- Run the `rosrun double_pendulum pendulum.py` command to start the real data generation via the pendulum node
- Run the `rosrun double_pendulum sensor.py` command to start the sensor node, which reads the data sent by the first node and outputs the noisy measurements
- Run the `rosrun double_pendulum ekf.py` command to start the ekf node, which reads the noisy measurements from the sensor and outputs the filtered states
- Run a plot command (e.g. `rqt_plot`) to visulize the various states

All the source files are contained inside the `scripts` folder.

The Simulation runs contiously.