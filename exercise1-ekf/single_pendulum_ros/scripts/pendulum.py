#! /usr/bin/env python3

import rospy

import numpy as np
from scipy.integrate import odeint
from single_pendulum_ros.msg import StateData

# System dynamics (continuous, non-linear) in state-space representation (https://en.wikipedia.org/wiki/State-space_representation)
def stateSpaceModel(x,t):
    """
        Dynamics may be described as a system of first-order
        differential equations: 
        dx(t)/dt = f(t, x(t))
    """
    g=9.81
    l=1
    dxdt=np.array([x[1], -(g/l)*np.sin(x[0])])
    return dxdt

def discreteTimeDynamics(x_t, dT):
        """
            Forward Euler integration.
            
            returns next state as x_t+1 = x_t + dT * (dx/dt)|_{x_t}
        """
        x_tp1 = x_t + dT*stateSpaceModel(x_t, None)
        return x_tp1

def pendulum():
    # Discretization time step (frequency of measurements)
    deltaTime=0.01

    # Initial true state
    x0 = np.array([np.pi/3, 0.5])

    # Publisher Init
    pub = rospy.Publisher('real_data',StateData,queue_size=1)
    rospy.init_node('pendulum',anonymous=False)
    rate = rospy.Rate(10) # 10 Hz

    x_old = discreteTimeDynamics(x0, deltaTime)

    # Send messages
    while not rospy.is_shutdown():

        x_new = discreteTimeDynamics(x_old, deltaTime)
        x_old = x_new

        real_data = StateData()
        real_data.x1 = x_new[0]
        real_data.x2 = x_new[1]

        pub.publish(real_data)
        rate.sleep()


if __name__ == "__main__":
    try:
        pendulum()
    except rospy.ROSInterruptException:
        pass