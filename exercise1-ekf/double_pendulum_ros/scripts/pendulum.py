#! /usr/bin/env python3

import rospy

import numpy as np
from scipy.integrate import odeint
from double_pendulum_ros.msg import StateData
from numpy import sin, cos

deltaTime=0.01

# Initial true state
x0 = np.array([np.pi/2, 0, np.pi/2, 0])

l1=1
l2=2
m1=1
m2=2
g=9.81

def stateSpaceModel(x,t):
    """
        Dynamics may be described as a system of first-order
        differential equations: 
        dx(t)/dt = f(t, x(t))
    """
    th1 = x[0]
    th2 = x[2]
    th1d = x[1]
    th2d = x[3]

    dxdt=np.array(
        [x[1], 
        -(l1*m2*cos(th1 - th2)*sin(th1 - th2)*th1d**2 + l2*m2*sin(th1 - th2)*th2d**2 + g*m1*sin(th1) + g*m2*sin(th1) - g*m2*cos(th1 - th2)*sin(th2))/(l1*(- m2*cos(th1 - th2)**2 + m1 + m2)),
        x[3],
        (g*m1*cos(th1 - th2)*sin(th1) - g*m2*sin(th2) - g*m1*sin(th2) + g*m2*cos(th1 - th2)*sin(th1) + l1*m1*th1d**2*sin(th1 - th2) + l1*m2*th1d**2*sin(th1 - th2) + l2*m2*th2d**2*cos(th1 - th2)*sin(th1 - th2))/(l2*(- m2*cos(th1 - th2)**2 + m1 + m2))
    ])
    
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

    x0 = np.array([np.pi/2, 0, np.pi/2, 0],dtype=float)

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
        real_data.x3 = x_new[2]
        real_data.x4 = x_new[3]

        pub.publish(real_data)
        rate.sleep()


if __name__ == "__main__":
    try:
        pendulum()
    except rospy.ROSInterruptException:
        pass