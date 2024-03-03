#! /usr/bin/env python3

import rospy
import numpy as np

from single_pendulum_ros.msg import OutputData
from single_pendulum_ros.msg import StateData
from ExtendedKalmanFilter import ExtendedKalmanFilter

def callback(sensor_data):

    # EKF Estimation Step
    EKF.forwardDynamics()
    EKF.updateEstimate(sensor_data.y)

    ekf_data = StateData()
    ekf_data.x1 = EKF.posteriorMeans[-1][0]
    ekf_data.x2 = EKF.posteriorMeans[-1][1]

    pub.publish(ekf_data)


def ekf():
    # Subscriber Init
    rospy.init_node('ekf', anonymous=False)
    rospy.Subscriber('sensor_data',OutputData,callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        # Publisher Init
        pub = rospy.Publisher('ekf_data',StateData,queue_size=400)
        
        # Discretization time step (frequency of measurements)
        deltaTime=0.01

        # Initial true state
        x0 = np.array([np.pi/3, 0.5])
        # Initial state belief distribution (EKF assumes Gaussian distributions)
        x_0_mean = np.zeros(shape=(2,1))  # column-vector
        x_0_mean[0] = x0[0] + 3*np.random.randn()
        x_0_mean[1] = x0[1] + 3*np.random.randn()
        x_0_cov = 10*np.eye(2,2)  # initial value of the covariance matrix

        # Process noise covariance matrix (close to zero, we do not want to model noisy dynamics)
        Q=0.00001*np.eye(2,2)

        # Measurement noise covariance matrix for EKF
        R = np.array([[0.05]])

        # create the extended Kalman filter object
        EKF = ExtendedKalmanFilter(x_0_mean, x_0_cov, Q, R, deltaTime)

        ekf()
    except rospy.ROSInterruptException:
        pass