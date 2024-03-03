import rospy
import numpy as np

from double_pendulum_ros.msg import OutputData
from double_pendulum_ros.msg import StateData
from ExtendedKalmanFilter import ExtendedKalmanFilter

def callback(sensor_data):

    # EKF Estimation Step
    EKF.forwardDynamics()
    z = np.array([sensor_data.y1, sensor_data.y2],dtype=float).reshape(2,1)
    EKF.updateEstimate(z)

    ekf_data = StateData()
    ekf_data.x1 = EKF.posteriorMeans[-1][0]
    ekf_data.x2 = EKF.posteriorMeans[-1][1]
    ekf_data.x3 = EKF.posteriorMeans[-1][2]
    ekf_data.x4 = EKF.posteriorMeans[-1][3]

    pub.publish(ekf_data)

def ekf():
    # Subscriber Init
    rospy.init_node('ekf', anonymous=False)
    rospy.Subscriber('sensor_data',OutputData,callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        # Publisher Init
        pub = rospy.Publisher('ekf_data',StateData,queue_size=10)
        
        # Discretization time step (frequency of measurements)
        deltaTime=0.01

        # Initial state belief distribution (EKF assumes Gaussian distributions)
        x_0_mean = np.zeros(shape=(4,1),dtype=float)  # column-vector
        x_0_cov = 10*np.eye(4)  # initial value of the covariance matrix

        # Process noise covariance matrix (close to zero, we do not want to model noisy dynamics)
        Q=0.00001*np.eye(4)

        # Measurement noise covariance matrix for EKF
        R = 0.05*np.eye(2)

        l1=1
        l2=2
        m1=1
        m2=2
        g=9.81

        # create the extended Kalman filter object
        EKF = ExtendedKalmanFilter(x_0_mean, x_0_cov, Q, R, deltaTime,l1,l2,m1,m2)

        ekf()
    except rospy.ROSInterruptException:
        pass