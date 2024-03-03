import rospy
import numpy as np

from double_pendulum_ros.msg import StateData
from double_pendulum_ros.msg import OutputData

def callback(real_data):

    # Noise the measurements
    measurement_noise_var = 0.05 # Actual measurement noise variance
    z = np.zeros((2,1), dtype=float)
    z[0] = real_data.x1 + np.sqrt(measurement_noise_var)*np.random.randn()
    z[1] = real_data.x2 + np.sqrt(measurement_noise_var)*np.random.randn()

    # Send noisy measurements
    sensor_data = OutputData()
    sensor_data.y1 = z[0]
    sensor_data.y2 = z[1]

    pub.publish(sensor_data)


def sensor():

    # Subscriber Init
    rospy.init_node('sensor', anonymous=False)
    rospy.Subscriber('real_data',StateData,callback)

    rospy.spin()


if __name__ == '__main__':
    try:
        # Publisher Init
        pub = rospy.Publisher('sensor_data',OutputData,queue_size=1)
        sensor()
    except rospy.ROSInterruptException:
        pass