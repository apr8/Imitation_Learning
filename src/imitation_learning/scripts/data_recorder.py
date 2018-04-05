#!/usr/bin/env python

##import cPickle as pickle
import numpy as np
import rospy
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
import cv2
print cv2.__version__
import time
from threading import Lock


"""
This node saves multiple image topics as pictures and encodes the filename with sequence,time,throttle,steering.
I know what rosbag is.
Note:
Throttle values 0.0-0.5 = reverse and 0.5-1.0 = forward
Steering values 0.0-0.5 = left and 0.5-1.0 = right, 0.5 is center position
This node is recieving the output topic of drop nodes that are dropping 2 out of 3 frames for the depth and rgb image topics.
The node struggles to record both rgb and depth topic at 30FPS with cv2.imwrite on the TX1 which is why messages are being dropped plus 30FPS seems to be unnecessary
Gstreamer could be used to create a similar function like cv2.imwrite that utilizes the onboard hardware encoders for jpg (nvjpg) this could allow for more FPS or image topics to be saved as jpg
Location of folders where images are saved are /home/ubuntu/DepthIMG/ /home/ubuntu/LidarIMG/ /home/ubuntu/TrainingIMG/

"""

class dataRecorder(object):

    def __init__(self):
        print "dataRecorder"
        self.record = True
        self.twist = None
        self.twistLock = Lock()
        self.bridge = CvBridge()
        self.globaltime = None

        # subscribe to camaera raw image
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.streamCB)

        #subscribe to cmd velocity for teleop
        rospy.Subscriber("/cmd_vel_mux/input/teleop", Twist, self.cmd_velCB)

        rospy.init_node('dataRecorder',anonymous=True)
        rospy.spin()

    """
    Receives an Image message and encodes the sequence,timestamp,throttle and steering values into the filename and saves it as a jpg
    """
    def streamCB(self, pic):
        if self.record == True:
            #rospy.loginfo("image recieved")
            try:
                cv2image = self.bridge.imgmsg_to_cv2(pic)
                if self.twist is not None:
                    fname = None
                    seq = str(pic.header.seq)
                    timestamp = str(pic.header.stamp)
                    with self.twistLock:
                        fname = seq + '_' + timestamp + '_' + str(round(self.twist.linear.x,8)) + '_' + str(round(self.twist.angular.z,8))
                    cv2.imwrite("../TrainingIMG/"+fname+".jpg",cv2image)
            except CvBridgeError as e:
                print(e)
        else:
            rospy.loginfo("Not Recording from kinect")

    """
    Receives a twist msg
    """
    def cmd_velCB(self, msg):
        ##rospy.loginfo("Linear: [%f, %f, %f]"%(msg.linear.x, msg.linear.y, msg.linear.z))
        ##rospy.loginfo("Angular: [%f, %f, %f]"%(msg.angular.x, msg.angular.y, msg.angular.z))
        with self.twistLock:
            self.twist = msg


if __name__ == '__main__':
    try:
        dataRecorder()
    except rospy.ROSInterruptException:
        pass
