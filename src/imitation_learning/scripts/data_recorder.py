#!/usr/bin/env python

##import cPickle as pickle
import numpy as np
import rospy
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
import sys
import cv2
import tf
from move_base_msgs.msg import MoveBaseActionResult, MoveBaseActionGoal
print cv2.__version__
import time
from threading import Lock
import os
import roslib
import rospy


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

    def __init__(self, action):
        print "Initializing Data Recorder"

        # create a set of actions
        self.action = action

        # create class variables
        self.createClassVariables()

        # create a directory to store the training data
        self.createDirectory()

        # create subscribers and publishers
        self.createSubAndPub()

        # list of lables
        self.lables = ['S','L','R','B']

        # create and keep ros node running
        rospy.init_node('dataRecorder',anonymous=True)

        # invoke shut down call back
        rospy.on_shutdown(self.shutdownCB)

        print("Recording data")
        rospy.spin()

    def createSubAndPub(self):

        if self.action == '1':
	        # subscribe to raw scan data
            rospy.Subscriber("/scan", LaserScan, self.stream_scanCB)

        elif self.action == '2':
            # subscribe to camaera raw image
            rospy.Subscriber("/camera/rgb/image_raw", Image, self.streamCB)

        #subscribe to cmd velocity for teleop
        rospy.Subscriber("/cmd_vel_mux/input/teleop", Twist, self.cmd_velCB)
        rospy.Subscriber("/cmd_vel_mux/input/navi", Twist, self.cmd_velCB)

        # move base subscribes
        rospy.Subscriber("/move_base/goal", MoveBaseActionGoal, self.moveBaseGoal)
        rospy.Subscriber("/move_base/result", MoveBaseActionResult, self.moveBaseResult)
        self.listener = tf.TransformListener()

    def createDirectory(self, directory='../TestIMG'):
	# create the folder to save the training data if not available
	if not os.path.exists(directory):
    	    os.makedirs(directory)

    def moveBaseGoal(self, data):
        pass
    def moveBaseResult(self, data):
        pass

    def createClassVariables(self):
        self.record = True
        self.twist = None
        self.twistLock = Lock()
        self.bridge = CvBridge()
        self.globaltime = None
        self.reg_vel = []
        self.scan_feature = []
        self.scan_label_class = []

    def stream_scanCB(self, scan):
        """
        Receives an scan message and encodes the scan msg and corresponing labels
        """
        scan_np = np.asarray(scan.ranges, np.float32)
        scan_np = np.reshape(scan_np, (-1,1))
        idx = np.where(np.isnan(scan_np))
        #print scan_np, scan_np.shape
        #print idx
        scan_np[idx] = scan.range_max
        scan_np -= np.mean(scan_np)
        #print scan_np, scan_np.shape
        if self.record == True:
            #rospy.loginfo("image recieved")
            try:
                if self.twist is not None and (self.twist.linear.x != 0.0 or self.twist.angular.z != 0.0):
                    twist = np.zeros((2))
		    with self.twistLock:
                        twist[0] = self.twist.linear.x
                        twist[1] = self.twist.angular.z
			if self.twist.angular.z > 0.0:
			    lable = 1
			elif self.twist.angular.z < 0.0:
			    lable = 2
			elif self.twist.angular.z == 0:
			    if self.twist.linear.x > 0.0:
				lable = 0
			    else:
				lable = 3
			print lable, type(lable)
                    self.scan_label_class.append(lable)
                    self.reg_vel.append(twist)
                    self.scan_feature.append(scan_np)
                    print 'DataNumber:', len(self.scan_feature)
            except CvBridgeError as e:
                print(e)
        else:
            rospy.loginfo("Not Recording from kinect")

    def shutdownCB(self):
        print "Now shutting down"
        if self.action == '1':
            labels = np.asarray(self.reg_vel)
            features = np.asarray(self.scan_feature)
            label_class = np.asarray(self.scan_label_class)
            print labels.shape, features.shape
            np.save('train_scan_features_dagger.npy', features)
            np.save('train_scan_labels_dagger.npy', labels)
            np.save('train_scan_labels_dagger_class.npy', label_class)

    def streamCB(self, pic):
        """
        Receives an Image message and encodes the sequence,timestamp,throttle and steering values into the filename and saves it as a jpg
        """
        if self.record == True:
            #rospy.loginfo("image recieved")
            try:
                cv2image = self.bridge.imgmsg_to_cv2(pic)
                if self.twist is not None and (self.twist.linear.x != 0.0 or self.twist.angular.z != 0.0):
                    fname = None
                    seq = str(pic.header.seq)
                    timestamp = str(pic.header.stamp)
		    with self.twistLock:
			if self.twist.angular.z > 0.0:
			    lable = 'L'
			elif self.twist.angular.z < 0.0:
			    lable = 'R'
			elif self.twist.angular.z == 0:
			    if self.twist.linear.x > 0.0:
				lable = 'S'
			    else:
				lable = 'B'
			print lable, type(lable)
                        fname = seq + '_' + timestamp + '_' + str(round(self.twist.linear.x,8)) + '_' + str(round(self.twist.angular.z,8)) + '_' + lable
                    cv2.imwrite("../TestIMG/"+fname+".png",cv2image)
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
    i = raw_input("Enter the data to record 1:scan, 2:images : ")
    try:
        dataRecorder(i)
    except rospy.ROSInterruptException:
        pass
