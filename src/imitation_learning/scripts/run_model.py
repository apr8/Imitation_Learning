from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import rospy
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import String
from sensor_msgs.msg import Image
import turtlebot_model2 as tb
import cv2
from cv_bridge import CvBridge, CvBridgeError

class RunModel(object):

    def __init__(self):
        # create subscribers and publishers
        self.createSubAndPub()

        # list of lables
        self.lables = ['S','L','R','B']
        self.bridge = CvBridge()

        # create and keep ros node running
        rospy.init_node('TurtlebotRunModel',anonymous=True)
        print("Running model")

        # create the object of the model
        self.tb = tb.TurtleBotModel(None)
        # initialize the model
        #self.classifier = tf.estimator.Estimator(model_fn=self.tb.turtlebot_model, model_dir="../TFModel_3")

        rospy.spin()

    def createSubAndPub(self):
        # subscribe to camaera raw image
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.imageCB)

        #subscribe to cmd velocity for teleop
        self.vel_pub = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size=10)


    def imageCB(self, img):

        # once we recive the image change to the format the tensorflow model can use to predict

        cv2image = self.bridge.imgmsg_to_cv2(img)
        cv2image = cv2.resize(cv2image, (150,150), interpolation = cv2.INTER_LINEAR)
        np_in = np.reshape(cv2image, (1,150,150,3))
        np_in = np.float32(np_in)
        #print type(cv2image), cv2image.shape
        # Print out predictions
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": np_in},
                num_epochs=1,
                shuffle=False)
        np_str = np.fromstring(img.data, np.uint8)
        #print np_str.shape
        #np_arr = np_str.reshape(img.height, img.width, 3)
        #tf_img = tf.convert_to_tensor(np_arr, dtype=tf.uint8)
        #image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        #print np_arr, np_arr.shape, tf_img.shape
        prediction = (self.tb.classifier.predict(input_fn = predict_input_fn))
        #prediction = self.tb.classifier.predict(input_fn = lambda:self.predict_input_fn(tf_img), predict_keys = ['0','1','2'])
        #print type(prediction), prediction
        #pre = [p["classes"] for p in prediction]
        drive = next(prediction)['classes']
        print drive
        self.turtlebotDrive(drive)
        #print pre
        #for i, p in enumerate(prediction):
        #    print i,':',p
        #train_input_fn = tf.estimator.inputs.numpy_input_fn(
        #    x={"x": train_data},
      	#    y=None,
      	#    batch_size=100,
      	#    num_epochs=None,
      	#    shuffle=True)

    def turtlebotDrive(self, drive):
        # turn based on the label drive
        twist = Twist()

        if drive == 0:
            twist.linear.x = 0.1
        elif drive == 1:
            twist.angular.z = 0.1
        elif drive == 2:
            twist.angular.z = -0.1

        self.vel_pub.publish(twist)

    def predict_input_fn(self, img, labels=None, perform_shuffle=False, repeat_count=1, batch_size=1):
            def _parse_function(img):
                print "parse", img.shape
                img.set_shape([None, None, None])
                tf_img = tf.image.resize_images(img, [150, 150])
                tf_img.set_shape([150, 150, 3])
                tf_img_dict = dict(zip(['in_1'], [tf_img]))
                #image = tf.reverse(image, axis=[2]) # 'RGB'->'BGR'
                return tf_img_dict
            #i = tf.Variable(img)
            print img.shape

            dataset = tf.data.Dataset.from_tensors(img)
            dataset = dataset.map(_parse_function)
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next()

            return features

if __name__ == "__main__":

    r = RunModel()
