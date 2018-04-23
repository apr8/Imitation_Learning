from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import cv2
import tensorflow as tf
import pickle

import os
import cv2
import numpy as np
import time

class DataRead(object):

    def __init__(self, path):

	# define the path for the training or test data
	self.path = path

	# initialize all the file names
	self.fnames = []
	# read the data from this path
	self.readPath(path)


    def readPath(self, path):

	for root, dirs, files in os.walk(path)
		self.fnames.extend(files)
