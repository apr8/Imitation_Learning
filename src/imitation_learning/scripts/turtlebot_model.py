from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import cv2

tf.logging.set_verbosity(tf.logging.INFO)

class TurtlebotModel(object):

    def __init__(self, tr_path, te_path, action, model_path):
	# have a place holder for x and y
	#selfdd.x =
        # define the path for the training or test data
        self.tr_path = tr_path
	self.te_path = te_path
        self.model_path = model_path

        # initialize all the file names and lables
	#self.labels = []
        # read the data from this path
        self.readPath(tr_path)

        # define what action to take
        if action == '2':
            self.action = 'TEST'
            os.chdir(te_path)
        elif action == '1':
            self.action = 'TRAIN'
            #os.chdir(tr_path)
        elif action == '3':
            self.action = 'PREDICT'
            #os.chdir(te_path)
        elif action == '4':
            self.action = 'SAVE_NP'
            os.chdir(te_path)
  	# Create the Estimator
   	self.classifier = tf.estimator.Estimator(
      	    model_fn=self.turtlebot_model, model_dir=self.model_path)
        # go to main
	#self.main()
        #self.x = tf.placeholder(tf.float32, [None, 150, 150, 3])
        #self.y = tf.placeholder(tf.int32, [None, 1])
        #self.train_x = np.load('train_data.npy')
        #self.test_x = np.load('')
    def readPath(self, path):
        self.fnames = []
	# TODO: Add lables here after splitting
        for root, dirs, files in os.walk(path):
            self.fnames.extend(files)

	label_list = []
 	for name in self.fnames:
	    l = name.split('_')[4]
	    l = l.split('.')[0]
	    if l == 'S':
		label = 0
	    elif l == 'L':
		label = 1
	    elif l == 'R':
		label = 2
	    elif l == 'B':
		label = 3
	    else:
		label = l
	    label_list.append(label)
	self.labels = np.asarray(label_list, dtype=np.int32)
	#self.labels = np.reshape(self.labels, (-1,1))
	print 'Filenames:',self.fnames,'Labels', self.labels

    def conv_layer_tf(self, x, conv=[5, 5], n_filters = 32):
	return tf.layers.conv2d(inputs=x, filters = n_filters, kernel_size = conv, padding="same", activation=tf.nn.relu)

    def max_pooling_tf(self, x, size = [2, 2], stri = 2):
	return tf.layers.max_pooling2d(inputs=x, pool_size = size, strides=stri)

    def flattened(self, x):
        layer_shape = x.get_shape()
        num_features = layer_shape[1:4].num_elements()
        product = 1
        for d in x.get_shape()[1:]:
            if d.value is not None:
                product *= d.value
        return tf.reshape(x, [-1, num_features])

    def turtlebot_model(self, features, labels, mode):

	for key, value in features.iteritems():
		print 'In model function, key:',key,'value', value
	print labels, type(features)
	self.x = tf.reshape(features["x"], [-1, 150, 150, 3])
	#self.y = labels
	#self.y = tf.cast(self.y, tf.int32)
	#self.y = tf.reshape(self.y, [-1, 1])
	# layer 1 convolution network
        print 'features:', features
	self.conv_layer1 = self.conv_layer_tf(self.x)

	# layer 2 max pool layer
	self.pool_layer1 = self.max_pooling_tf(self.conv_layer1)

	# layer 3 convolution layer
	self.conv_layer2  = self.conv_layer_tf(self.pool_layer1)

	# layer 4 max pool layer
	self.pool_layer2 = self.max_pooling_tf(self.conv_layer2)

	# create a dense layer
	self.pool_layer2_flat = self.flattened(self.pool_layer2)
	#self.pool_layer2_flat = tf.reshape(self.pool_layer2, [1,-1])

	self.dense1 = tf.layers.dense(inputs=self.pool_layer2_flat, units=1024, activation=tf.nn.relu)
	self.dropout1 = tf.layers.dropout(inputs=self.dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

	# add another dense layer or classify
	self.logits = tf.layers.dense(inputs=self.dropout1, units=3)
	#self.dense2 = tf.layers.dense(inputs=self.dropout1, units=2, activation=tf.nn.relu)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
      	    "classes": tf.argmax(input=self.logits, axis=1),
      	    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      	    # `logging_hook`.
            "probabilities": tf.nn.softmax(self.logits, name="softmax_tensor")
  	}
        print predictions
  	if mode == tf.estimator.ModeKeys.PREDICT:
    	    print 'In prediction'
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

 	# Calculate Loss (for both TRAIN and EVAL modes)
	#print 'y:',labels, type(self.y)
  	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self.logits)

  	# Configure the Training Op (for TRAIN mode)
  	if mode == tf.estimator.ModeKeys.TRAIN:
    	    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    	    train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
    	    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  	# Add evaluation metrics (for EVAL mode)
  	eval_metric_ops = {
      	    "accuracy": tf.metrics.accuracy(
             labels=self.y, predictions=predictions["classes"])}
  	return tf.estimator.EstimatorSpec(
      	    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def main(self):
  	    # Load training and eval data
  	    #train_data = mnist.train.images  # Returns np.array
   	    #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  	    #eval_data = mnist.test.images  # Returns np.array
  	    #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


  	    # Set up logging for predictions
  	    # Log the values in the "Softmax" tensor with label "probabilities"
  	    tensors_to_log = {"probabilities": "softmax_tensor"}
  	    logging_hook = tf.train.LoggingTensorHook(
      		tensors=tensors_to_log, every_n_iter=50)

	    #train_input_fn =
            if self.action == 'TRAIN':
  	    # Train the model
                print 'In Training'
                train_x = np.load('test_data.npy')
                #self.predict_input_fn()
                train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": train_x},
                    y=self.labels,
                    batch_size=1,
                    num_epochs=None,
                    shuffle=True)
  	        #self.classifier.train(
      	    	#    input_fn=lambda:self.imgs_input_fn(self.fnames, self.labels)                    ,steps=20000,
      	    	#    hooks=[logging_hook])
                self.classifier.train(
                    input_fn=train_input_fn,
                    steps=20000,
                    hooks=[logging_hook])

  	    # Evaluate the model and print results
  	    #eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      	    #	x={"x": eval_data},
      	    # 	y=eval_labels,
      	    #	num_epochs=1,
      	    # 	shuffle=False)
  	    #eval_results = classifier.evaluate(input_fn=eval_input_fn)

            if self.action == 'TEST':
                self.readPath(self.te_path)
  	        eval_results = self.classifier.evaluate(input_fn=lambda:self.imgs_input_fn(self.fnames, self.labels))
	        print(eval_results)
            if self.action == 'PREDICT':
                predict_x = np.load('test_data.npy')
                #self.predict_input_fn()
                #print self.in_x.shape
                pred_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": predict_x},
                    num_epochs=1,
                    shuffle=False)
  	        #predict_results = self.classifier.predict(input_fn=lambda:self.imgs_input_fn(self.fnames, self.labels))
  	        predict_results = self.classifier.predict(input_fn=pred_input_fn)
	        print (next(predict_results))
            if self.action == 'SAVE_NP':
                self.readPath(self.te_path)
                self.predict_input_fn()
                np.save('test_data.npy', self.in_x)


    def predict_input_fn(self):
        for c,f in enumerate(self.fnames):
            print 'IMg:', c
            img = cv2.imread(f)
            img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_CUBIC)
            img = np.reshape(img, ((-1,150,150,3)))
            img = np.float32(img)
            if c == 0:
                self.in_x = img
            else:
                self.in_x = np.append(self.in_x, img, axis = 0)

    def imgs_input_fn(self, filenames, labels=None, perform_shuffle=False, repeat_count=1, batch_size=1):
	    def _parse_function(filename, label):
                image_string = tf.read_file(filename)
                image = tf.image.decode_image(image_string, channels=3)
                image.set_shape([None, None, None])
                image = tf.image.resize_images(image, [150, 150])
                #image = tf.subtract(image, 116.779) # Zero-center by mean pixel
                image.set_shape([150, 150, 3])
                #image = tf.reverse(image, axis=[2]) # 'RGB'->'BGR'
                d = dict(zip(["x"], [image])), label
                return d
	    if labels is None:
		labels = [0]*len(filenames)
		labels=np.array(labels)
	    # Expand the shape of "labels" if necessory
	    if len(labels.shape) == 1:
		labels = np.expand_dims(labels, axis=1)
	    filenames = tf.constant(filenames)
	    labels = tf.constant(labels)
	    #labels = tf.cast(labels, tf.float32)
	    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
	    dataset = dataset.map(_parse_function)
	    if perform_shuffle:
	    # Randomizes input using a window of 256 elements (read into memory)
		dataset = dataset.shuffle(buffer_size=256)
	    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
	    dataset = dataset.batch(batch_size)  # Batch size to use
	    iterator = dataset.make_one_shot_iterator()
	    batch_features, batch_labels = iterator.get_next()
	    batch_features['x'] = tf.reshape(batch_features["x"], [-1, 150, 150, 3])
            batch_labels = tf.reshape(batch_labels, [-1,1])
            print batch_features, batch_labels
            return batch_features, batch_labels

if __name__ == "__main__":
  action = raw_input("Enter 1 for train and 2 for test and 3 for predict: ")
  model_path = raw_input("Give the name of the model directory:")
  #path = raw_input("Give the path of training data set :")
  tb = TurtlebotModel('../TrainingIMG','../TestIMG', action, model_path)
  #tb.main()
  tf.app.run(main = tb.main())
