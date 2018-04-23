<<<<<<< HEAD
=======
#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""
>>>>>>> 6379dda3836162c11ce6b9673bca5136c74c71fb

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import turtlebot_model as tb
import os

tf.logging.set_verbosity(tf.logging.INFO)

class TurtleBotModel():
    def __init__(self, action, model_path="../TFModel_new"):
        # Create the Estimator
        self.classifier = tf.estimator.Estimator(
          model_fn=self.turtlebot_model_fn, model_dir=model_path)
        self.model_path = model_path

        # set action
        self.action = action
        # load test and train data
        if self.action == '1':
            self.train_data = np.load('train_data.npy')
            self.train_labels = self.readPath('../TrainingIMG')
        elif self.action == '2' or self.action == '3':
            self.eval_data = np.load('test_data.npy')
            self.eval_labels = self.readPath('../TestIMG')

    def readPath(self, path):
        self.fnames = []
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
        labels = np.asarray(label_list, dtype=np.int32)
        #self.labels = np.reshape(self.labels, (-1,1))
        print('Filenames:',self.fnames,'Labels', labels)
        return labels

    def flattened(self, x):
        layer_shape = x.get_shape()
        num_features = layer_shape[1:4].num_elements()
        product = 1
        for d in x.get_shape()[1:]:
            if d.value is not None:
                product *= d.value
        return tf.reshape(x, [-1, num_features])

    def turtlebot_model_fn(self, features, labels, mode):
      """Model function for CNN."""
      # Input Layer
      # Reshape X to 4-D tensor: [batch_size, width, height, channels]
      # MNIST images are 28x28 pixels, and have one color channel
      input_layer = tf.reshape(features["x"], [-1, 150, 150, 3])

      # Convolutional Layer #1
      # Computes 32 features using a 5x5 filter with ReLU activation.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, 28, 28, 1]
      # Output Tensor Shape: [batch_size, 28, 28, 32]
      conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

      # Pooling Layer #1
      # First max pooling layer with a 2x2 filter and stride of 2
      # Input Tensor Shape: [batch_size, 28, 28, 32]
      # Output Tensor Shape: [batch_size, 14, 14, 32]
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

      # Convolutional Layer #2
      # Computes 64 features using a 5x5 filter.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, 14, 14, 32]
      # Output Tensor Shape: [batch_size, 14, 14, 64]
      conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

      # Pooling Layer #2
      # Second max pooling layer with a 2x2 filter and stride of 2
      # Input Tensor Shape: [batch_size, 14, 14, 64]
      # Output Tensor Shape: [batch_size, 7, 7, 64]
      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

      # Flatten tensor into a batch of vectors
      # Input Tensor Shape: [batch_size, 7, 7, 64]
      # Output Tensor Shape: [batch_size, 7 * 7 * 64]
      #pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
      pool2_flat = self.flattened(pool2)

      # Dense Layer
      # Densely connected layer with 1024 neurons
      # Input Tensor Shape: [batch_size, 7 * 7 * 64]
      # Output Tensor Shape: [batch_size, 1024]
      dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

      # Add dropout operation; 0.6 probability that element will be kept
      dropout = tf.layers.dropout(
          inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

      # Logits layer
      # Input Tensor Shape: [batch_size, 1024]
      # Output Tensor Shape: [batch_size, 10]
      logits = tf.layers.dense(inputs=dropout, units=3)

      predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          "classes": tf.argmax(input=logits, axis=1),
          # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
          # `logging_hook`.
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      }
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

      # Calculate Loss (for both TRAIN and EVAL modes)
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

      # Configure the Training Op (for TRAIN mode)
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

      # Add evaluation metrics (for EVAL mode)
      eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])}
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


    def main(self):
      # Load training and eval data
      #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
      #self.train_data = mnist.train.images  # Returns np.array
      #self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
      #self.eval_data = mnist.test.images  # Returns np.array
      #self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

          #model_fn=turtlebot_model_fn, model_dir="/tmp/mnist_convnet_model")

      # Set up logging for predictions
      # Log the values in the "Softmax" tensor with label "probabilities"
      tensors_to_log = {"probabilities": "softmax_tensor"}
      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=50)

      #print(self.train_data, self.train_data.shape, self.train_labels, self.train_labels.shape)
      # Train the model
      if self.action == '1':
          train_input_fn = tf.estimator.inputs.numpy_input_fn(
              x={"x": self.train_data},
              y=self.train_labels,
              batch_size=100,
              num_epochs=None,
              shuffle=True)
          self.classifier.train(
              input_fn=train_input_fn,
              steps=20000,
              hooks=[logging_hook])

      # Evaluate the model and print results
      elif self.action == '2':
          eval_input_fn = tf.estimator.inputs.numpy_input_fn(
              x={"x": self.eval_data},
              y=self.eval_labels,
              num_epochs=1,
              shuffle=False)
          eval_results = self.classifier.evaluate(input_fn=eval_input_fn)
          print(eval_results)

      elif self.action == '3':
          pred_input_fn = tf.estimator.inputs.numpy_input_fn(
              x={"x": self.eval_data},
              num_epochs=1,
              shuffle=False)
          pred_results = self.classifier.predict(input_fn=pred_input_fn)
          print(list(pred_results))

if __name__ == "__main__":
  action = raw_input("Enter 1 for train and 2 for test and 3 for predict: ")
  model_path = raw_input("Give the name of the model directory:")
  t = TurtleBotModel(action, model_path)
  t.main()
  #tf.app.run(main=t.main())
