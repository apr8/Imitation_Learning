
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import turtlebot_model as tb
import os

tf.logging.set_verbosity(tf.logging.INFO)

class TurtleBotModel():
    def __init__(self, action, model_path="../TFModel_reg"):
        # Create the Estimator
        self.classifier = tf.estimator.Estimator(
          model_fn=self.turtlebot_model_fn, model_dir=model_path)
        self.model_path = model_path

        # set action
        self.action = action
        # load test and train data
        if self.action == '1':
            self.train_data = np.load('train_scan_features_dagger.npy')
            self.train_labels = np.load('train_scan_labels_dagger_class.npy')
            print(self.train_data, self.train_labels)
        elif self.action == '2' or self.action == '3':
            # TODO Add test data
            pass
            #self.eval_data = np.load('test_data.npy')
            #self.eval_labels = self.readPath('../TestIMG')


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
      print('x',features["x"].shape)
      input_layer = tf.reshape(features["x"], [-1, 640, 1])
      #labels = tf.reshape(labels,[-1,2])
      #print(labels, input_layer)
      regularizer = tf.contrib.layers.l1_regularizer(scale=0.1)
      # Convolutional Layer #1
      # Computes 32 features using a 5x5 filter with ReLU activation.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, 28, 28, 1]
      # Output Tensor Shape: [batch_size, 28, 28, 32]
      conv1 = tf.layers.conv1d(
          inputs=input_layer,
          filters=64,
          kernel_size=(7),
          strides = 3,
          padding="same",
          activation=tf.nn.relu,
          kernel_regularizer=regularizer)

      # Pooling Layer #1
      # First max pooling layer with a 2x2 filter and stride of 2
      # Input Tensor Shape: [batch_size, 28, 28, 32]
      # Output Tensor Shape: [batch_size, 14, 14, 32]
      pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=(3), strides = 1)

      # Convolutional Layer #2V
      # Computes 64 features using a 5x5 filter.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, 14, 14, 32]
      # Output Tensor Shape: [batch_size, 14, 14, 64]
      conv2 = tf.layers.conv1d(
          inputs=pool1,
          filters=64,
          kernel_size=(3),
          padding="same",
          activation=tf.nn.relu,
          kernel_regularizer=regularizer)

      # Pooling Layer #2
      # Second max pooling layer with a 2x2 filter and stride of 2
      # Input Tensor Shape: [batch_size, 14, 14, 64]
      # Output Tensor Shape: [batch_size, 7, 7, 64]
      pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=(3), strides = 1)

      # Convolutional Layer #3
      # Computes 64 features using a 5x5 filter.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, 14, 14, 32]
      # Output Tensor Shape: [batch_size, 14, 14, 64]
      conv3 = tf.layers.conv1d(
          inputs=pool2,
          filters=64,
          kernel_size=(3),
          padding="same",
          activation=tf.nn.relu,
          kernel_regularizer=regularizer)

      # Convolutional Layer #4
      # Computes 64 features using a 5x5 filter.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, 14, 14, 32]
      # Output Tensor Shape: [batch_size, 14, 14, 64]
      conv4 = tf.layers.conv1d(
          inputs=conv3,
          filters=64,
          kernel_size=(3),
          padding="same",
          activation=tf.nn.relu,
          kernel_regularizer=regularizer)

      # Pooling Layer #2
      # Second max pooling layer with a 2x2 filter and stride of 2
      # Input Tensor Shape: [batch_size, 14, 14, 64]
      # Output Tensor Shape: [batch_size, 7, 7, 64]
      pool_avg = tf.layers.average_pooling1d(inputs=conv4, pool_size=(3), strides = 1)

      # Flatten tensor into a batch of vectors
      # Input Tensor Shape: [batch_size, 7, 7, 64]
      # Output Tensor Shape: [batch_size, 7 * 7 * 64]
      #pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
      pool2_flat = self.flattened(pool_avg)

      # Dense Layer #1
      # Densely connected layer with 1024 neurons
      # Input Tensor Shape: [batch_size, 7 * 7 * 64]
      # Output Tensor Shape: [batch_size, 1024]
      dense1 = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu, kernel_regularizer=regularizer)

      # Dense Layer #2
      # Densely connected layer with 1024 neurons
      # Input Tensor Shape: [batch_size, 7 * 7 * 64]
      # Output Tensor Shape: [batch_size, 1024]
      dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu, kernel_regularizer=regularizer)

      # Dense Layer #3
      # Densely connected layer with 1024 neurons
      # Input Tensor Shape: [batch_size, 7 * 7 * 64]
      # Output Tensor Shape: [batch_size, 1024]
      dense = tf.layers.dense(inputs=dense2, units=256, activation=tf.nn.relu, kernel_regularizer=regularizer)

      # Add dropout operation; 0.6 probability that element will be kept
      dropout = tf.layers.dropout(
          inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

      # Output layer
      # Input Tensor Shape: [batch_size, 256]
      # Output Tensor Shape: [batch_size, 2]
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
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
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
      #tensors_to_log = {"vel"}
      #logging_hook = tf.train.LoggingTensorHook(
      #    tensors=tensors_to_log, every_n_iter=50)
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
              steps=200000)
              #hooks=[logging_hook])

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
