import tensorflow as tf
import numpy as np

def conv2d(x_tensor, conv_features, conv_filter, conv_strides):
    input_depth = x_tensor.get_shape().as_list()[-1]
    Weight = tf.Variable(tf.random_normal(shape=[conv_filter[0], conv_filter[1], input_depth, conv_features], stddev=0.1))
    bias = tf.Variable(tf.zeros(conv_features))
    conv = tf.nn.relu(tf.nn.conv2d(x_tensor, Weight, [1, conv_strides[0], conv_strides[1], 1], 'SAME') + bias)

    return conv

def max_pool(x_tensor, pool_size, pool_stride):

    return tf.nn.max_pool(x_tensor, [1, pool_size[0], pool_size[1], 1], [1, pool_stride[0], pool_stride[1] ,1], padding='SAME')

def flatten(x_tensor):
    shape = x_tensor.get_shape().as_list()

    return tf.reshape(x_tensor, [-1, np.prod(shape[1:])])

def fully_conn(x_tensor, num_outputs):
    shape = x_tensor.get_shape().as_list()
    Weight = tf.Variable(tf.random_normal([shape[-1], num_outputs], stddev=0.1))
    bias = tf.Variable(tf.zeros(num_outputs))

    return tf.nn.relu(tf.add(tf.matmul(x_tensor, Weight), bias))

def output(x_tensor, num_outputs):
    shape = x_tensor.get_shape().as_list()
    Weight = tf.Variable(tf.random_normal([shape[-1], num_outputs]))
    bias = tf.Variable(tf.zeros(num_outputs))

    return tf.add(tf.matmul(x_tensor, Weight), bias)