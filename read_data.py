from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import pickle
import tensorflow as tf
import numpy as np
import os
data_path = ".\\cifar10python"
IMAGE_SIZE = 32

def read_cifar10(filename_queue):

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  label_bytes = 1
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth

  record_bytes = label_bytes + image_bytes


  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  record_bytes = tf.decode_raw(value, tf.uint8)

  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])

  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result

def _load_label_names():

    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):

    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    images = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return images, labels

def _preprocess_and_save(normalize, one_hot_encode, images, labels, filename):
    """
    Preprocess data and save it to file
    """
    images = normalize(images)
    labels = one_hot_encode(labels)

    pickle.dump((images, labels), open(filename, 'wb'))


def preprocess_and_save_data(normalize, one_hot_encode):
    # Preprocess Training and Validation Data
    n_batches = 5
    valid_images = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        images, labels = load_cfar10_batch(data_path, batch_i)
        validation_count = int(len(images) * 0.1) #10% validation

        # Prprocess and save a batch of training data
        _preprocess_and_save(
            normalize,
            one_hot_encode,
            images[:-validation_count],
            labels[:-validation_count],
            'preprocess_batch_' + str(batch_i) + '.p')

        # Use a portion of training batch for validation
        valid_images.extend(images[-validation_count:])
        valid_labels.extend(labels[-validation_count:])

    # Preprocess and Save all validation data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(valid_images),
        np.array(valid_labels),
        'preprocess_validation.p')

    with open(data_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # load the test data
    test_images = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all training data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(test_images),
        np.array(test_labels),
        'preprocess_training.p')


def batch_images_labels(images, labels, batch_size):

    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        yield images[start:end], labels[start:end]


def load_preprocess_training_batch(batch_id, batch_size):
    #Load the Preprocessed Training data and return them in batches of <batch_size> or less
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    images, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_images_labels(images, labels, batch_size)

