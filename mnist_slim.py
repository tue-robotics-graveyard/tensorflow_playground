#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.examples.tutorials.mnist as mnist

batch_size = 1000
IMAGE_SIZE = mnist.mnist.IMAGE_SIZE


def lenet(images):
    net = slim.layers.fully_connected(images, 20, scope='fully_connected4')
    net = slim.layers.fully_connected(net, 10, activation_fn=None, scope='fully_connected5')
    return net


def inputs(one_hot_labels=False):
    data = mnist.input_data.read_data_sets("MNIST_data/", one_hot=one_hot_labels)
    return data.train.images, data.train.labels


def main(log_dir):
    input_images, input_labels = inputs(one_hot_labels=True)

    with tf.name_scope('input'):
        image, label = tf.train.slice_input_producer([input_images, input_labels], num_epochs=None)
        label = tf.cast(label, tf.int32)
        # image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 1])
        images, labels = tf.train.batch([image, label], batch_size=batch_size)

    predictions = lenet(images)

    slim.losses.softmax_cross_entropy(predictions, labels)
    total_loss = slim.losses.get_total_loss()
    tf.scalar_summary('loss', total_loss)

    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
    train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

    print('training started')
    slim.learning.train(train_op, log_dir, save_summaries_secs=20, log_every_n_steps=1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--log_dir', default='log/train')

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.DEBUG)
    main(**vars(args))
