#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib.slim as slim

from dataset_factory import get_split


def get_dataset(dataset_dir):
    return get_split('train', dataset_dir)


def get_network_fn(num_classes, is_training=False):
    func = lambda: None
    func.default_image_size = 299
    return func


def image_preprocessing_fn(image, height, width, scope=None):
    with tf.name_scope(scope, 'distort_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        distorted_image = tf.image.resize_images(image, [height, width])
        tf.image_summary('cropped_resized_image', tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        tf.image_summary('final_distorted_image',
                         tf.expand_dims(distorted_image, 0))
        distorted_image = tf.sub(distorted_image, 0.5)
        distorted_image = tf.mul(distorted_image, 2.0)
        return distorted_image


def main(dataset_dir, batch_size, log_dir):
    tf.logging.set_verbosity(tf.logging.INFO)

    ######################
    # Select the dataset #
    ######################
    dataset = get_dataset(dataset_dir)

    ####################
    # Select the network #
    ####################
    network_fn = get_network_fn(num_classes=dataset.num_classes, is_training=True)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              common_queue_capacity=20 * batch_size,
                                                              common_queue_min=10 * batch_size)

    image, label = provider.get(['image', 'label'])

    train_image_size = network_fn.default_image_size
    image = image_preprocessing_fn(image, train_image_size, train_image_size)

    images, labels = tf.train.batch([image, label],
                                    batch_size=batch_size,
                                    capacity=5 * batch_size)
    labels = slim.one_hot_encoding(labels, dataset.num_classes)
    batch_queue = slim.prefetch_queue.prefetch_queue([images, labels])

    ####################
    # Define the model #
    ####################

    ###########################
    # Kicks off the training. #
    ###########################
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # import ipdb; ipdb.set_trace()
        for _ in range(5):
            print(labels.eval())

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('dataset_dir')
    parser.add_argument('--log_dir', default='log/train')
    parser.add_argument('--batch_size', default=32)

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.DEBUG)
    main(**vars(args))
