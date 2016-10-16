#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib.slim as slim

from dataset_factory import get_split


def get_dataset(dataset_dir):
    return get_split('train', dataset_dir, file_pattern=None, reader=None)


def main(dataset_dir, batch_size, log_dir):
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset = get_dataset(dataset_dir)
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              common_queue_capacity=20 * batch_size,
                                                              common_queue_min=10 * batch_size)

    image, label = provider.get(['image', 'label'])
    images, labels = tf.train.batch([image, label],
                                    batch_size=batch_size,
                                    capacity=5 * batch_size)
    # labels = slim.one_hot_encoding(labels, dataset.num_classes)
    # batch_queue = slim.prefetch_queue.prefetch_queue([images, labels])

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # import ipdb; ipdb.set_trace()
        for _ in range(5):
            print(label.eval())

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
