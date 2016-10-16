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
    def model(images):
        net = slim.layers.fully_connected(images, 20, scope='fully_connected')
        return net, None
    model.default_image_size = 299
    return model


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
    ground_truth_input = slim.one_hot_encoding(labels, dataset.num_classes)

    ####################
    # Define the model #
    ####################
    logits, end_points = network_fn(images)
    predictions = tf.argmax(logits, 1)

    import ipdb; ipdb.set_trace()
    # Choose the metrics to compute:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'eval/accuracy': slim.metrics.accuracy(predictions, labels),
        # 'eval/precision': slim.metrics.precision(predictions, labels),
        # "eval/mean_absolute_error": slim.metrics.mean_absolute_error(predictions, labels),
        # "eval/mean_squared_error": slim.metrics.mean_squared_error(predictions, labels),
    })


    #############################
    # Specify the loss function #
    #############################
    slim.losses.softmax_cross_entropy(logits, labels)

    total_loss = slim.losses.get_total_loss()
    tf.scalar_summary('losses/total_loss', total_loss)
    optimizer = tf.train.GradientDescentOptimizer(0.001)

    # create_train_op that ensures that when we evaluate it to get the loss,
    # the update_ops are done and the gradient updates are computed.
    train_tensor = slim.learning.create_train_op(total_loss, optimizer)

    # Actually runs training.
    slim.learning.train(train_tensor, log_dir)

    ###########################
    # Kicks off the training. #
    ###########################
    # with tf.Session() as sess:
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #
    #     # import ipdb; ipdb.set_trace()
    #     for _ in range(5):
    #         print(labels.eval())
    #
    #     coord.request_stop()
    #     coord.join(threads)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('dataset_dir')
    parser.add_argument('--log_dir', default='log/train')
    parser.add_argument('--batch_size', default=32)

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.DEBUG)
    main(**vars(args))
