from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

slim = tf.contrib.slim

FILE_PATTERN = 'flowers_%s_*.tfrecord'
SPLIT_NAMES = {'train', 'test'}
LABELS_FILENAME = 'labels.txt'

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    labels_filename = os.path.join(dataset_dir, filename)
    with open(labels_filename) as f:
        lines = f.read().split('\n')

    lines = filter(None, lines)
    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]
    return labels_to_class_names


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    if split_name not in SPLIT_NAMES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    # build the decoder
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=None,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=len(labels_to_names),
        labels_to_names=labels_to_names)
