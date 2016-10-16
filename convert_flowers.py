#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import glob
import logging
import os
import re
from argparse import ArgumentParser
from itertools import islice  # for Python 2.x
from sklearn.cross_validation import train_test_split

import tensorflow as tf
from tqdm import tqdm

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
LABELS_FILENAME = 'labels.txt'

bytes_feature = lambda v: tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
int64_feature = lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def mkdir_p(path):
    """mkdir -p

    :param path:
    :return: True if it had to create the directory
    """
    try:
        os.makedirs(path)
        return True
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return False
        else:
            raise


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def match_images(image_dir):
    if not os.path.isdir(image_dir):
        exit("Image directory '" + image_dir + "' not found.")

    images = []
    labels = []

    sub_dirs = [x[0] for x in os.walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue

        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            print('No files found')
            continue
        if len(file_list) < 20:
            print('WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print('WARNING: Folder {} has more than {} images. Some images will '
                  'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))

        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        for file_name in file_list:
            images.append(file_name)
            labels.append(label_name)

    class_count = len(set(labels))
    if class_count == 0:
        exit('No valid folders of images found at ' + image_dir)
    if class_count == 1:
        exit('Only one valid folder of images found at ' + image_dir +
             ' - multiple classes are needed for classification.')

    return images, labels


def get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
    output_filename = 'flowers_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)


def convert_dataset(split_name, filenames, class_ids, dataset_dir):
    assert split_name in ['train', 'validation']
    print('Converting %s dataset' % split_name)

    num_per_shard = 1000

    # Initializes function that decodes RGB JPEG data.
    decode_jpeg_data = tf.placeholder(dtype=tf.string)
    decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)

    with tf.Session() as sess:
        # process chunks of 1000
        for i, (filename, class_id) in enumerate(tqdm(zip(filenames, class_ids), desc='Images', unit='image')):
            shard_id = i // num_per_shard
            output_filename = get_dataset_filename(dataset_dir, split_name, shard_id, len(filenames) // num_per_shard)

            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                # Read the filename:
                image_data = open(filename, 'r').read()

                # decode JPEG
                image = sess.run(decode_jpeg, feed_dict={decode_jpeg_data: image_data})
                assert len(image.shape) == 3
                assert image.shape[2] == 3
                height, width = image.shape[0], image.shape[1]

                example = tf.train.Example(features=tf.train.Features(feature={
                    'image/encoded': bytes_feature(image_data),
                    'image/format': bytes_feature('jpg'),
                    'image/class/label': int64_feature(class_id),
                    'image/height': int64_feature(height),
                    'image/width': int64_feature(width),
                }))

                tfrecord_writer.write(example.SerializeToString())


def write_label_file(labels_to_class_names, dataset_dir, filename=LABELS_FILENAME):
    labels_filename = os.path.join(dataset_dir, filename)
    with open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def main(image_dir, example_dir):
    if not mkdir_p(example_dir):
        logger.warn('Output directory already existed. Be careful for duplicate files')

    images, labels = match_images(image_dir)
    class_names = sorted(set(labels))

    # convert labels to ids
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    class_ids = [class_names_to_ids[label] for label in labels]

    x_train, x_test, y_train, y_test = train_test_split(images, class_ids, test_size=0.1, random_state=42)

    # First, convert the training and validation sets.
    convert_dataset('train', x_train, y_train, example_dir)
    convert_dataset('validation', x_test, y_test, example_dir)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, example_dir)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)

    parser = ArgumentParser()

    parser.add_argument('image_dir', help='Folder with input files')
    parser.add_argument('example_dir', help='Folder with output files')

    args = parser.parse_args()
    exit(main(**vars(args)))
