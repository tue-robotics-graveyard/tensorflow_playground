#!/usr/bin/env python
from argparse import ArgumentParser
import tensorflow as tf
import os
import glob
import re
import hashlib
from tensorflow.python.util import compat


MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


validation_percentage = 0.1


def inputs(image_dir, validation=False):
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
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))

            if (percentage_hash < validation_percentage) == validation:
                images.append(file_name)
                labels.append(label_name)

    class_count = len(set(labels))
    if class_count == 0:
        exit('No valid folders of images found at ' + image_dir)
    if class_count == 1:
        exit('Only one valid folder of images found at ' + image_dir +
             ' - multiple classes are needed for classification.')

    return images, labels


def main(image_dir):
    input_images, input_labels = inputs(image_dir)

    filename_queue = tf.train.string_input_producer(input_images)
    label_queue = tf.train.input_producer(input_labels)

    reader = tf.WholeFileReader()
    _, whole_file = reader.read(filename_queue)

    image_batch = tf.train.shuffle_batch([whole_file, label_queue],
        batch_size=32,
        capacity=50000,
        min_after_dequeue=10000)

    # image_batch, label_batch = tf.train.shuffle_batch([whole_file])
        # batch_size=32,
        # num_threads=4,
        # capacity=50000,
        # min_after_dequeue=10000)

    import ipdb; ipdb.set_trace()


    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1):  # length of your filename list
            print key.eval()
            print len(value.eval())

        # print(len(image))
        # Image.show(Image.fromarray(np.asarray(image)))

        coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('image_dir', help='Folder with input files')

    args = parser.parse_args()
    exit(main(**vars(args)))