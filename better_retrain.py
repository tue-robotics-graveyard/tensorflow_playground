#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
from datetime import datetime
import glob
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat


def create_inception_graph(model_dir):
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    Returns:
        Graph holding the trained Inception network, and various tensors we'll be
        manipulating.
    """
    BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
    JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
    RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

    with tf.Session() as sess:
        model_filename = os.path.join(model_dir, 'classify_image_graph_def.pb')
        with open(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def main(image_dir):
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = \
        create_inception_graph('model')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('image_dir', help='Image folder')

    args = parser.parse_args()

    main(**vars(args))