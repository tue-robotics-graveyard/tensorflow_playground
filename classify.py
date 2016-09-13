#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser(description='Classify images with trained tf graph from file.')
parser.add_argument('model', help='Path of Tensorflow Graph model')
parser.add_argument('image', help='Path of image; should be 320x240')
parser.add_argument('labels', help='Path to generated Tensorflow class labels from the retrain.py')

args = parser.parse_args()

import os.path
import re
import sys
import tarfile
import argparse

import numpy as np
from six.moves import urllib
import tensorflow as tf

"""1. Create a graph from saved GraphDef file """
with open(args.model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


"""2. Open tf session"""
with tf.Session() as sess:

    """3. Get result tensor"""
    result_tensor = sess.graph.get_tensor_by_name("final_result:0")

    """4. Open Image and perform prediction"""
    predictions = []
    with open(args.image, 'rb') as f:
        predictions = sess.run(result_tensor, {'DecodeJpeg/contents:0': f.read()})
        predictions = np.squeeze(predictions)

    """5. Open output_labels and construct dict from result"""
    result = {}
    with open(args.labels, 'rb') as f:
        labels = f.read().split("\n")
        result = dict(zip(labels, predictions))
        
    """6. Print result"""
    print(result)