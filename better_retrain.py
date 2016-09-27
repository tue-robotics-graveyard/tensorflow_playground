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


def main(image_dir):
    print(image_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('image_dir', help='Image folder')

    args = parser.parse_args()

    main(**vars(args))