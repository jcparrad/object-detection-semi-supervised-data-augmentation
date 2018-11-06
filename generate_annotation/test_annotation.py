import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO

# friendly backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image

sys.path.insert(0, "/home/obj_api_2/models/research/object_detection")
sys.path.append("/home/obj_api_2/models/research")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

import os
import shutil
import xml.etree.ElementTree as ET
import annotation_utilities as an_util 
