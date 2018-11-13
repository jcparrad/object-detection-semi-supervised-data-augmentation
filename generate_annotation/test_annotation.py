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

annotation = ET.Element("annotation")
#doc = ET.SubElement(root, "doc")

ET.SubElement(annotation, "folder").text = "Images"
ET.SubElement(annotation, "filename").text = "DJI_0210.JPG"
ET.SubElement(annotation, "path").text ="/home/path"

source = ET.SubElement(annotation, "source")
ET.SubElement(source, "database").text ="Unknown"

size = ET.SubElement(annotation, "size")
ET.SubElement(size, "width").text ="TODO"
ET.SubElement(size, "height").text ="TODO"
ET.SubElement(size, "depth").text ="TODO"

ET.SubElement(annotation, "segmented").text ="0"

# This element is repeated for each label that the model generates
elements = [] # intentar con append 
for i in range (0, 5):
	elements.append(ET.SubElement(annotation, "object"))
	object_ = elements[i]
	ET.SubElement(object_, "name").text ="var_name_" + str(i)
	ET.SubElement(object_, "pose").text ="Unspecified"
	ET.SubElement(object_, "truncated").text ="0"
	ET.SubElement(object_, "difficult").text ="0"
	bndbox = ET.SubElement(object_, "bndbox")
	ET.SubElement(bndbox, "xmin").text ="var_xmin"
	ET.SubElement(bndbox, "ymin").text ="var_ymin"
	ET.SubElement(bndbox, "xmax").text ="var_xmax"
	ET.SubElement(bndbox, "ymax").text ="var_ymax"

tree = ET.ElementTree(annotation)
tree.write("filename.xml")