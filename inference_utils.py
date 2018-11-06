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



class inferencer_on_image:
	def __init__ (self, PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS, NUM_CLASSES):
		
		#self.PATH_TO_LABELS = PATH_TO_LABELS
		#self.PATH_TO_FROZEN_GRAPH = PATH_TO_FROZEN_GRAPH
		#self.NUM_CLASSES = NUM_CLASSES

		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
		  od_graph_def = tf.GraphDef()
		  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
		    serialized_graph = fid.read()
		    od_graph_def.ParseFromString(serialized_graph)
		    tf.import_graph_def(od_graph_def, name='')

		# Loading label map
		self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
		self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
		self.category_index = label_map_util.create_category_index(self.categories)

	# Helper code
	def load_image_into_numpy_array(self, image):
	  (im_width, im_height) = image.size
	  return np.array(image.getdata()).reshape(
	      (im_height, im_width, 3)).astype(np.uint8)


	def run_inference_for_single_image(self, image, graph):
	  with graph.as_default():
	    with tf.Session() as sess:
	      # Get handles to input and output tensors
	      ops = tf.get_default_graph().get_operations()
	      #print (ops)
	      all_tensor_names = {output.name for op in ops for output in op.outputs}
	      all_tensor_names_input = {input.name for op in ops for input in op.inputs}
	      #print ("all_tensor_names_input")
	      #print (all_tensor_names_input)
	      #print (all_tensor_names)
	      tensor_dict = {}
	      for key in [
	          'num_detections', 'detection_boxes', 'detection_scores',
	          'detection_classes', 'detection_masks'
	      ]:
	        tensor_name = key + ':0'
	        #print (tensor_name)
	        if tensor_name in all_tensor_names:
	          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
	              tensor_name)
	    
	      
	      if 'detection_masks' in tensor_dict:
	        # The following processing is only for single image
	        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
	        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
	        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
	        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
	        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
	        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
	        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
	            detection_masks, detection_boxes, image.shape[0], image.shape[1])
	        detection_masks_reframed = tf.cast(
	            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
	        # Follow the convention by adding back the batch dimension
	        tensor_dict['detection_masks'] = tf.expand_dims(
	            detection_masks_reframed, 0)
	      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

	      # Run inference
	      #print (tensor_dict)
	    
	      output_dict = sess.run(tensor_dict,
	                             feed_dict={image_tensor: np.expand_dims(image, 0)})

	      # all outputs are float32 numpy arrays, so convert types as appropriate
	      #print ("output_dict")
	      #print (output_dict)
	      output_dict['num_detections'] = int(output_dict['num_detections'][0])
	      output_dict['detection_classes'] = output_dict[
	          'detection_classes'][0].astype(np.uint8)
	      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
	      output_dict['detection_scores'] = output_dict['detection_scores'][0]
	      if 'detection_masks' in output_dict:
	        output_dict['detection_masks'] = output_dict['detection_masks'][0]
	  return output_dict

	def detection(self, image_path):

		image = Image.open(image_path)
		# the array based representation of the image will be used later in order to prepare the
		# result image with boxes and labels on it.
		image_np = self.load_image_into_numpy_array(image)
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		# Actual detection.
		output_dict = self.run_inference_for_single_image(image_np, self.detection_graph)
		return output_dict







