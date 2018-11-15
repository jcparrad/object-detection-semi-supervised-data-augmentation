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
		self.threshold = 0.6
		self.labels =   {
						  1: "P1",
						  2: "P1_1",
						  3: "P1_2",
						  4: "P2",
						  5: "P2_1",
						  6: "P3",
						  7: "P3_1",
						  8: "P4",
						  9: "P5",
						  10: "P5_1",
						  11: "P6",
						  12: "P7",
						  13: "P8"
						}

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

	def detect(self, image_path):

		image = Image.open(image_path)
		# the array based representation of the image will be used later in order to prepare the
		# result image with boxes and labels on it.
		image_np = self.load_image_into_numpy_array(image)
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		# Actual detection.
		output_dict = self.run_inference_for_single_image(image_np, self.detection_graph)
		return output_dict, image_np

	def save_detected_image(self, image_path, saving_path, saved_name_image):
		output_dict, image_np = self.detect(image_path)
		# for saving the images in a folder
		# # TODO: save images in a folder whose name is based on the model and date of the data
		
        
		vis_util.visualize_boxes_and_labels_on_image_array(
			image_np,
			output_dict['detection_boxes'],
			output_dict['detection_classes'],
			output_dict['detection_scores'],
			self.category_index,
			instance_masks=output_dict.get('detection_masks'),
			use_normalized_coordinates=True,
			line_thickness=8)
		
		im = Image.fromarray(image_np)
		# 	# relative paths
		#dir = os.path.dirname(os.path.realpath('__file__'))
		#directory = os.path.join(dir, 'out')


		if not os.path.exists(saving_path):
			os.makedirs(saving_path)

		# name = "out_" + str(i) + ".jpg"
		# file_path = os.path.join(directory, name)
		file_path = os.path.join(saving_path, saved_name_image)
		im.save(file_path)


		boxes = output_dict['detection_boxes']
		print ("boxes = output_dict[detection_boxes]")
		im_width, im_height = im.size
		for box in boxes:
			if box.any() != 0: #[0, 0, 0, 0]:
				ymin, xmin, ymax, xmax = box
				(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
				print (left, right, top, bottom)
	

	def get_detection_boxes(self, image_path):
		# this function detects the elements in the image 
		# and return the boxes coordinates (left, right, top, bottom)
		# and also the image

		output_dict, image_np = self.detect(image_path)
		im = Image.fromarray(image_np)

		boxes = output_dict['detection_boxes']
		classes = output_dict['detection_classes']
		scores = output_dict['detection_scores']
		im_width, im_height = im.size

		boxes_data = []
		for i in range (0, len(boxes)):
			if boxes[i].any() != 0:
				ymin, xmin, ymax, xmax = boxes[i]
				(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
				boxes_data.append([classes[i], scores[i], [int(left), int(right), int(top), int(bottom)]])

		#for i in range(0, len(boxes_data)):
		#	print (boxes_data[i])

		return boxes_data, im
		
	def set_threshold(self, threshold):
		# this functions sets the htreshold that will allow a detected box be available
		# to be annotated in the xml VOC file
		self.threshold = threshold

	def generate_xml_annotation(self, image_path, xml_path):
		# this function generates the xml file
		# their inputs are:
		# 1. image_path: path where the image file is located
		# 2. xml_path: path folder where to save the generated xml file
		# It returns:
		# 1. boxes_data: all the boxes that where detected in the image
		# 2. cont_low_score: the number of boxes that had a score less than the threshold
		# therefore these boxes are not available for the xml annotation file

		boxes_data, im = self.get_detection_boxes(image_path)
		annotation = ET.Element("annotation")

		ET.SubElement(annotation, "folder").text = "Images"
		filename = image_path.split("/")[-1]
		ET.SubElement(annotation, "filename").text = filename #"DJI_0210.JPG"
		ET.SubElement(annotation, "path").text ="/home/path"

		source = ET.SubElement(annotation, "source")
		ET.SubElement(source, "database").text ="Unknown"

		im_width, im_height = im.size
		#mode_to_bpp = {'1':1, 'L':8, 'P':8, 'RGB':24, 'RGBA':32, 'CMYK':32, 'YCbCr':24, 'I':32, 'F':32}
		#bpp = mode_to_bpp[im.mode]
		#print ("im_width", im_width)
		#print ("im_height", im_height)
		#print ("data.mode", im.mode)
		#print ("bpp", bpp)

		size = ET.SubElement(annotation, "size")
		ET.SubElement(size, "width").text =str(im_width)
		ET.SubElement(size, "height").text =str(im_height)
		ET.SubElement(size, "depth").text =str(3)

		ET.SubElement(annotation, "segmented").text ="0"

		# This element is repeated for each label that the model generates
		cont_low_score = 0
		elements = [] # intentar con append 
		for i in range (0, len(boxes_data)):
			if self.threshold <= boxes_data[i][1]:
				print (boxes_data[i])
				elements.append(ET.SubElement(annotation, "object"))
				object_ = elements[i]
				label = self.labels[boxes_data[i][0]]
				ET.SubElement(object_, "name").text = label#"var_name_" + str(i)
				ET.SubElement(object_, "pose").text ="Unspecified"
				ET.SubElement(object_, "truncated").text ="0"
				ET.SubElement(object_, "difficult").text ="0"
				bndbox = ET.SubElement(object_, "bndbox")
				# boxes are as follows:
				#(left, right, top, bottom) = 
				#	(xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
				box = boxes_data[i][2]
				ET.SubElement(bndbox, "xmin").text = str(box[0])
				ET.SubElement(bndbox, "ymin").text = str(box[2])
				ET.SubElement(bndbox, "xmax").text = str(box[1])
				ET.SubElement(bndbox, "ymax").text = str(box[3])
			else:
				cont_low_score += 1


		tree = ET.ElementTree(annotation)

		
		if not os.path.exists(xml_path):
			os.makedirs(xml_path)

		name_xml = image_path.split("/")[-1].split(".")[0] + ".xml"		
		xml_path = os.path.join(xml_path,name_xml)
		tree.write(xml_path)

		return boxes_data, cont_low_score