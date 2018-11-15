import inference_utils as in_utl 
import os 


def generate_xml_files():
	# Path to the frozen model
	PATH_TO_FROZEN_GRAPH = "/home/detection/flores/models/exported/model_ssd_mobilenet_v1_fpn_4/frozen_model/frozen_inference_graph.pb"
	
	# List of the strings that is used to add correct label for each box.
	PATH_TO_LABELS = os.path.join("/home/detection/flores/models/exported/model_ssd_mobilenet_v1_fpn_4/flowers_label_map_ssd_mobilenet_v1_fpn_4.pbtxt")
	NUM_CLASSES = 13

	# Initilize the detector
	detector = in_utl.inferencer_on_image(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS, NUM_CLASSES)

	# set the xml path folder where to save the generated xml files
	xml_path = "/home/detection/source/inference/test_images/Annotations"
	# set the folder that contains the images
	image_directory = "/home/detection/source/inference/test_images"
	# set the minimun threshold that the detected boxes must have for being able to be a label
	detector.set_threshold(0.55)
	for filename in os.listdir(image_directory):
	    if filename.endswith(".jpg") or filename.endswith(".JPG"): 
	    	image_path = os.path.join(image_directory, filename)
	    	print(image_path)
	    	# generate the xml annotation
	    	detector.generate_xml_annotation(image_path, xml_path)

def detect_elements_and_save_image():

	# Path to the frozen model
	PATH_TO_FROZEN_GRAPH = "/home/detection/flores/models/exported/model_ssd_mobilenet_v1_fpn_4/frozen_model/frozen_inference_graph.pb"
	
	# List of the strings that is used to add correct label for each box.
	PATH_TO_LABELS = os.path.join("/home/detection/flores/models/exported/model_ssd_mobilenet_v1_fpn_4/flowers_label_map_ssd_mobilenet_v1_fpn_4.pbtxt")
	NUM_CLASSES = 13

	# Initilize the detector
	detector = in_utl.inferencer_on_image(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS, NUM_CLASSES)
    
	#for all images in a folder, detect and save the detected elements inside the image
	saving_path = "/home/detection/source/inference/detected_images"
	directory = "/home/detection/source/data/inference/test_images/DJI_3468"
	for filename in os.listdir(directory):
	    if filename.endswith(".jpg") or filename.endswith(".JPG"): 
	        image_path =  os.path.join(directory, filename)
	        print (image_path)
	        saved_name_image = filename.split(".")[0] + "_detected" + ".jpg"
	        detector.save_detected_image(image_path, saving_path, saved_name_image)



#generate_xml_files()
detect_elements_and_save_image()

