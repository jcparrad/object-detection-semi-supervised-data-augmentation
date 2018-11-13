import inference_utils as in_utl 
import os 

PATH_TO_FROZEN_GRAPH = "/home/detection/flores/models/exported/model_ssd_mobilenet_v1_fpn_4/frozen_model/frozen_inference_graph.pb"
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join("/home/detection/flores/models/exported/model_ssd_mobilenet_v1_fpn_4/flowers_label_map_ssd_mobilenet_v1_fpn_4.pbtxt")
NUM_CLASSES = 13

detector = in_utl.inferencer_on_image(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS, NUM_CLASSES)

# only detect elements in the image
#image_path = "/home/detection/source/inference/test_images/frame_281.jpg"
#output_dict, image_np = detector.detect(image_path)
#print (output_dict['num_detections'])
#print (output_dict['detection_classes'])
#print (output_dict['detection_boxes'])
#print (output_dict['detection_scores'])
#print (output_dict['detection_boxes'])

# detect and save the detected elements inside the image
#image_path = "/home/detection/source/inference/test_images/frame_281.jpg"
#saving_path = "/home/detection/source/inference/detected_images"
#saved_name_image = "frame_281_detected.jpg"
#xml_path = "/home/detection/source/inference/test_images/Annotations"
#detector.save_detected_image(image_path, saving_path, saved_name_image)
#detector.get_detection_boxes(image_path)
#detector.generate_xml_annotation(image_path, xml_path)

xml_path = "/home/detection/source/inference/test_images/Annotations"
image_directory = "/home/detection/source/inference/test_images"
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".JPG"): 
    	image_path = os.path.join(image_directory, filename)
    	print(image_path)
    	detector.generate_xml_annotation(image_path, xml_path)
    
    
# for all images in a folder, detect and save the detected elements inside the image

# saving_path = "/home/detection/source/inference/detected_images"
# directory = "/home/detection/source/data/inference/test_images/DJI_3468"
# for filename in os.listdir(directory):
#     if filename.endswith(".jpg") or filename.endswith(".JPG"): 
#         image_path =  os.path.join(directory, filename)
#         print (image_path)
#         saved_name_image = filename.split(".")[0] + "_detected" + ".jpg"
#         detector.save_detected_image(image_path, saving_path, saved_name_image)