import inference_utils as in_utl 
import os 

PATH_TO_FROZEN_GRAPH = "/home/detection/flores/models/exported/model_ssd_mobilenet_v1_fpn_4/frozen_model/frozen_inference_graph.pb"
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join("/home/detection/flores/models/exported/model_ssd_mobilenet_v1_fpn_4/flowers_label_map_ssd_mobilenet_v1_fpn_4.pbtxt")
NUM_CLASSES = 13

detector = in_utl.inferencer_on_image(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS, NUM_CLASSES)

image_path = "/home/detection/source/inference/test_images/DJI_3468/frame_1389.jpg"
output_dict = detector.detection(image_path)
print (output_dict['num_detections'])
print (output_dict['detection_classes'])
print (output_dict['detection_boxes'])

