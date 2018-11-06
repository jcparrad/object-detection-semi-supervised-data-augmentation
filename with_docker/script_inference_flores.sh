#! /bin/bash

echo "the script starts now"

#cp -r /home/flores/flores /home/obj_api_2
#cp -r /home/flores/with_docker /home/obj_api_2

workon cpu-py3
workon gpu-py3

cd /home/obj_api_2/models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py

# copy custom py script: create_tf_record.py
#cd /home/obj_api_2/models/research/object_detection/dataset_tools
#cp /home/obj_api_2/flores/custom_py_scripts/create_tf_record.py .

#python /home/detection/flores/custom_py_scripts/rename_jpg_files.py
#python /home/detection/flores/custom_py_scripts/correct_xml.py  /home/detection/flores/data/raw_data/

# vreation of tf record files
python /home/obj_api_2/models/research/object_detection/dataset_tools/create_tf_record.py \
    --data_dir=/home/detection/flores/data/raw_data \
    --output_path=/home/detection/flores/data/output_train_tf_flowers.record \
    --label_map_path=/home/detection/flores/data/flowers_label_map_ssd_mobilenet_v1_fpn_4.pbtxt \
    --output_val_path=/home/detection/flores/data/output_val_tf_flowers.record

# running the training job
#${PIPELINE_CONFIG_PATH} points to the pipeline config
PIPELINE_CONFIG_PATH=/home/detection/flores/models/tl_models/model_ssd_mobilenet_v1_fpn/pipeline_model_ssd_mobilenet_v1_fpn_4.config
MODEL_DIR=/home/detection/flores/models/train_eval/model_ssd_mobilenet_v1_fpn_4
NUM_TRAIN_STEPS=50000
NUM_EVAL_STEPS=2000
python /home/obj_api_2/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr 
#    2>&1 | tee  /home/detection/flores/models/train_eval/model_ssd_mobilenet_v1_fpn_4/output_log.txt


#${MODEL_DIR} points to the directory in which training checkpoints and events will be written to. Note that this binary will interleave both training and evaluation.
MODEL_DIR=/home/detection/flores/models/train_eval/model_ssd_mobilenet_v1_fpn_4
tensorboard --logdir=${MODEL_DIR}

######### exporting trained model for inference ########

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/home/detection/flores/models/tl_models/model_ssd_mobilenet_v1_fpn/pipeline_model_ssd_mobilenet_v1_fpn.config
TRAINED_CKPT_PREFIX=/home/detection/flores/models/train_eval/model_ssd_mobilenet_v1_fpn_2/model.ckpt-33214 #{path to model.ckpt}
EXPORT_DIR=/home/detection/flores/models/exported/model_ssd_mobilenet_v1_fpn_2
python /home/obj_api_2/models/research/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}

