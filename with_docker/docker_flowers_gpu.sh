#! /bin/bash

clear
echo "the script starts now"

#xhost +
export DISPLAY=127.0.0.1:0.0
#sudo docker run -it \
sudo nvidia-docker run -it \
  --privileged \
  --rm \
  --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro  \
  -e DISPLAY=$DISPLAY \
  --env="QT_X11_NO_MITSHM=1" \
  --privileged -v /dev/video0:/dev/video0 \
  --device /dev/video0 \
  -v /home/millenium/Juan_Camilo/flores/source/inference:/home/detection  \
  -p 8889:8888 -p 6009:6006 camilol/tensorflow:gpu-py3-object-detection /bin/bash

#  --net="host"\
#  -v /home/jcparrad/obj_api_flores:/home/obj_api_flores \
#  -v /home/jcparrad/obj_api_2:/home/obj_api_ \  
#-v /home/jcparrad/obj_api_flores:/home/obj_api_flores \
#/home/tesla/Juan_Camilo/obj_api_flores
#-v /home/tesla/Juan_Camilo/obj_api_pets/annotations:/home/obj_api_pets \
#-v /home/tesla/Juan_Camilo/obj_api_pets/images:/home/obj_api/models/research \
#tensorflow/tensorflow:latest-gpu-py3 /bin/bash
#sudo nvidia-docker run -it tensorflow/tensorflow:latest-gpu-py3 /bin/bash



