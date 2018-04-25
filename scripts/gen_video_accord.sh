#!/bin/bash
#

DATA_PRE=coco
FILE_MODEL=pretrained/faster_rcnn_nas_coco_2017_11_08/frozen_inference_graph.pb
LABEL_MAP=../models/research/object_detection/data/mscoco_label_map.pbtxt
NUM_CLASS=9
FPS_IN=30
FPS_OUT=30
MODEL_NAME=faster_rcnn_nas

python cstopp_gen_video.py \
        --data_pre=${DATA_PRE} \
        --model=${FILE_MODEL} \
        --label=${LABEL_MAP} \
        --num_classes=${NUM_CLASS} \
        --input=/data/cstopp/video_accord \
        --fps_in=${FPS_IN} \
        --fps_out=${FPS_OUT} \
        --is_rotate=True \
        --model_name=${MODEL_NAME}
