#!/bin/bash
#

DATA_PRE=coco
FILE_MODEL=pretrained/faster_rcnn_nas_coco_2017_11_08/frozen_inference_graph.pb
LABEL_MAP=../models/research/object_detection/data/mscoco_label_map.pbtxt
NUM_CLASS=9
INPUT=/data/cstopp
FILE_SPLIT=/data/cstopp/data_split.txt
FPS_IN=10
FPS_OUT=10

python ped_detect.py \
        --data_pre=${DATA_PRE} \
        --model=${FILE_MODEL} \
        --label=${LABEL_MAP} \
        --num_classes=${NUM_CLASS} \
        --input=${INPUT} \
        --f_split=${FILE_SPLIT} \
        --fps_in=${FPS_IN} \
        --fps_out=${FPS_OUT}
