#!/bin/bash
#

DATA_PRE=coco
FILE_MODEL=pretrained/faster_rcnn_nas_coco_2017_11_08/frozen_inference_graph.pb
LABEL_MAP=../models/research/object_detection/data/mscoco_label_map.pbtxt
NUM_CLASS=9
INPUT=/data/cstopp/accord_040118
OUTPUT=/data/cstopp
FILE_SPLIT=/data/cstopp/data_split.txt
FPS_OUT=10

python cstopp_gen_data.py \
        --data_pre=${DATA_PRE} \
        --model=${FILE_MODEL} \
        --label=${LABEL_MAP} \
        --num_classes=${NUM_CLASS} \
        --input=${INPUT} \
        --output=${OUTPUT} \
        --f_split=${FILE_SPLIT} \
        --fps_out=${FPS_OUT}
