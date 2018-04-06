#!/bin/bash
#

DATA_PRE=coco
FILE_MODEL=pretrained/faster_rcnn_nas_coco_2017_11_08/frozen_inference_graph.pb
LABEL_MAP=../models/research/object_detection/data/mscoco_label_map.pbtxt
NUM_CLASS=9

python cstopp_gen_video.py \
        --data_pre=${DATA_PRE} \
        --model=${FILE_MODEL} \
        --label=${LABEL_MAP} \
        --num_classes=${NUM_CLASS} \
        --input=/data/cstopp/video_accord

python cstopp_gen_video.py \
        --data_pre=${DATA_PRE} \
        --model=${FILE_MODEL} \
        --label=${LABEL_MAP} \
        --num_classes=${NUM_CLASS} \
        --input=/data/cstopp/video_tower