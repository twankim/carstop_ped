#!/bin/bash
#

DATA_PRE=coco
FILE_MODEL=pretrained/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb
LABEL_MAP=../models/research/object_detection/data/mscoco_label_map.pbtxt
NUM_CLASS=9
FPS_IN=30
FPS_OUT=30
MODEL_NAME=ssd_inception_v2

python cstopp_gen_video.py \
        --data_pre=${DATA_PRE} \
        --model=${FILE_MODEL} \
        --label=${LABEL_MAP} \
        --num_classes=${NUM_CLASS} \
        --input=/data/cstopp/accord_040118 \
        --fps_in=${FPS_IN} \
        --fps_out=${FPS_OUT} \
        --is_rotate=True \
        --model_name=${MODEL_NAME}
