# Copyright 2017 Tensorflow. All Rights Reserved.
# Modifications copyright 2018 UT Austin/Taewan Kim
# We follow the object detection API of Tensorflow
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
from skvideo.io import (vreader,FFmpegWriter)

import _init_paths
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MIN_SCORE = .5

tf.app.flags.DEFINE_string(
    'data_pre', 'coco',
    'Type of dataset for pretrained model ex) coco, kitti')

tf.app.flags.DEFINE_string(
    'model',
    'pretrained/faster_rcnn_nas_coco_2017_11_08/frozen_inference_graph.pb',
    'path to the frozenm model graph file for object detection')

tf.app.flags.DEFINE_string(
    'label', 
    '../models/research/object_detection/data/mscoco_label_map.pbtxt',
    'file path for the labels')

tf.app.flags.DEFINE_integer(
    'num_class', 9,
    'Number of Classes to consider from 1 in the label map')

tf.app.flags.DEFINE_string(
    'output', None,
    'path to save the ground truth dataset')

tf.app.flags.DEFINE_string(
    'input', 'data',
    'path to the directory containing data folders. '
    'Each data folder has cam.mp4')

tf.app.flags.DEFINE_boolean(
    'is_rotate', False,
    'Whether to rotate 180 degree or not (For Accord, True)')

FLAGS = tf.app.flags.FLAGS

_FILE_VIDEO = 'cam.mp4'

def get_valid_label_list(data_pre):
    if data_pre == 'coco':
        return [u'person',u'car',u'bus',u'truck']
    elif data_pre == 'kitti':
        return [u'car',u'pedestrian']
    else:
        return [u'car',u'pedestrian']

def convert_label(data_pre,label):
    if data_pre == 'coco':
        if label == u'person':
            return u'pedestrian'
        elif label in [u'car',u'truck',u'bus']:
            return u'car'
    else:
        return label

class Detector:
    def __init__(self,file_model_pb):
        self.det_graph = tf.Graph()
        self.file_model_pb = file_model_pb
        self.load_model()

    def load_model(self):
        # Preload frozen Tensorflow Model
        with self.det_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.file_model_pb,'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')
        
            # Definite input and output Tensors for detection_graph
            self.image_tensor = self.det_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image 
            # where a particular object was detected.
            self.det_boxes = self.det_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.det_scores = self.det_graph.get_tensor_by_name('detection_scores:0')
            self.det_classes = self.det_graph.get_tensor_by_name('detection_classes:0')
            self.num_det = self.det_graph.get_tensor_by_name('num_detections:0')

    def load_sess(self):
        self.sess = tf.Session(graph=self.det_graph)

    def close_sess(self):
        self.sess.close()

    def detect(self,image):
        image_np_expanded = np.expand_dims(image, axis=0)
        boxes,scores,classes,num = self.sess.run(
                        [self.det_boxes,self.det_scores,self.det_classes,self.num_det],
                        feed_dict={self.image_tensor:image_np_expanded})
        return boxes,scores,classes,num

def gen_video(list_dpath,category_index=None,list_valid_ids=None,
              detector=None,is_rotate=False):

    detector.load_sess() # Load tf.Session
    for d_path in list_dpath:
        vwriter = FFmpegWriter(os.path.join(d_path,'cam_labeled.mp4'))

        # ------------------------------ IMAGE ----------------------------------
        # Read video file and do object detection for generating images per frame
        print('...Processing: {}'.format(d_path))
        input_video = os.path.join(d_path,_FILE_VIDEO)
        
        videogen = vreader(input_video)

        for i_frame,image in enumerate(videogen):
            # Rotate image
            if is_rotate:
                image = np.rot90(image,k=2,axes=(0,1))
            # ----- Process object detection -----
            (boxes, scores, classes, num) = detector.detect(image)
            classes = np.squeeze(classes).astype(np.int32)
            # Select only valid classes
            idx_consider = [cid in list_valid_ids for cid in classes]
            classes = classes[idx_consider]
            boxes = np.squeeze(boxes)[idx_consider,:]
            scores = np.squeeze(scores)[idx_consider]
            
            # Save video with bounding boxes
            image_labeled = np.copy(image)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_labeled,
                boxes,
                classes,
                scores,
                category_index,
                max_boxes_to_draw=None,
                min_score_thresh=MIN_SCORE,
                use_normalized_coordinates=True,
                line_thickness=2)
            vwriter.writeFrame(image_labeled)
        vwriter.close()
    detector.close_sess() # Close tf.Session

def main(_):
    if tf.__version__ < '1.4.0':
        raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

    assert os.path.exists(FLAGS.input),\
        "Directory {} doesn't exist! (Data folder)".format(FLAGS.input)

    # Define Paths for data generation
    out_path = FLAGS.output if FLAGS.output else FLAGS.input

    list_dpath = [os.path.join(FLAGS.input,subdir) \
                  for subdir in os.listdir(FLAGS.input)]

    # Load label map (pretrained model)
    label_map = label_map_util.load_labelmap(FLAGS.label)
    categories = label_map_util.convert_label_map_to_categories(
                    label_map,
                    max_num_classes=FLAGS.num_class,
                    use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Get list of valid label ids (pretrained model's label -> CARSTOP label)
    list_labels = get_valid_label_list(FLAGS.data_pre)
    list_valid_ids = [cid for cid in category_index.keys() \
                      if category_index[cid]['name'] in list_labels]
    for cid in list_valid_ids:
        category_index[cid]['name'] = convert_label(FLAGS.data_pre,
                                                    category_index[cid]['name'])

    # Load object detection model
    obj_detector = Detector(FLAGS.model)

    # Generate detection video
    gen_video(list_dpath,
              category_index=category_index,
              list_valid_ids=list_valid_ids,
              detector=obj_detector,
              is_rotate=FLAGS.is_rotate)

if __name__ == '__main__':
    tf.app.run()
