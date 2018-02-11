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
from skvideo.io import (vreader,vwrite)
from skimage.io import imsave
from matplotlib import pyplot as plt

import _init_paths
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from importLidarSimple import read as lread

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
    'dout', None,
    'path to save the ground truth dataset')

tf.app.flags.DEFINE_string(
    'input', 'data/sample0',
    'name of the the input video/image file at data folder')

tf.app.flags.DEFINE_integer(
    'fps_in', 10,
    'frame rate of the video (original)')

tf.app.flags.DEFINE_integer(
    'fps_out', 10,
    'frame rate of the video (output)')

tf.app.flags.DEFINE_boolean(
    'is_plot', False, 'Show and plot the result image')

tf.app.flags.DEFINE_boolean(
    'is_vout', True, 'Generate a video with bounding boxes')

FLAGS = tf.app.flags.FLAGS


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

# Convert hh:mm:ss format string to seconds
def tstamp2sec(tstamp):
    hh,mm,ss = tstamp.split(':')
    return 3600*int(hh) + 60*int(mm) + int(ss)

# Convert seconds to hh:mm:ss format string
def sec2tstamp(tsec):
    hh = int(tsec / 3600)
    mm = int((tsec % 3600) /60)
    sec = int(tsec % 60)
    return '{:02d}:{:02d}:{:02d}'.format(hh,mm,sec)

def convert_timestamps(timeline):
    t_start,t_end = timeline.split('\n')[0].split(' ')
    sec_duration = tstamp2sec(t_end) - tstamp2sec(t_start)
    return t_start,sec2tstamp(sec_duration)

def main(_):
    gen_image = False
    gen_lidar = False
    if tf.__version__ < '1.4.0':
        raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

    assert os.path.exists(FLAGS.input),\
        "File {} doesn't exist!".format(FLAGS.input)
    
    # Define PATHs for data generation
    fname = os.path.splitext(os.path.basename(FLAGS.input))[0]
    if FLAGS.dout:
        path_out = os.path.join(FLAGS.dout,fname)
    else:
        path_out = FLAGS.input
    if os.path.exists(os.path.join(FLAGS.input,'cam.mkv')):
        path_image = os.path.join(path_out,'image')
        gen_image = True
        if not os.path.exists(path_image):
            os.makedirs(path_image)
    if os.path.exists(os.path.join(FLAGS.input,'lidar.dat')):
        path_lidar = os.path.join(path_out,'lidar')
        gen_lidar = True
        if not os.path.exists(path_lidar):
            os.makedirs(path_lidar)
    path_label = os.path.join(path_out,'label')
    if not os.path.exists(path_label):
        os.makedirs(path_label)

    # Load time stamps
    # Text file 'timestamps.txt' is located with 'cam.mkv' and 'lidar.dat'
    path_time = os.path.join(FLAGS.input,'timestamps.txt')
    time_stamps = []
    with open(path_time,'r') as f_time:
        for line in f_time:
            time_stamps.append(convert_timestamps(line))

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

    # ------------------------------ IMAGE ----------------------------------
    # Read video file and do object detection for generating images per frame
    if gen_image:
        print('... Generating images per frame.')
        # Load a frozen Tensorflow model
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(FLAGS.model,'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Read video and do object detection
        input_video = os.path.join(FLAGS.input,'cam.mkv')
        i_save = 0 # image name indexing
        r_fps = FLAGS.fps_in/FLAGS.fps_out # ratio of input/output fps
        assert (FLAGS.fps_in % FLAGS.fps_out) == 0,\
            "{} must be divisible by the Target FPS".format(FLAGS.fps_in)
        if FLAGS.is_vout:
            video_out = [] # Output video with labels

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name(
                                    'image_tensor:0')
                # Each box represents a part of the image 
                # where a particular object was detected.
                det_boxes = detection_graph.get_tensor_by_name(
                                    'detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                det_scores = detection_graph.get_tensor_by_name(
                                    'detection_scores:0')
                det_classes = detection_graph.get_tensor_by_name(
                                    'detection_classes:0')
                num_det = detection_graph.get_tensor_by_name(
                                    'num_detections:0')
        
                # Save frames only from the selected time frames
                for i_test,time_stamp in enumerate(time_stamps):  
                    print('!!!!!{}'.format(i_test))              
                    videogen = vreader(input_video,
                                       inputdict={'-r':str(FLAGS.fps_in),
                                                  '-ss':time_stamp[0],
                                                  '-t':time_stamp[1]})
                    
                    i_frame = 0
                    for image in videogen:
                        if i_frame % r_fps == 0:
                            # Save image frame
                            print("-frame {}".format(i_save))
                            imsave(os.path.join(path_image,'{:06d}.png'.format(i_save)),image)

                            # ----- Process object detection -----
                            # the array based representation of the image 
                            # will be used later in order to prepare the result image 
                            # with boxes and labels on it.
                            
                            # Expand dimensions since the model expects images 
                            # to have shape: [1, None, None, 3]
                            image_np_expanded = np.expand_dims(image, axis=0)
                            
                            # Actual detection.
                            (boxes, scores, classes, num) = sess.run(
                                [det_boxes,det_scores,det_classes,num_det],
                                feed_dict={image_tensor: image_np_expanded})

                            classes = np.squeeze(classes).astype(np.int32)
                            # Select only valid classes
                            idx_consider = [cid in list_valid_ids for cid in classes]

                            classes = classes[idx_consider]
                            boxes = np.squeeze(boxes)[idx_consider,:]
                            scores = np.squeeze(scores)[idx_consider]
                            
                            # Save bounding boxes with score
                            with open(os.path.join(path_label,'{:06d}.txt'.format(i_save)),'w') as f_label:
                                # Format splitted by space.
                                # 1: class
                                # 4: bbox (ymin, xmin, ymax, xmax) (normalized 0~1)
                                # 1: score
                                for i_obj in xrange(boxes.shape[0]):
                                    if scores[i_obj]>MIN_SCORE:
                                        line = [category_index[classes[i_obj]]['name']]
                                        line += [str(coord) for coord in boxes[i_obj]]
                                        line += [str(scores[i_obj])]
                                        line = ' '.join(line)+'\n'
                                        f_label.write(line)

                            if FLAGS.is_vout:
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
                                    line_thickness=8)
                                video_out.append(image_labeled)
                            
                            if FLAGS.is_plot:
                                plt.figure(figsize=IMAGE_SIZE)
                                plt.imshow(image_np)
                            i_save +=1
                        i_frame += 1
        
        # Save output video with bounding boxes
        if FLAGS.is_vout:
            vwrite(os.path.join(path_out,fname+'_labeled.mp4'),
                   np.array(video_out),
                   outputdict={'-r':str(FLAGS.fps_out)})
    
    # ------------------------------ LIDAR ----------------------------------
    # Read LIDAR data and generate point cloud files per frame
    if gen_lidar:
        print('... Generating LIDAR point clouds per frame.')
if __name__ == '__main__':
    tf.app.run()
