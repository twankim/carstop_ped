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
from skimage.io import imsave
from matplotlib import pyplot as plt

import _init_paths
from object_detection.utils import label_map_util

from importVelo import loadKrotations
from config import visualization_utils as vis_util
from config.utils_data import *

MIN_SCORE = .70

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
    'Each data folder has cam.mkv(or cam.mp4), lidar.dat, and timestamps.txt'
    'This path also includes a txt file specifying split (train/val/test)')

tf.app.flags.DEFINE_string(
    'f_split', 'None',
    'path to the txt file specifying the data split.')

tf.app.flags.DEFINE_integer(
    'fps_out', 10,
    'frame rate of the video (output)')

tf.app.flags.DEFINE_boolean(
    'is_vout', True, 'Generate a video with bounding boxes')

tf.app.flags.DEFINE_boolean(
    'is_rotate', True, 'Whether to rotate 180 degree or not (For Accord, True)')

tf.app.flags.DEFINE_boolean(
    'gen_dist', False, 
    'Whether to generate distance value from LIDAR as ground truth or not.')

tf.app.flags.DEFINE_string('intrinsic_calib_path',
    'config/velo/calib_intrinsic.txt',
    'Path to a intrinsic calibration matrix config file.')
tf.app.flags.DEFINE_string('extrinsic_calib_path',
    'config/velo/calib_extrinsic.txt',
    'Path to a extrinsic calibration matrix config file.')

FLAGS = tf.app.flags.FLAGS

_FILE_TIMES = 'timestamps.txt'
_FILE_CONFIFGS = 'configs.txt'
_FILE_OUT = '{:06d}' # Format of output files

def get_split_dict(f_split,in_path):
    dict_split = {}
    split_list = ['train','val','test']
    split_bool = [False,False,False]
    with open(f_split,'r') as f_s:
        for line in f_s:
            tmp_list = line.split('\n')[0].split(' ')
            split = tmp_list[0]
            idx_split = split_list.index(split)
            assert split in split_list, \
                'Error in {} file! {} split is not supported'.format(
                    f_split,split)
            assert not split_bool[idx_split], \
                'Duplicate in {} file! ({}) already visited.'.format(
                    f_split,split)
            dict_split[split] = list(set(tmp_list[1:]))
            # Combine with the input path
            dict_split[split] = [os.path.join(in_path,dpath) \
                                 for dpath in dict_split[split]]
            split_bool[idx_split] = True
    return dict_split

def get_configs(f_configs):
    dict_cfg = {}
    cfg_list = ['f_video','f_lidar','fps_video','fps_lidar',
                'time_lidar','time_tower']
    cfg_type = [str,str,int,int,float,float]
    cfg_bool = [False] * len(cfg_list)

    # Read config file
    with open (f_configs,'r') as f_c:
        for line in f_c:
            tmp_list = line.split('\n')[0].split(' ')
            key_cfg = tmp_list[0]
            idx_cfg = cfg_list.index(key_cfg)
            assert key_cfg in cfg_list, \
                'Error in {} file! {} is not supported'.format(
                    f_configs,key_cfg)
            assert not cfg_bool[idx_cfg], \
                'Duplicate in {} file! ({}) already visited.'.format(
                    f_configs,key_cfg)
            dict_cfg[key_cfg] = cfg_type[idx_cfg](tmp_list[1])
            cfg_bool[idx_cfg] = True
    return dict_cfg

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
    return t_start,sec2tstamp(sec_duration),sec_duration

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

def gen_data(split,list_dpath,out_path,fps_out,
             category_index=None,list_valid_ids=None,is_vout=False,
             detector=None,is_rotate=False,dict_calib=None):
    # Create directories to save image, lidar, and label
    opath_image = os.path.join(out_path,'image')
    opath_lidar = os.path.join(out_path,'lidar')
    opath_label = os.path.join(out_path,'label')
    if not os.path.exists(opath_image):
        os.makedirs(opath_image)
    if not os.path.exists(opath_lidar):
        os.makedirs(opath_lidar)
    if not os.path.exists(opath_label):
        os.makedirs(opath_label)

    i_save = 0 # Frame name indexing
    # sum_frames = 0 # Current total number of frames
    if is_vout:
        vwriter = FFmpegWriter(os.path.join(out_path,split+'_labeled.mp4'),
                               inputdict={'-r':str(fps_out)},
                               outputdict={'-r':str(fps_out)})

    detector.load_sess() # Load tf.Session
    print('\n<Generating {} set>'.format(split))
    # Iterate over different tasks
    for d_path in list_dpath:
        # Load configurations for each task
        dict_cfg = get_configs(os.path.join(d_path,_FILE_CONFIFGS))
        _FILE_VIDEO = dict_cfg['f_video']
        _FILE_LIDAR = dict_cfg['f_lidar']
        fps_cam = dict_cfg['fps_video']
        fps_lidar = dict_cfg['fps_lidar']

        i_save_lidar = i_save # Frame name indexing for LIDAR

        r_fps_cam = fps_cam/fps_out # ratio of input/output fps
        assert (fps_cam % fps_out) == 0,\
            "Input FPS (Cam) {} must be divisible by the Target FPS {}".format(
                fps_cam,fps_out)
        r_fps_lidar = fps_lidar/fps_out # ratio of input/output fps
        assert (fps_lidar % fps_out) == 0,\
            "Input FPS (Lidar) {} must be divisible by the Target FPS {}".format(
                fps_lidar,fps_out)

        # Load time stamps
        input_time = os.path.join(d_path,_FILE_TIMES)
        with open(input_time,'r') as f_time:
            time_stamps = map(lambda x: convert_timestamps(x),f_time)

        input_lidar = os.path.join(d_path,_FILE_LIDAR)
        print(input_lidar)

        # ------------------------------ IMAGE ----------------------------------
        # Read video file and do object detection for generating images per frame
        print('...({})Generating images/points clouds per frame: {}'.format(
                        split,d_path))
        input_video = os.path.join(d_path,_FILE_VIDEO)
        
        # Save frames only from the selected time frames
        for time_stamp in time_stamps:
            # Number of frames to be saved
            n_frame = int(time_stamp[2]*fps_out)

            # Read lidar points
            print('   LIDAR Start: {}, Duration: {} (secs)'.format(
                            time_stamp[0],
                            time_stamp[2]))
            start_time = tstamp2sec(time_stamp[0])+dict_cfg['time_lidar']
            list_points = loadKrotations(input_lidar,start_time,
                                         n_frame,r_fps_lidar)
            for points in list_points:
                out_lidar = os.path.join(opath_lidar,
                                         _FILE_OUT.format(i_save_lidar)+'.bin')
                points.tofile(out_lidar)
                i_save_lidar += 1

            # Read Video file
            videogen = vreader(input_video,
                               num_frames=int(time_stamp[2]*fps_cam),
                               inputdict={'-r':str(fps_cam)},
                               outputdict={'-r':str(fps_cam),
                                           '-ss':time_stamp[0],
                                           '-t':time_stamp[1]})
            print('   Image&Label Start: {}, Duration: {} (secs)'.format(
                            time_stamp[0],
                            time_stamp[2]))
            for i_frame,image in enumerate(videogen):
                if is_rotate:
                    image = np.rot90(image,k=2,axes=(0,1))
                if i_frame % r_fps_cam == 0:
                    out_image = os.path.join(opath_image,
                                             _FILE_OUT.format(i_save)+'.png')
                    out_label = os.path.join(opath_label,
                                             _FILE_OUT.format(i_save)+'.txt')
                    # Save image frame
                    imsave(out_image,image)

                    # ----- Process object detection -----
                    (boxes, scores, classes, num) = detector.detect(image)
                    classes = np.squeeze(classes).astype(np.int32)
                    # Select only valid classes
                    idx_consider = [cid in list_valid_ids for cid in classes]
                    classes = classes[idx_consider]
                    boxes = np.squeeze(boxes)[idx_consider,:]
                    scores = np.squeeze(scores)[idx_consider]

                    if dict_calib:
                        # Load Corresponding point clouds
                        points = list_points[i_save-sum_frames]
                        dists = np.zeros(len(scores))
                        im_height,im_width = np.shape(image)[:2]
                        points2D, pointsDist, pointsDistR = project_lidar_to_img(
                                                    dict_calib,
                                                    points,
                                                    im_height,
                                                    im_width)
                    else:
                        dists = None
                    # Save Labels (including bounding boxes with score)
                    with open(out_label,'w') as f_label:
                        # Format splitted by space.
                        # 1: class
                        # 4: bbox (ymin, xmin, ymax, xmax) (normalized 0~1)
                        # 1: score
                        # 1: distance (if dict_calib is not None)
                        for i_obj in range(boxes.shape[0]):
                            if scores[i_obj]>MIN_SCORE:
                                line = [category_index[classes[i_obj]]['name']]
                                line += [str(coord) for coord in boxes[i_obj]]
                                line += [str(scores[i_obj])]
                                # Save distance value
                                if dict_calib:
                                    dists[i_obj] = dist_from_lidar_bbox(
                                                            points2D,
                                                            pointsDist,
                                                            pointsDistR,
                                                            boxes[i_obj],
                                                            im_height,
                                                            im_width)
                                    line += [str(dists[i_obj])]
                                f_label.write(' '.join(line)+'\n')

                    # Save video with bounding boxes
                    #TODO Make video with distance values
                    if is_vout:
                        image_labeled = np.copy(image)
                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_labeled,
                            boxes,
                            classes,
                            scores,
                            category_index,
                            dists=dists,
                            max_boxes_to_draw=None,
                            min_score_thresh=MIN_SCORE,
                            use_normalized_coordinates=True,
                            line_thickness=2)
                        vwriter.writeFrame(image_labeled)
                    i_save +=1

            # # Read lidar points
            # start_time = tstamp2sec(time_stamp[0])+dict_cfg['time_lidar']
            # list_points = loadKrotations(input_lidar,start_time,
            #                              i_save-sum_frames,r_fps_lidar)

            # for points in list_points:
            #     out_lidar = os.path.join(opath_lidar,
            #                              _FILE_OUT.format(i_save_lidar)+'.bin')
            #     points.tofile(out_lidar)
            #     i_save_lidar += 1

            # n_frames.append(i_save-sum_frames)
            sum_frames = i_save
    
    detector.close_sess() # Close tf.Session
    
    # Close video writer for video with bounding boxes
    if is_vout:
        vwriter.close()

def main(_):
    if tf.__version__ < '1.4.0':
        raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

    assert os.path.exists(FLAGS.input),\
        "Directory {} doesn't exist! (Data folder)".format(FLAGS.input)

    assert os.path.exists(FLAGS.f_split),\
        "File {} doesn't exist! (txt file for data split)".format(FLAGS.f_split)

    # Define Paths for data generation
    out_path = FLAGS.output if FLAGS.output else FLAGS.input

    # Read split file
    dict_split = get_split_dict(FLAGS.f_split,FLAGS.input)

    # Load calibration matrices for lidar & Camera
    if FLAGS.gen_dist:        
        dict_calib = loadCalib(FLAGS.intrinsic_calib_path,
                               FLAGS.extrinsic_calib_path)
    else:
        dict_calib = None

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

    # Generate Data per split
    for split in dict_split.keys():
        base_dpath = os.path.join(out_path,split)
        if not os.path.exists(base_dpath):
            os.makedirs(base_dpath)

        # Get list of paths to be used in generating specific split
        list_dpath = dict_split[split]
        gen_data(split,list_dpath,base_dpath,FLAGS.fps_out,
                 category_index=category_index,
                 list_valid_ids=list_valid_ids,
                 is_vout=FLAGS.is_vout,
                 detector=obj_detector,
                 is_rotate=FLAGS.is_rotate,
                 dict_calib=dict_calib)
    
if __name__ == '__main__':
    tf.app.run()
