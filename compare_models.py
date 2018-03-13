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
from past.builtins import xrange

import _init_paths
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import metrics
from object_detection.utils import object_detection_evaluation as obj_eval
from object_detection.core import standard_fields

import os
import sys
import numpy as np
import tensorflow as tf
from skvideo.io import (vreader,vwrite)
from skimage.io import imsave, imread
from matplotlib import pyplot as plt
import collections
import json
import time
import pandas as pd

from importLidarSimple import read as lread

MIN_SCORE = .5

tf.app.flags.DEFINE_boolean(
    'is_compare', False,
    'If is_compare is true, all models specified in compare.txt (in CURR) will be evaluated')

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
    'Each data folder has cam.mkv,lidar.dat, and timestamps.txt'
    'This path also includes a txt file specifying split (train/val/test)')

tf.app.flags.DEFINE_string(
    'f_split', 'None',
    'path to the txt file specifying the data split.')

tf.app.flags.DEFINE_integer(
    'fps_in', 10,
    'frame rate of the video (original)')

tf.app.flags.DEFINE_integer(
    'fps_out', 10,
    'frame rate of the video (output)')

tf.app.flags.DEFINE_boolean(
    'is_vout', True, 'Generate a video with bounding boxes')

tf.app.flags.DEFINE_boolean(
    'is_lidar', False, 'Look for Lidar if set to true')

FLAGS = tf.app.flags.FLAGS

_FILE_VIDEO = 'cam.mkv'
_FILE_LIDAR = 'lidar.dat'
_FILE_TIMES = 'timestamps.txt'
_FILE_OUT = '{:06d}' # Format of output files

def get_split_dict(f_split, in_path):
    dict_split = {}
    split_list = ['train', 'val', 'test']
    split_bool = [False, False, False]
    with open(f_split, 'r') as f_s:
        for line in f_s:
            tmp_list = line.split('\n')[0].split(' ')
            split = tmp_list[0]
            assert split in split_list, \
                'Error in {} file! {} split is not supported'.format(f_split, split)
            assert not split_bool[split_list.index(split)], \
                'Duplicate in {} file! ({}) already visited.'.format(f_split, split)
            dict_split[split] = list(set(tmp_list[1:]))
            # Combine with the input path
            dict_split[split] = [os.path.join(in_path, dpath) \
                                 for dpath in dict_split[split]]
            split_bool[split_list.index(split)] = True
    return dict_split

def get_valid_label_list(data_pre):
    if data_pre == 'coco':
        return [u'person', u'car', u'bus', u'truck']
    elif data_pre == 'kitti':
        return [u'car', u'pedestrian']
    else:
        return [u'car', u'pedestrian']

def convert_label(data_pre, label):
    if data_pre == 'coco':
        if label == u'person':
            return u'pedestrian'
        elif label in [u'car', u'truck', u'bus']:
            return u'car'
    else:
        return label

# Convert hh:mm:ss format string to seconds
def tstamp2sec(tstamp):
    hh, mm, ss = tstamp.split(':')
    return 3600*int(hh) + 60*int(mm) + int(ss)

# Convert seconds to hh:mm:ss format string
def sec2tstamp(tsec):
    hh = int(tsec / 3600)
    mm = int((tsec % 3600) /60)
    sec = int(tsec % 60)
    return '{:02d}:{:02d}:{:02d}'.format(hh, mm, sec)

def convert_timestamps(timeline):
    t_start, t_end = timeline.split('\n')[0].split(' ')
    sec_duration = tstamp2sec(t_end) - tstamp2sec(t_start)
    return t_start, sec2tstamp(sec_duration), sec_duration

class Detector:
    def __init__(self, file_model_pb):
        self.det_graph = tf.Graph()
        self.file_model_pb = file_model_pb
        self.load_model()

    def load_model(self):
        # Preload frozen Tensorflow Model
        with self.det_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.file_model_pb, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def,  name='')
        
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

    def detect(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        boxes, scores, classes, num = self.sess.run(
                        [self.det_boxes, self.det_scores, self.det_classes, self.num_det], 
                        feed_dict={self.image_tensor:image_np_expanded})
        return boxes, scores, classes, num 

def get_ground_truth(gt_file, cat_idx_map):
    f = open(gt_file)
    lines = f.read()
    l_list = lines.split('\n')
    l_list = [i for i in l_list if i]
    bbox = []
    classes = []
    for idx, l in enumerate(l_list):
        l_split = l.split()
        classes.append(cat_idx_map[l_split[0].strip()])
        bb = l_split[1:-1]
        bb = [float(i) for i in bb]
        bbox.append(bb)
    return np.array(bbox, dtype='float32'), np.array(classes, dtype='int32')

def gen_data(split, list_dpath, out_path, fps_in, fps_out, 
             category_index=None, list_valid_ids=None, is_vout=False, detector=None):
    # Create directories to save image,  lidar,  and label
    print("List dpath ", list_dpath)
    opath_image = []
    opath_lidar = []
    opath_label = []
    opath_metrics = []
    for model_out in out_path:
        print("Model out ", model_out)
        opath_image.append(os.path.join(model_out, 'image'))
        opath_lidar.append(os.path.join(model_out, 'lidar'))
        opath_label.append(os.path.join(model_out, 'label'))
        opath_metrics.append(os.path.join(model_out, 'metrics'))
        if not os.path.exists(opath_image[-1]):
            os.makedirs(opath_image[-1])
        if not os.path.exists(opath_lidar[-1]):
            os.makedirs(opath_lidar[-1])
        if not os.path.exists(opath_label[-1]):
            os.makedirs(opath_label[-1])

    # Input images will be stored here
    ipath_image = list(map(lambda p: os.path.join(p, 'image'), list_dpath))
    for p in ipath_image:
        if not os.path.exists(p):
            os.makedirs(p)

    print("Input path is ", ipath_image)
    i_save = 0 # Frame name indexing
    sum_frames = 0 # Current total number of frames
    r_fps = fps_in/fps_out # ratio of input/output fps
    assert (fps_in % fps_out) == 0, \
            "Input FPS {} must be divisible by the Target FPS".format(fps_in)
    if is_vout:
        video_out = [] # Output video with labels

    for det in detector:
        det.load_sess()

    # print("Obj Detectors loaded")
    # print('\n<Generating {} set>'.format(split))

    # print('\n<Creating evaluator>')

    cat_uidx = 1
    cat_idx_map = {}
    idx_idx_map = {}
    for k,v in category_index.items():
        if k in list_valid_ids:
            if v['name'] not in cat_idx_map.keys():
                cat_idx_map[v['name']] = cat_uidx
                idx_idx_map[v['id']] = cat_uidx
                cat_uidx += 1
            else:
                idx_idx_map[v['id']] = cat_idx_map[v['name']]
    cat_out = [{'id':v, 'name':k.strip()} for k,v in cat_idx_map.items()]
    evaluator = obj_eval.ObjectDetectionEvaluator(cat_out)
    ground_truth_path = os.path.join(FLAGS.output, 'train/label')

    eval_out = collections.defaultdict(list)
    eval_summarize_out = {}

    for ipath_idx, d_path in enumerate(list_dpath):
        i_save_lidar = i_save # Frame name indexing for LIDAR
        n_frames = [] # Number of frames to be saved

        # Load time stamps
        input_time = os.path.join(d_path, _FILE_TIMES)
        time_stamps = []
        with open(input_time, 'r') as f_time:
            time_stamps.extend(list(map(lambda x: convert_timestamps(x), f_time)))

        # ------------------------------ IMAGE ----------------------------------
        # Read video file and do object detection for generating images per frame
        print('...({})Generating images per frame: {}'.format(split, d_path))
        input_video = os.path.join(d_path, _FILE_VIDEO)
        
        # Save frames only from the selected time frames
        for time_stamp in time_stamps:
            videogen = vreader(input_video, 
                               num_frames=int(time_stamp[2]*fps_in), 
                               inputdict={'-r':str(fps_in)}, 
                               outputdict={'-r':str(fps_in), 
                                           '-ss':time_stamp[0], 
                                           '-t':time_stamp[1]})
            print('   Image&Label Start: {},  Duration: {} (secs)'.format(
                            time_stamp[0], 
                            time_stamp[2]))

            for i_frame, image in enumerate(videogen):
                if i_frame % r_fps == 0:
                    # Save image frame
                    in_image = os.path.join(ipath_image[ipath_idx], _FILE_OUT.format(i_save)+'.png')
                    print("Saving input image at ", in_image)
                    imsave(in_image, image)

                    # ----- Process object detection ----
                    for idx, det in enumerate(detector):
                        print("Detecting ", opath_image[idx])
                        out_image = os.path.join(opath_image[idx], _FILE_OUT.format(i_save)+'.png')
                        start = time.time()
                        (boxes,  scores,  classes,  num) = det.detect(image)
                        end = time.time()
                        classes = np.squeeze(classes).astype(np.int32)
                        # Select only valid classes
                        idx_consider = [cid in list_valid_ids for cid in classes]
                        classes = classes[idx_consider]
                        boxes = np.squeeze(boxes)[idx_consider, :]
                        scores = np.squeeze(scores)[idx_consider]

                        # Evaluation
                        dt_classes = np.array([idx_idx_map[i] for i in classes])
                        ground_truth_label = os.path.join(ground_truth_path, _FILE_OUT.format(i_save)+'.txt')
                        gt_bbox, gt_classes = get_ground_truth(ground_truth_label, cat_idx_map)
                        ground_dict = {standard_fields.InputDataFields.groundtruth_boxes: gt_bbox, standard_fields.InputDataFields.groundtruth_classes: gt_classes}
                        det_dict = {standard_fields.DetectionResultFields.detection_boxes: boxes[:len(gt_classes)], standard_fields.DetectionResultFields.detection_scores: scores[:len(gt_classes)], standard_fields.DetectionResultFields.detection_classes: dt_classes[:len(gt_classes)]}
                        eval_label = os.path.join(opath_image[idx], _FILE_OUT.format(i_save))
                        evaluator.add_single_ground_truth_image_info(eval_label, ground_dict)
                        evaluator.add_single_detected_image_info(eval_label, det_dict)
                        curr_out = evaluator.evaluate()
                        curr_out['Speed'] = end - start
                        eval_out[opath_metrics[idx]].append(curr_out)
                        for k,v in curr_out.items():
                            if opath_metrics[idx] in eval_summarize_out.keys():
                                if k in eval_summarize_out[opath_metrics[idx]].keys():
                                    eval_summarize_out[opath_metrics[idx]][k] += v 
                                else:
                                    eval_summarize_out[opath_metrics[idx]][k] = v 
                            else:
                                eval_summarize_out[opath_metrics[idx]] = {}
                                eval_summarize_out[opath_metrics[idx]][k] = v 
                        # print(eval_summarize_out)

                        # Save bounding boxes with score
                        out_label = os.path.join(opath_label[idx], _FILE_OUT.format(i_save)+'.txt')
                        with open(out_label, 'w') as f_label:
                            # Format splitted by space.
                            # 1: class
                            # 4: bbox (ymin,  xmin,  ymax,  xmax) (normalized 0~1)
                            # 1: score
                            for i_obj in xrange(boxes.shape[0]):
                                if scores[i_obj]>MIN_SCORE:
                                    line = [category_index[classes[i_obj]]['name']]
                                    line += [str(coord) for coord in boxes[i_obj]]
                                    line += [str(scores[i_obj])]
                                    line = ' '.join(line)+'\n'
                                    f_label.write(line)

                        # Save video with bounding boxes
                        if is_vout:
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
                            print("Saving output image at ", out_image)
                            imsave(out_image, image_labeled)
                            #video_out.append(image_labeled)
                i_save +=1
                n_frames.append(i_save-sum_frames)
                sum_frames = i_save

        # Save output video with bounding boxes
        if is_vout:
            for opath in opath_image:
                video_out = []
                for image_num in xrange(i_save):
                    p = os.path.join(opath, _FILE_OUT.format(image_num)+'.png')
                    video_out.append(imread(p))
                vwrite(os.path.join(opath, split+'_labeled.mp4'), 
                   np.array(video_out), 
                   inputdict={'-r':str(fps_out)}, 
                   outputdict={'-r':str(fps_out)})

        for k,v in eval_out.items():
            print('Writing metrics for ' + k)
            f = open(k,'w')
            f.write(json.dumps(v))
            f.close()

        for k,v in eval_summarize_out.items():
            print('Writing summary for ' + k + '_summary')
            vavg = {i:(j/i_frame) for i,j in v.items()}
            f = open(k + '_summary', 'w')
            f.write(json.dumps(vavg))
            f.close()

        print('Writing cumulative summary')  
        summary = []
        columns = []
        for k,v in eval_summarize_out.items():
            summary.append([])
            summary[-1].append(k)
            summary[-1].extend([(j/i_frame) for _,j in v.items()])
        v = eval_summarize_out[eval_summarize_out.keys()[0]]
        columns = [i for i,_ in v.items()]
        columns.insert(0, 'model')
        print(summary)
        print(columns)

        df = pd.DataFrame.from_records(summary, columns=columns)
        df.to_csv(os.path.join(FLAGS.output,'summary'))

    for det in detector:
        det.close_sess()

def get_model(model_name):
    if model_name == None:
        print("Model name is missing. Cannot load that which doesn't exist")
        return None
    # print("Model Name")
    # print(model_name)
    model_path = os.path.join('../models/research/object_detection',  model_name,  'frozen_inference_graph.pb')
    # print("Model Path")
    # print(model_path)
    if os.path.isfile(model_path):
        print("Found ",  model_path)
        return Detector(model_path)
    else:
        print("Frozen model for %s does not exist. Going to download",  model_name)
        MODEL_FILE = model_name + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
        try:
            opener = urllib.request.URLopener()
            opener.retrieve(DOWNLOAD_BASE + MODEL_FILE,  MODEL_FILE)
            tar_file = tarfile.open(MODEL_FILE)
            for file in tar_file.getmembers():
                file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file,  os.getcwd())
            return Detector(model_path)
        except:
            print("Download failed. Model name %s incorrect",  model_name)
            return None

def main(_):
    if tf.__version__ < '1.4.0':
        raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

    assert os.path.exists(FLAGS.input), \
        "Directory {} doesn't exist! (Data folder)".format(FLAGS.input)

    assert os.path.exists(FLAGS.f_split), \
        "File {} doesn't exist! (txt file for data split)".format(FLAGS.f_split)

    # Define Paths for data generation
    out_path = FLAGS.output if FLAGS.output else FLAGS.input

    # Read split file
    dict_split = get_split_dict(FLAGS.f_split, FLAGS.input)
    print(dict_split)

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

    print(list_valid_ids)
    print(category_index)
    # return

    # Load object detection model
    if FLAGS.is_compare:
        out_parent = out_path
        out_path = []
        obj_detector = []
        with open('compare.txt', 'r') as f:
            for model in f:
                m = get_model(model.strip())
                if m:
                    obj_detector.append(m)
                    out_path.append(os.path.join(out_parent,  model.strip()))
    else:
        obj_detector = Detector(FLAGS.model)

    print(len(obj_detector))

    # Generate Data per split
    for split in dict_split.keys():
        base_dpath = []
        for out in out_path:
            base_dpath.append(os.path.join(out, split))
            if not os.path.exists(base_dpath[-1]):
                os.makedirs(base_dpath[-1])

        # Get list of paths to be used in generating specific split
        list_dpath = dict_split[split]
        print(list_dpath)
        gen_data(split, list_dpath, base_dpath, FLAGS.fps_in, FLAGS.fps_out, 
                 category_index=category_index, 
                 list_valid_ids=list_valid_ids, 
                 is_vout=FLAGS.is_vout, 
                 detector=obj_detector)
    
if __name__ == '__main__':
    tf.app.run()
