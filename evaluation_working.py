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
# from skvideo.io import (vreader,FFmpegWriter)
# from skimage.io import imsave
# from matplotlib import pyplot as plt

import _init_paths
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from object_detection.utils import metrics
from object_detection.utils import object_detection_evaluation as obj_eval
from object_detection.core import standard_fields


tf.app.flags.DEFINE_string('data_dir', '', 'Location of root directory for the '
                           'data. Folder structure is assumed to be:'
                           '<data_dir>/cstopp_train.tfrecord,'
                           '<data_dir>/cstopp_test.tfrecord'
                           '<data_dir>/cstopp_val.tfrecord')
tf.app.flags.DEFINE_string('output_dir', '', 'Path to which metrics'
                           'will be written.')

FLAGS = tf.app.flags.FLAGS

def read_from_tf(data_dir=FLAGS.data_dir, output_dir=FLAGS.output_dir):
	data_path = []
	data_path.append(os.path.join(data_dir, 'cstopp_train.tfrecord'))
	category = [{'id': 1, 'name':'pedestrian'}, {'id':2, 'name':'car'}]
	evaluator = obj_eval.ObjectDetectionEvaluator(category)
#	data_path.append(os.path.join(data_dir, 'cstopp_test.tfrecord'))
#	data_path.append(os.path.join(data_dir, 'cstopp_val.tfrecord'))
	print(data_path)

        #exit()
	with tf.Session() as sess:

		sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())
		feature = {
		  'image/height': tf.FixedLenFeature([], tf.int64),
		  'image/width': tf.FixedLenFeature([], tf.int64),
		  'image/filename': tf.FixedLenFeature([], tf.string),
		  'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
		  'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
		  'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
		  'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
		  'image/object/class/label': tf.VarLenFeature(tf.int64),
		  # 'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
		  # 'lidar/xyz': dataset_util.float_list_feature(lidar_xyz),
		}
		# Get number of records to process
		num_records = sum(1 for _ in tf.python_io.tf_record_iterator(data_path[0]))
		# Create a list of filenames and pass it to a queue
		filename_queue = tf.train.string_input_producer(data_path)
		# Define a reader and read the next record
		reader = tf.TFRecordReader()
		# Iterate over all the records
		for _ in range(0, num_records):
			k, serialized_example = reader.read(filename_queue)
			# # Decode the record read by the reader
			features = tf.parse_single_example(serialized_example, features=feature)
			# # Convert the image data from string back to the numbers
			height = features['image/height']
			xmin = features['image/object/bbox/xmin']
			xmax = features['image/object/bbox/xmax']
			ymin = features['image/object/bbox/ymin']
			ymax = features['image/object/bbox/ymax']
			label = features['image/object/class/label']
			filename = features['image/filename']
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			# # Cast label data into int32
			# label = tf.cast(features['train/label'], tf.int32)
			# # Reshape image data into the original shape
			# image = tf.reshape(image, [224, 224, 3])

			# # Any preprocessing here ...

			# # Creates batches by randomly shuffling tensors
			# images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

			xmin_eval, xmax_eval, ymin_eval, ymax_eval, label_eval, filename_eval = sess.run([xmin, xmax, ymin, ymax, label, filename])
			label_dense = tf.sparse_tensor_to_dense(label_eval).eval()
			b_shape = xmax_eval.dense_shape[0]
			
			print('filename ' + str(filename_eval))
			print('xmin ' + str(xmin_eval))
			print('xmax ' + str(xmax_eval))
			print('ymin ' + str(ymin_eval))
			print('ymax ' + str(ymax_eval))
			print('label ' + str(label_eval))
			print('label_dense ' + str(label_dense))

			bbox = []
			for i in range(0, b_shape):
				bbox.append([ymin_eval.values[xmin_eval.indices[i]][0], 
					xmin_eval.values[xmax_eval.indices[i]][0], ymax_eval.values[ymin_eval.indices[i]][0], 
					xmax_eval.values[ymax_eval.indices[i]][0]])
			print(bbox)
			bbox = np.array(bbox)	
			print('Done')
			
			gt_bbox = bbox
			gt_classes = label_dense

			dt_bbox = bbox
			dt_classes = label_dense
			dt_scores = np.array([0.9]*len(label_dense))
			ground_dict = {standard_fields.InputDataFields.groundtruth_boxes: gt_bbox, standard_fields.InputDataFields.groundtruth_classes: gt_classes}
			det_dict = {standard_fields.DetectionResultFields.detection_boxes: dt_bbox[:len(gt_classes)], standard_fields.DetectionResultFields.detection_scores: dt_scores[:len(gt_classes)], standard_fields.DetectionResultFields.detection_classes: dt_classes[:len(gt_classes)]}
			evaluator.add_single_ground_truth_image_info(filename_eval, ground_dict)
			evaluator.add_single_detected_image_info(filename_eval, det_dict)
			print("Evaluate is " +  str(evaluator.evaluate()))
			# curr_out['Speed'] = end - start
			# eval_out[opath_metrics[idx]].append(curr_out)			
# def get_ground_truth():
# 	bbox = []
# 	for i in range(0, xmax.shape()):
# 		bbox.append([xmin[i], xmax[i], ymin[i], ymax[i]])
# 	print(bbox)


# def get_ground_truth(gt_file, cat_idx_map):
#     f = open(gt_file)
#     lines = f.read()
#     l_list = lines.split('\n')
#     l_list = [i for i in l_list if i]
#     bbox = []
#     classes = []
#     for idx, l in enumerate(l_list):
#         l_split = l.split()
#         classes.append(cat_idx_map[l_split[0].strip()])
#         bb = l_split[1:-1]
#         bb = [float(i) for i in bb]
#         bbox.append(bb)
#     return np.array(bbox, dtype='float32'), np.array(classes, dtype='int32')

def evaluate():

	# Get predictions


	# Compare and print bbox
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
	# for k,v in curr_out.items():
	# if opath_metrics[idx] in eval_summarize_out.keys():
	# if k in eval_summarize_out[opath_metrics[idx]].keys():
	# eval_summarize_out[opath_metrics[idx]][k] += v 
	# else:
	# eval_summarize_out[opath_metrics[idx]][k] = v 
	# else:
	# eval_summarize_out[opath_metrics[idx]] = {}
	eval_summarize_out[opath_metrics[idx]][k] = v 
	# print(eval_summarize_out)

def write_record():
	pass

if __name__ == '__main__':
  read_from_tf(
      data_dir=FLAGS.data_dir,
      output_dir=FLAGS.output_dir)	
