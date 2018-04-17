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


tf.app.flags.DEFINE_string('gt_dir', '', 'Location of root directory for the '
                           'ground truth data. Folder structure is assumed to be:'
                           '<gt_dir>/cstopp_train.tfrecord,'
                           '<gt_dir>/cstopp_test.tfrecord'
                           '<gt_dir>/cstopp_val.tfrecord')
tf.app.flags.DEFINE_string('det_dir', '', 'Location of root directory for the '
                           'inference data. Folder structure is assumed to be:'
                           '<det_dir>/cstopp_train.tfrecord,'
                           '<det_dir>/cstopp_test.tfrecord'
                           '<det_dir>/cstopp_val.tfrecord')
tf.app.flags.DEFINE_string('output_dir', '', 'Path to which metrics'
                           'will be written.')

FLAGS = tf.app.flags.FLAGS

gt_feature = {
  'image/filename': tf.FixedLenFeature([], tf.string),
  'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
  'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
  'image/object/class/label': tf.VarLenFeature(tf.int64),
  # 'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
  # 'lidar/xyz': dataset_util.float_list_feature(lidar_xyz),
}

det_feature = {
  'image/filename': tf.FixedLenFeature([], tf.string),
  'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
  'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
  'image/object/class/label': tf.VarLenFeature(tf.int64),
  'image/object/score': tf.VarLenFeature(tf.float32),
  # 'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
  # 'lidar/xyz': dataset_util.float_list_feature(lidar_xyz),	
}

def parse(filename_queue, reader, sess, parse_type='gt'):
	_, serialized_example = reader.read(filename_queue)
	if 'gt' in parse_type:
		features = tf.parse_single_example(serialized_example, features=gt_feature)
	else:
		features = tf.parse_single_example(serialized_example, features=det_feature)
	xmin = features['image/object/bbox/xmin']
	xmax = features['image/object/bbox/xmax']
	ymin = features['image/object/bbox/ymin']
	ymax = features['image/object/bbox/ymax']
	label = features['image/object/class/label']
	filename = features['image/filename']
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	if 'gt' in parse_type:
		xmin_eval, xmax_eval, ymin_eval, ymax_eval, label_eval, filename_eval = sess.run([xmin, xmax, ymin, ymax, label, filename])
	else:
		scores = features['image/object/score']
		xmin_eval, xmax_eval, ymin_eval, ymax_eval, label_eval, filename_eval, scores_eval = sess.run([xmin, xmax, ymin, ymax, label, filename, scores])

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

	if 'det' in parse_type:
		return bbox, label_dense, scores_eval, filename_eval
	else:
		return bbox, label_dense, filename_eval



def evaluate(gt_dir=FLAGS.gt_dir, det_dir=FLAGS.det_dir, output_dir=FLAGS.output_dir):

	data_path = []
	data_path.append(os.path.join(gt_dir, 'cstopp_train.tfrecord'))
	category = [{'id': 1, 'name':'pedestrian'}, {'id':2, 'name':'car'}]
	evaluator = obj_eval.ObjectDetectionEvaluator(category)
#	data_path.append(os.path.join(gt_dir, 'cstopp_test.tfrecord'))
#	data_path.append(os.path.join(gt_dir, 'cstopp_val.tfrecord'))
	print(data_path)

	with tf.Session() as sess:
		# Initialize variables
		sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())

		# Get number of records to process
		num_records = sum(1 for _ in tf.python_io.tf_record_iterator(data_path[0]))
		# Create a list of filenames and pass it to a queue
		gt_filename_queue = tf.train.string_input_producer(data_path)
		# Define a reader and read the next record
		gt_reader = tf.TFRecordReader()
		# Iterate over all the records
		for _ in range(0, num_records):

			# k, serialized_example = reader.read(gt_filename_queue)
			# # # Decode the record read by the reader
			# features = tf.parse_single_example(serialized_example, features=feature)
			# # # Convert the image data from string back to the numbers
			# height = features['image/height']
			# xmin = features['image/object/bbox/xmin']
			# xmax = features['image/object/bbox/xmax']
			# ymin = features['image/object/bbox/ymin']
			# ymax = features['image/object/bbox/ymax']
			# label = features['image/object/class/label']
			# filename = features['image/filename']
			# coord = tf.train.Coordinator()
			# threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			# # Cast label data into int32
			# label = tf.cast(features['train/label'], tf.int32)
			# # Reshape image data into the original shape
			# image = tf.reshape(image, [224, 224, 3])

			# # Any preprocessing here ...

			# # Creates batches by randomly shuffling tensors
			# images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

			# xmin_eval, xmax_eval, ymin_eval, ymax_eval, label_eval, filename_eval = sess.run([xmin, xmax, ymin, ymax, label, filename])
			# label_dense = tf.sparse_tensor_to_dense(label_eval).eval()
			# b_shape = xmax_eval.dense_shape[0]
			
			# print('filename ' + str(filename_eval))
			# print('xmin ' + str(xmin_eval))
			# print('xmax ' + str(xmax_eval))
			# print('ymin ' + str(ymin_eval))
			# print('ymax ' + str(ymax_eval))
			# print('label ' + str(label_eval))
			# print('label_dense ' + str(label_dense))

			# bbox = []
			# for i in range(0, b_shape):
			# 	bbox.append([ymin_eval.values[xmin_eval.indices[i]][0], 
			# 		xmin_eval.values[xmax_eval.indices[i]][0], ymax_eval.values[ymin_eval.indices[i]][0], 
			# 		xmax_eval.values[ymax_eval.indices[i]][0]])
			# print(bbox)
			# bbox = np.array(bbox)	
			bbox, label_dense, filename_eval = parse(gt_filename_queue, gt_reader, sess, 'gt')
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


if __name__ == '__main__':
  evaluate(
      gt_dir=FLAGS.gt_dir,
      det_dir=FLAGS.det_dir,
      output_dir=FLAGS.output_dir)	
