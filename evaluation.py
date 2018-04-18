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
  'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
  'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
  'image/object/class': tf.VarLenFeature(tf.int64),
  'image/filename': tf.FixedLenFeature([], tf.string),
}

det_feature = {
  'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
  'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
  'image/object/class': tf.VarLenFeature(tf.int64),
  'image/object/scores': tf.VarLenFeature(tf.float32),	
  'image/filename': tf.FixedLenFeature([], tf.string),
}


class Reader:
	def __init__(self, record_path):
		data_path = []
		data_path.append(os.path.join(record_path, 'cstopp_train.tfrecord'))
		#data_path.append(os.path.join(record_path, 'cstopp_test.tfrecord'))
		#data_path.append(os.path.join(record_path, 'cstopp_val.tfrecord'))
		self.read_graph = tf.Graph()
		with self.read_graph.as_default():
			old_graph_def = tf.GraphDef()
			self.filename_queue = tf.train.string_input_producer(data_path)
			self.reader = tf.TFRecordReader()
			self.num_records = 0
			for f in data_path:
				self.num_records += sum(1 for _ in tf.python_io.tf_record_iterator(f))
			tf.import_graph_def(old_graph_def, name='')
		self.sess = tf.Session(graph=self.read_graph)

	def get_field(self, field, decode=False):
		if decode == False:
			return self.features[field]
		else:
			return tf.image.decode_png(self.features[field])

	def get_fields(self, feature_dict):
		# print(dir(self.sess))
		# Modify graph to add these ops
		with self.read_graph.as_default():
			old_graph_def = tf.GraphDef()

			# Read next record from queue
			_, serialized_example = self.reader.read(self.filename_queue)
			self.features = tf.parse_single_example(serialized_example, features=feature_dict)
			# Get required fields from record
			fields_out = [self.get_field(f) for f in feature_dict.keys()]
			#print(feature_dict.keys())
			# Close queue
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
			# Import updated graph in current read_graph
			tf.import_graph_def(old_graph_def, name='')
		#print('Coord done')
		print(self.sess)
		eval_out = np.array(self.sess.run(fields_out))
		out_dict = dict(zip(feature_dict.keys(), eval_out))
		#print(feature_dict.keys())
		return out_dict

def get_bbox(box_list):
	ymin_eval = box_list['image/object/bbox/ymin']
	xmin_eval = box_list['image/object/bbox/xmin']
	ymax_eval = box_list['image/object/bbox/ymax']
	xmax_eval = box_list['image/object/bbox/xmax']
	b_shape = xmax_eval.dense_shape[0]
	#print(b_shape)
	#print('ymin_eval is ' + str(ymin_eval))
	bbox = []
	for i in range(0, b_shape):
		bbox.append([ymin_eval.values[xmin_eval.indices[i]][0], 
			xmin_eval.values[xmax_eval.indices[i]][0], ymax_eval.values[ymin_eval.indices[i]][0], 
			xmax_eval.values[ymax_eval.indices[i]][0]])
	print(bbox)
	bbox = np.array(bbox)
	return bbox	
# def read_data(filename_queue, reader, sess, parse_type='gt'):
# 	_, serialized_example = reader.read(filename_queue)
# 	if 'gt' in parse_type:
# 		features = tf.parse_single_example(serialized_example, features=gt_feature)
# 	else:
# 		features = tf.parse_single_example(serialized_example, features=det_feature)
# 	xmin = features['image/object/bbox/xmin']
# 	xmax = features['image/object/bbox/xmax']
# 	ymin = features['image/object/bbox/ymin']
# 	ymax = features['image/object/bbox/ymax']
# 	label = features['image/object/class/label']
# 	filename = features['image/filename']
# 	coord = tf.train.Coordinator()
# 	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 	if 'gt' in parse_type:
# 		xmin_eval, xmax_eval, ymin_eval, ymax_eval, label_eval, filename_eval = sess.run([xmin, xmax, ymin, ymax, label, filename])
# 	else:
# 		scores = features['image/object/score']
# 		xmin_eval, xmax_eval, ymin_eval, ymax_eval, label_eval, filename_eval, scores_eval = sess.run([xmin, xmax, ymin, ymax, label, filename, scores])

# 	label_dense = tf.sparse_tensor_to_dense(label_eval).eval()
# 	b_shape = xmax_eval.dense_shape[0]
	
# 	print('filename ' + str(filename_eval))
# 	print('xmin ' + str(xmin_eval))
# 	print('xmax ' + str(xmax_eval))
# 	print('ymin ' + str(ymin_eval))
# 	print('ymax ' + str(ymax_eval))
# 	print('label ' + str(label_eval))
# 	print('label_dense ' + str(label_dense))

# 	bbox = []
# 	for i in range(0, b_shape):
# 		bbox.append([ymin_eval.values[xmin_eval.indices[i]][0], 
# 			xmin_eval.values[xmax_eval.indices[i]][0], ymax_eval.values[ymin_eval.indices[i]][0], 
# 			xmax_eval.values[ymax_eval.indices[i]][0]])
# 	print(bbox)
# 	bbox = np.array(bbox)

# 	if 'det' in parse_type:
# 		return bbox, label_dense, scores_eval, filename_eval
# 	else:
# 		return bbox, label_dense, filename_eval



def evaluate(gt_dir=FLAGS.gt_dir, det_dir=FLAGS.det_dir, output_dir=FLAGS.output_dir):

	gt_reader = Reader(gt_dir)
	num_records = gt_reader.num_records
	det_reader = Reader(det_dir)

	category = [{'id': 1, 'name':'pedestrian'}, {'id':2, 'name':'car'}]
	evaluator = obj_eval.ObjectDetectionEvaluator(category)

	for _ in range(0, num_records):
		gt_fields = gt_reader.get_fields(gt_feature)
		gt_bbox = get_bbox(gt_fields)
		gt_classes = np.array(gt_fields['image/object/class'].values)

		det_fields = det_reader.get_fields(det_feature)
		det_bbox = get_bbox(det_fields)
		det_classes = np.array(det_fields['image/object/class'].values)
		det_scores = det_fields['image/object/scores'].values
		filename = gt_fields['image/filename']
		#print('gt_bbox is ' + str(gt_bbox))
		print('gt_classes is ' + str(gt_classes))
		#print('det_bbox is ' + str(det_bbox))
		print('det_classes is ' + str(det_classes))
		#print('det_scores is ' + str(det_scores))
	
		#print('det_trunc is ' + str(det_bbox[:len(gt_classes)]))	
		ground_dict = {standard_fields.InputDataFields.groundtruth_boxes: gt_bbox, standard_fields.InputDataFields.groundtruth_classes: gt_classes}
		det_dict = {standard_fields.DetectionResultFields.detection_boxes: det_bbox[:len(gt_classes)], standard_fields.DetectionResultFields.detection_scores: det_scores[:len(gt_classes)], standard_fields.DetectionResultFields.detection_classes: det_classes[:len(gt_classes)]}
		evaluator.add_single_ground_truth_image_info(filename, ground_dict)
		evaluator.add_single_detected_image_info(filename, det_dict)
		print("Evaluate is " +  str(evaluator.evaluate()))


	# with tf.Session() as sess:
	# 	# Initialize variables
	# 	sess.run(tf.local_variables_initializer())
	# 	sess.run(tf.global_variables_initializer())
	# 	# Get number of records to process
	# 	num_records = sum(1 for _ in tf.python_io.tf_record_iterator(gt_data_path[0]))
	# 	det_num_records = sum(1 for _ in tf.python_io.tf_record_iterator(det_data_path[0]))
	# 	assert (num_records == det_num_records), "Unequal images in ground truth and detection"
	# 	# Create a list of filenames and pass it to a queue
	# 	gt_filename_queue = tf.train.string_input_producer(gt_data_path)
	# 	# Define a reader and read the next record
	# 	gt_reader = tf.TFRecordReader()
	# 	# Create a similar queue for detection 
	# 	det_filename_queue = tf.train.string_input_producer(det_data_path)
	# 	# Define a reader and read the next record
	# 	det_reader = tf.TFRecordReader()
	# 	# Iterate over all the records
		# for _ in range(0, num_records):
		# 	gt_bbox, gt_classes, filename_eval = read_data(gt_filename_queue, gt_reader, sess, 'gt')
		# 	dt_bbox, dt_classes, dt_scores, filename_eval = read_data(gt_filename_queue, gt_reader, sess, 'det')
		# 	print('gt_bbox is ' + str(gt_bbox))
		# 	print('gt_classes is ' + str(gt_classes))
		# 	print('det_bbox is ' + str(dt_bbox))
		# 	print('det_classes is ' + str(dt_classes))
		# 	ground_dict = {standard_fields.InputDataFields.groundtruth_boxes: gt_bbox, standard_fields.InputDataFields.groundtruth_classes: gt_classes}
		# 	det_dict = {standard_fields.DetectionResultFields.detection_boxes: dt_bbox[:len(gt_classes)], standard_fields.DetectionResultFields.detection_scores: dt_scores[:len(gt_classes)], standard_fields.DetectionResultFields.detection_classes: dt_classes[:len(gt_classes)]}
		# 	evaluator.add_single_ground_truth_image_info(filename_eval, ground_dict)
		# 	evaluator.add_single_detected_image_info(filename_eval, det_dict)
		# 	print("Evaluate is " +  str(evaluator.evaluate()))		


if __name__ == '__main__':
  evaluate(
      gt_dir=FLAGS.gt_dir,
      det_dir=FLAGS.det_dir,
      output_dir=FLAGS.output_dir)	
