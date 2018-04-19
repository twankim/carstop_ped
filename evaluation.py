# Copyright 2017 Tensorflow. All Rights Reserved.
# Modifications copyright 2018 UT Austin/Saharsh Oza
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

tf.app.flags.DEFINE_string('split', 'train', 'Data split when record file is being read from gt_dir and det_dir ex: train, test, val')

FLAGS = tf.app.flags.FLAGS

gt_feature = {
  'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
  'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
  'image/object/class/text': tf.VarLenFeature(tf.string),
  'image/filename': tf.FixedLenFeature([], tf.string),
  'image/object/difficult': tf.VarLenFeature(tf.int64),
}

det_feature = {
  'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
  'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
  'image/object/class/text': tf.VarLenFeature(tf.string),
  'image/object/scores': tf.VarLenFeature(tf.float32),	
  'image/filename': tf.FixedLenFeature([], tf.string),
}


class Reader:
	def __init__(self, record_path, split):
		data_path = []
		data_path.append(os.path.join(record_path, 'cstopp_{}.tfrecord'.format(split)))
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
		# Modify graph to add these ops
		with self.read_graph.as_default():
			old_graph_def = tf.GraphDef()

			# Read next record from queue
			_, serialized_example = self.reader.read(self.filename_queue)
			self.features = tf.parse_single_example(serialized_example, features=feature_dict)
			# Get required fields from record
			fields_out = [self.get_field(f) for f in feature_dict.keys()]
			# Close queue
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
			# Import updated graph in current read_graph
			tf.import_graph_def(old_graph_def, name='')
		eval_out = np.array(self.sess.run(fields_out))
		out_dict = dict(zip(feature_dict.keys(), eval_out))
		return out_dict

def get_bbox(box_list):
	ymin_eval = box_list['image/object/bbox/ymin']
	xmin_eval = box_list['image/object/bbox/xmin']
	ymax_eval = box_list['image/object/bbox/ymax']
	xmax_eval = box_list['image/object/bbox/xmax']
	b_shape = xmax_eval.dense_shape[0]
	bbox = []
	for i in range(0, b_shape):
		bbox.append([ymin_eval.values[xmin_eval.indices[i]][0], 
			xmin_eval.values[xmax_eval.indices[i]][0], ymax_eval.values[ymin_eval.indices[i]][0], 
			xmax_eval.values[ymax_eval.indices[i]][0]])
	bbox = np.array(bbox)
	return bbox	

def update_result(image_evaluation, image_num, csv_handle):
	metric_string = str(image_num) + "," + str(image_evaluation['PerformanceByCategory/AP@0.5IOU/pedestrian']) + "\n"
	csv_handle.write(metric_string)

def evaluate(gt_dir=FLAGS.gt_dir, det_dir=FLAGS.det_dir, output_dir=FLAGS.output_dir, split='train'):
	
	gt_reader = Reader(gt_dir, split)
	num_records = gt_reader.num_records
	det_reader = Reader(det_dir, split)

	category = [{'id': 1, 'name':'pedestrian'}]
	category_map = {'pedestrian': 1, 'car': 2}
	evaluator = obj_eval.ObjectDetectionEvaluator(category)
	
	output_path = os.path.join(output_dir, 'cstopp_{}_eval.csv'.format(split))
	csv_handle = open(output_path, 'w')

	final_eval = None
	for image_num in range(0, num_records):
		print('Evaluating ' + str(image_num) + " from " + str(num_records) + " images" )
		gt_fields = gt_reader.get_fields(gt_feature)
		gt_bbox = get_bbox(gt_fields)
		gt_classes = np.array([category_map[i] for i in gt_fields['image/object/class/text'].values])
		gt_diff = np.array(gt_fields['image/object/difficult'].values)

		det_fields = det_reader.get_fields(det_feature)
		det_bbox = get_bbox(det_fields)
		det_scores = det_fields['image/object/scores'].values
		det_classes = np.array([category_map[i] for i in det_fields['image/object/class/text'].values])
		filename = gt_fields['image/filename']

		ground_dict = {standard_fields.InputDataFields.groundtruth_boxes: gt_bbox, standard_fields.InputDataFields.groundtruth_classes: gt_classes, standard_fields.InputDataFields.groundtruth_difficult: gt_diff}
		det_dict = {standard_fields.DetectionResultFields.detection_boxes: det_bbox[:len(gt_classes)], standard_fields.DetectionResultFields.detection_scores: det_scores[:len(gt_classes)], standard_fields.DetectionResultFields.detection_classes: det_classes[:len(gt_classes)]}
		evaluator.add_single_ground_truth_image_info(filename, ground_dict)
		evaluator.add_single_detected_image_info(filename, det_dict)
		eval_result = evaluator.evaluate()
	#	print(eval_result)
		
		if final_eval is None:
			final_eval = {k:v for (k,v) in eval_result.iteritems()}
		else:
			final_eval = {k: (v+final_eval[k]) for (k,v) in eval_result.iteritems()}
		
		update_result(eval_result, image_num, csv_handle)	
	final_eval = {k:(v/num_records) for (k,v) in final_eval.iteritems()}
	csv_handle.close()


if __name__ == '__main__':
  evaluate(
      gt_dir=FLAGS.gt_dir,
      det_dir=FLAGS.det_dir,
      output_dir=FLAGS.output_dir,
      split=FLAGS.split)	
