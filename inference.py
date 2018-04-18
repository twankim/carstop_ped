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
from object_detection.utils import dataset_util

from object_detection.utils import metrics
from object_detection.utils import object_detection_evaluation as obj_eval
from object_detection.core import standard_fields

tf.app.flags.DEFINE_string('data_dir', '', 'Location of root directory for the '
                           'ground truth data. Folder structure is assumed to be:'
                           '<gt_dir>/cstopp_train.tfrecord,'
                           '<gt_dir>/cstopp_test.tfrecord'
                           '<gt_dir>/cstopp_val.tfrecord')
tf.app.flags.DEFINE_string('model_dir', '', 'Location of root directory for the '
                           'inference data. Folder structure is assumed to be:'
                           '<det_dir>/cstopp_train.tfrecord,'
                           '<det_dir>/cstopp_test.tfrecord'
                           '<det_dir>/cstopp_val.tfrecord')
tf.app.flags.DEFINE_string('output_dir', '', 'Path to which metrics'
                           'will be written.')

tf.app.flags.DEFINE_string(
    'data_pre', 'coco',
    'Type of dataset for pretrained model ex) coco, kitti')

tf.app.flags.DEFINE_string(
    'label', 
    '../models/research/object_detection/data/mscoco_label_map.pbtxt',
    'file path for the labels')

tf.app.flags.DEFINE_integer(
    'num_class', 9,
    'Number of Classes to consider from 1 in the label map')

FLAGS = tf.app.flags.FLAGS

data_feature = {
	'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
	'image/format': tf.FixedLenFeature([], tf.string),
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
		#self.sess.run(tf.local_variables_initializer())
		#self.sess.run(tf.global_variables_initializer())

	def get_image(self):
		#print(dir(self.sess))
		with self.read_graph.as_default():
			old_graph_def = tf.GraphDef()
			_, serialized_example = self.reader.read(self.filename_queue)
			features = tf.parse_single_example(serialized_example, features=data_feature)
			image_decoded = tf.image.decode_png(features['image/encoded'])
			filename = features['image/filename']
			width = features['image/width']
			height = features['image/height']
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
			tf.import_graph_def(old_graph_def, name='')
		print('Coord done')
		#print(self.sess)
		image_eval, width_eval, height_eval, f_eval = np.array(self.sess.run([image_decoded, width, height, filename]))
		print('file ' + str(f_eval) + ' is of shape ' + str( image_eval.shape))
		#print(max(image_eval))
		#print(min(image_eval))
		#image_eval = image_eval.reshape(width_eval, height_eval)
		#self.close_sess()
		return image_eval, f_eval

class Detector:
    def __init__(self, file_model_pb):
        self.det_graph = tf.Graph()
        self.file_model_pb = file_model_pb
        self.load_model()
        self.load_sess()

    def load_model(self):
    	print("Loading model...")
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

    def detect(self, image):
	self.load_sess()
        image_np_expanded = np.expand_dims(image, axis=0)
        boxes,scores,classes,num = self.sess.run(
        		[self.det_boxes,self.det_scores,self.det_classes,self.num_det],
        		feed_dict={self.image_tensor:image_np_expanded})
        self.close_sess()
        return boxes,scores,classes,num

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

def prepare_example(filename, bbox, scores, classes):
	ymin = []
	xmin = []
	ymax = []
	xmax = []
	sc = []
	lab =  []
	for i in range(0, scores.shape[0]):
		if scores[i] >= 0.5:
			ymin.append(bbox[i][0])
			xmin.append(bbox[i][1])
			ymax.append(bbox[i][2])
			xmax.append(bbox[i][3])	
			sc.append(scores[i])
			lab.append(classes[i].encode('utf8'))
	
	print('scores ' + str(sc))
	print('labels ' + str(lab))

	example = tf.train.Example(features=tf.train.Features(feature={		'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
	'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
	'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
	'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
	'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
	'image/object/class': dataset_util.bytes_list_feature(lab),
	'image/object/scores': dataset_util.float_list_feature(sc)
	}))

	return example


def inference(data_dir=FLAGS.data_dir, model_dir=FLAGS.model_dir, output_dir=FLAGS.output_dir, split='train'):

	# Define output
	output_path = os.path.join(output_dir,'cstopp_{}.tfrecord'.format(split))
	tf_writer = tf.python_io.TFRecordWriter(output_path)

	# Create detector class
	detector = Detector(model_dir)
	inference_reader = Reader(data_dir)
	print(inference_reader.num_records)

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

	for i in range(0, inference_reader.num_records):
		print(i)
		image, filename = inference_reader.get_image()
		#print('Image type is ' + str(image))
		bbox, scores, classes, num = detector.detect(image)

		#print('bbox is ' + str(bbox))
		#print('score is ' + str(scores))
		#print('classes is ' + str(classes))

		classes = np.squeeze(classes).astype(np.int32)
		# Select only valid classes
		idx_consider = [cid in list_valid_ids for cid in classes]
		classes = classes[idx_consider]
		boxes = np.squeeze(bbox)[idx_consider,:]
		scores = np.squeeze(scores)[idx_consider]
		
		classes = [category_index[classes[i]]['name'] for i in range(0, boxes.shape[0])]
		#print('c is ' + str(c))
      		example = prepare_example(filename, boxes, scores, classes)
		print('record is ' + str(example))
      		tf_writer.write(example.SerializeToString())

	tf_writer.close()




if __name__ == '__main__':
  inference(
      data_dir=FLAGS.data_dir,
      model_dir=FLAGS.model_dir,
      output_dir=FLAGS.output_dir)	
