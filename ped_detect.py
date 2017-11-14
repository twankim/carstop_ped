# Copyright 2017 UT Austin/ Michal Motro. All Rights Reserved.
# Modifications copyright 2017 UT Austin/Taewan Kim
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

import numpy as numpy
import os
import sys
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'model', 'faster_rcnn_nas_coco_2017_11_08',
    'model name to use for object detection')

# tf.app.flags.DEFINE_integer(
#     'is_rand', True, 'Turn on random decalibration')

# tf.app.flags.DEFINE_boolean(
#     'is_crop', True, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

def main(_):
    if tf.__version__ != '1.4.0':
        raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')
    
    # if tf.gfile.IsDirectory(FLAGS.model):
    model_name = os.path.join('pretrained',FLAGS.model,'frozen_inference_graph.pb')


if __name__ == '__main__':
    tf.app.run()