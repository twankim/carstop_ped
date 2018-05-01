import numpy as np
import os
import _init_paths

import tensorflow as tf
from skimage.io import imread
from config.utils_data import *

def area(boxes):
  """Computes area of boxes.

  Args:
    boxes: Numpy array with shape [N, 4] holding N boxes

  Returns:
    a numpy array with shape [N*1] representing box areas
  """
  return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2):
  """Compute pairwise intersection areas between boxes.

  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes
    boxes2: a numpy array with shape [M, 4] holding M boxes

  Returns:
    a numpy array with shape [N*M] representing pairwise intersection area
  """
  [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
  [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

  all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
  all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
  intersect_heights = np.maximum(
      np.zeros(all_pairs_max_ymin.shape),
      all_pairs_min_ymax - all_pairs_max_ymin)
  all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
  all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
  intersect_widths = np.maximum(
      np.zeros(all_pairs_max_xmin.shape),
      all_pairs_min_xmax - all_pairs_max_xmin)
  return intersect_heights * intersect_widths


def iou(boxes1, boxes2):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes.
    boxes2: a numpy array with shape [M, 4] holding N boxes.

  Returns:
    a numpy array with shape [N, M] representing pairwise iou scores.
  """
  intersect = intersection(boxes1, boxes2)
  area1 = area(boxes1)
  area2 = area(boxes2)
  union = np.expand_dims(area1, axis=1) + np.expand_dims(
      area2, axis=0) - intersect
  return intersect / union

def read_annotation_file(filename):
  """Reads a CSTOPP annotation file.

  Converts a CSTOPP annotation file into a dictionary containing all the
  relevant information.
  Format splitted by space:
    1: class (string)
    4: bbox (ymin, xmin, ymax, xmax) (normalized 0~1)
    1: distance (m) (will be supported later...)
    1: score (Probability score 0~1) (Don't need it for ground truth)

  Args:
    filename: the path to the annotataion text file.

  Returns:
    anno: A dictionary with the converted annotation information.
  """
  #TODO Add depth Information in annotation
  with open(filename) as f:
    content = f.readlines()
  content = [x.strip().split(' ') for x in content]

  anno = {}
  anno['type'] = np.array([x[0].lower() for x in content])
  anno['2d_bbox_top'] = np.array([float(x[1]) for x in content])
  anno['2d_bbox_left'] = np.array([float(x[2]) for x in content])
  anno['2d_bbox_bottom'] = np.array([float(x[3]) for x in content])
  anno['2d_bbox_right'] = np.array([float(x[4]) for x in content])
  anno['score'] = np.array([float(x[5]) for x in content])
  anno['distance'] = np.array([float(x[6]) for x in content])

  return anno

diff_dict = {}
diff_dict['train'] = []
diff_dict['test'] = []

num_points = {}
num_points['train'] = []
num_points['test'] = []

data_dir = 'data'

model_name = 'channel_16/partial_ssd_inception_v2_carstop'
pred_dir = '../rtpdd/data/results/{}'.format(model_name)
dict_calib = loadCalib('config/velo/calib_intrinsic.txt',
                       'config/velo/calib_extrinsic.txt')

for split in ['train','test']:
  annotation_dir = os.path.join(data_dir,split,'label')
  pred_anno_dir = os.path.join(pred_dir,split,'label')
  im_dir = os.path.join(data_dir,split,'image')
  lidar_dir = os.path.join(data_dir,split,'lidar')

  names = sorted(tf.gfile.ListDirectory(annotation_dir))
  for name in names:
    print(name)
    name_base = name.split('.')[0]
    image = imread(os.path.join(im_dir,name_base+'.png'))
    points = np.fromfile(os.path.join(lidar_dir,name_base+'.bin')).reshape(-1,3)
    anno = read_annotation_file(os.path.join(annotation_dir,name))
    anno_pred = read_annotation_file(os.path.join(pred_anno_dir,name))
    
    gt_boxes = np.vstack([anno['2d_bbox_top'],
                          anno['2d_bbox_left'],
                          anno['2d_bbox_bottom'],
                          anno['2d_bbox_right']]).T
    det_boxes = np.vstack([anno_pred['2d_bbox_top'],
                           anno_pred['2d_bbox_left'],
                           anno_pred['2d_bbox_bottom'],
                           anno_pred['2d_bbox_right']]).T

    im_height,im_width = np.shape(image)[:2]
    points2D, pointsDist, pointsDistR = project_lidar_to_img(
                                                    dict_calib,
                                                    points,
                                                    im_height,
                                                    im_width)
    iou_mat = iou(det_boxes,gt_boxes)
    if len(gt_boxes)>0:
      max_overlap_gt_ids = np.argmax(iou_mat,axis=1)

      is_gt_box_detected = np.zeros(gt_boxes.shape[0],dtype=bool)

      for i in range(len(anno_pred['score'])):
        bbox = det_boxes[i,:]
        idx_in = (points2D[:,0]>=(bbox[0]*im_height)) & \
                 (points2D[:,0]<(bbox[2]*im_height)) & \
                 (points2D[:,1]>=(bbox[1]*im_width)) & \
                 (points2D[:,1]<(bbox[3]*im_width))
        gt_id = max_overlap_gt_ids[i]
        if iou_mat[i,gt_id] >= 0.5:
          if not is_gt_box_detected[gt_id]:
            is_gt_box_detected[gt_id] = True
            if anno['distance'][gt_id] >0:
              diff_dict[split].append(
                    abs(anno_pred['distance'][i]-anno['distance'][gt_id]))
              num_points[split].append(sum(idx_in))

diff_dict['train'] = np.array(diff_dict['train'])
diff_dict['test'] = np.array(diff_dict['test'])
num_points['train'] = np.array(num_points['train'])
num_points['test'] = np.array(num_points['test'])
print(
  model_name,
  len(diff_dict['train']),
  np.sqrt(np.sum(diff_dict['train']**2/float(len(diff_dict['train'])))))
train_zero = diff_dict['train'][num_points['train']==0]
print(len(train_zero),
  np.sqrt(np.sum(train_zero**2/float(len(train_zero)))))
print('----------------------------------')
print(model_name,
  len(diff_dict['test']), 
  np.sqrt(np.sum(diff_dict['test']**2/float(len(diff_dict['test'])))))
test_zero = diff_dict['test'][num_points['test']==0]
print(len(test_zero),
  np.sqrt(np.sum(test_zero**2/float(len(test_zero)))))
