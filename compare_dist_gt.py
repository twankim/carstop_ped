import numpy as np
import os
import _init_paths

import tensorflow as tf

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

data_dir = 'data'
for split in ['train','test']:
  annotation_dir = os.path.join(data_dir,split,'label')
#  lidar_anno_dir = os.path.join(data_dir,split,'label_lidar')
  lidar_anno_dir = os.path.join(data_dir,split,'label_lidar_new')

  names = sorted(tf.gfile.ListDirectory(annotation_dir))
  for name in names:
    anno = read_annotation_file(os.path.join(annotation_dir,name))
    anno_lidar = read_annotation_file(os.path.join(lidar_anno_dir,name))
    
    for i in range(len(anno['score'])):
      if anno['distance'][i]>0:
        for j in range(len(anno_lidar['score'])):
          if (anno_lidar['type'][j]=='pedestrian') & \
             (abs(anno_lidar['2d_bbox_top'][j]-anno['2d_bbox_top'][i])<=1e-4) & \
             (abs(anno_lidar['2d_bbox_left'][j]-anno['2d_bbox_left'][i])<=1e-4) & \
             (abs(anno_lidar['2d_bbox_bottom'][j]-anno['2d_bbox_bottom'][i])<=1e-4) & \
             (abs(anno_lidar['2d_bbox_right'][j]-anno['2d_bbox_right'][i])<=1e-4):
            diff_dict[split].append(abs(anno_lidar['distance'][j]-anno['distance'][i]))
            if anno_lidar['distance'][j]==10.0:
              print(split, anno['distance'][i],abs(anno_lidar['distance'][j]-anno['distance'][i]))

print(len(diff_dict['train']), np.sqrt(np.sum(np.array(diff_dict['train'])**2/float(len(diff_dict['train'])))))
print(len(diff_dict['test']), np.sqrt(np.sum(np.array(diff_dict['test'])**2/float(len(diff_dict['test'])))))
