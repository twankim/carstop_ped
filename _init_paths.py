# Modifications copyright 2017 UT Austin/Taewan Kim

import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.append(path)

this_path = os.path.dirname(__file__)

PATH_TF_RESEARCH = os.path.join(this_path,'..','models','research')

# det_path = os.path.join(tf_research_path,'object_detection')
slim_path = os.path.join(PATH_TF_RESEARCH,'slim')

if not os.path.exists(PATH_TF_RESEARCH):
    raise ValueError('You must download tensorflow research models'
                     'https://github.com/tensorflow/models/tree/master/research')
# if not os.path.exists(det_path):
#     raise ValueError('You must download tensorflow research object detectino API')
# if not os.path.exists(slim_path):
#     raise ValueError('You must download tensorflow research slim API')

add_path(PATH_TF_RESEARCH)
# add_path(slim_path)