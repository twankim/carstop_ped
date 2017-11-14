# Modifications copyright 2017 UT Austin/Taewan Kim

import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.append(path)

this_path = os.path.dirname(__file__)

tf_research_path = os.path.join(this_path,'..','models','research')

# det_path = os.path.join(tf_research_path,'object_detection')
slim_path = os.path.join(tf_research_path,'slim')

if not os.path.exists(tf_research_path):
    raise ValueError('You must download tensorflow research models'
                     'https://github.com/tensorflow/models/tree/master/research')
# if not os.path.exists(det_path):
#     raise ValueError('You must download tensorflow research object detectino API')
# if not os.path.exists(slim_path):
#     raise ValueError('You must download tensorflow research slim API')

add_path(tf_research_path)
# add_path(slim_path)