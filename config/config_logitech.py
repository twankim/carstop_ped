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

# Configureation file for dataset
from myeasydict import EasyDict as edict

__C = edict()

cfg = __C

__C.nvertlines = 10
__C.nhorzlines = 5
__C.gridlen = 2. * .3048
__C.camera_center = (640, 360) # depends on image size
__C.camera_focal = (919., 915.) # depends on camera, zoom, and image size
__C.initial_camera_guess = [-226*.3048/12, 134*.3048/12, 81.*.3048/12, 0, 0, -.05]
__C.mydpi = 96. # depends on monitor
__C.removezone = (600,700,1000,1100)
__C.road_curve = 0.
