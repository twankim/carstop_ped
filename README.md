# carstop_ped
Generating Pedestrian Detection Dataset for CARSTOP Project

Taewan Kim and Micahel Motro, The University of Texas at Austin.

[TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) is used for the developement. If your current path $CURR, please download [tensorflow/models repo](https://github.com/tensorflow/models) (clone repo) as $CURR/models. Then, follow the instructions for installing [object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

## Dataset Directory Structure
Save data collected from field testings at location *INPUT*. In this directory, you should have the sub folders containing collected data with valid *timestamps.txt* file, and a textfile *data_split.txt* to provide the information of the structure of train/val/test datasets.
```
INPUT
    data_split.txt # Text file specifyng the subfolders to be used in train/val/test set
    dataset0 # Name of a directory can be anything
        cam.mkv # video file
        lidar.dat # LIDAR file
        timestamps.txt # Text file specifying the timestamps to be used in data generation
    .
    .
    .
    
    datasetk
        cam.mkv # video file
        lidar.dat # LIDAR file
        timestamps.txt # Text file specifying the timestamps to be used in data generation
```

#### Example of a *data_split.txt* file
```
train dataset0 dataset1 dataset2 dataset3 dataset4
val dataset5 dataset6
test dataset7 dataset8 dataset9
```
You can have up to 3 lines, where each line starts with the *split set* name. Must be one of these: (*train*, *val*, or *test*). Then name of the subfolders to be used for each split set is provided in the same line separated by a single space. Assume that the subfolder is located in the directory *INPUT*. (ex. path to the subfolders are *INPUT/dataset0, INPUT/dataset1, ...*)

#### Example of a *timestamps.txt* file
```
hh:mm:ss hh:mm:ss
hh:mm:ss hh:mm:ss
hh:mm:ss hh:mm:ss
```
Each line is composed of two *hh:mm:ss* formatted strings separated by a single space, which are *start* and *end* time stamps, respectively. These start/end timestamps defines the portion of data (*cam.mkv* or *lidar.dat*) to be used as train/val/test set. New dataset will be generated per frame only on these specified segments.

## Generating the dataset
A sample command for generating a dataset is provided in the [scripts folder](https://github.com/twankim/carstop_ped/blob/master/scripts/gen_data.sh). Basically you are running a python file *cstopp_gen_data.py* with some input arguments.
```
    --data_pre      Type of a dataset for using pretrained model (coco or kitti)
    --model         Path to the frozen tensorflow object detection API's graph file. ex)frozen_inference_graph.pb
    --label         Path to the label map file. Usually located in tensorflow object detection API 
                    (ex) object_detection/data/mscoco_label_map.pbtxt
    --num_classes   Number of classes to consider for the pretrained model. 
                    (If you select 9, labels 0~9 are considered. Usually 0 are kept for a background)
    --input         Path to the input data (ex) /data/cstopp
    --output        Path to save the output data. If not specified, --input location will be used.
    --f_split       Path to the text file specifying split info. ex) /data/cstopp/data_split.txt
    --fps_in        fps(Frames per second) of the input video (Usally recorded as 10 fps)
    --fps_out       fps(Frames per second) of the output file. Data will be generated per frame using this rate)
    --is_vout       Boolean value. Whether to generate a video with labels (bounding boxes) or not.
```
Download pretrained models from [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
After running a code, three folders (depends on the split text file) will be generated with names *train,val,test* in the *--output* path.
