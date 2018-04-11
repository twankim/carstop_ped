# Copyright 2018 UT Austin/ Michal Motro. All Rights Reserved.
# Modifications copyright 2018 UT Austin/Taewan Kim
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
import numpy as np
import struct
import sys

distance_convert = .002
vert_angles = [-15,1,-13,3,-11,5,-9,7,-7,9,-5,11,-3,13,-1,15]
vert_cos = [np.cos(vert_angle*np.pi/180)*distance_convert \
            for vert_angle in vert_angles]
vert_sin = [np.sin(vert_angle*np.pi/180)*distance_convert \
            for vert_angle in vert_angles]
vert_sin = vert_sin * 2 # two firings at once
shortStruct = struct.Struct('<H')
longStruct = struct.Struct('<L')
firefmt = struct.Struct('<'+ 'HB'*32)
angle_convert = np.pi / 18000.
time_convert = 1e-6
angle_step = .0033 # radians of offset firing
step_cos = np.cos(angle_step)
step_sin = np.sin(angle_step)

min_distance_to_keep = 2. / distance_convert
max_distance_to_keep = 100. / distance_convert


def processLidarPacket(msg, last_angle = 0):
    xyz = np.empty((384,3))
    keep = np.empty((384,),dtype=bool)
    cut = False
    cut_idx = 384
    
    for k in range(12):
        angle = shortStruct.unpack(msg[k*100 + 2: k*100 + 4])[0] * angle_convert
        hcos = np.cos(angle)
        hsin = np.sin(angle)
        hcos2 = hcos * step_cos - hsin * step_sin
        hsin2 = hcos * step_sin + hsin * step_cos
        if angle < last_angle:
            cut = True
            cut_idx = k*32
        last_angle = angle
        
        fires = np.array(firefmt.unpack(msg[k*100 + 4 : k*100 + 100])[::2])
        xyz[k*32:k*32+16,0] = fires[:16] * hcos * vert_cos
        xyz[k*32+16:k*32+32,0] = fires[16:32] * hcos2 * vert_cos
        xyz[k*32:k*32+16,1] = fires[:16] * -hsin * vert_cos
        xyz[k*32+16:k*32+32,1] = fires[16:32] * -hsin2 * vert_cos
        xyz[k*32:k*32+32,2] = fires * vert_sin
        keep[k*32:k*32+32] = (fires > min_distance_to_keep) &\
                             (fires < max_distance_to_keep)
        
    time = longStruct.unpack(msg[1200:1204])[0] * time_convert
    return (xyz[:cut_idx][keep[:cut_idx]],
            cut,
            xyz[cut_idx:][keep[cut_idx:]],
            time,
            angle)


""" df = a readable object, ie from open()
    sweeptime = float in s, at most 30 minutes
    reads from filein until rotation closest to sweeptime
    returns this rotation,
        beginning data of next rotation that was already read,
        and current angle"""
def singleRotation(df, sweeptime):
    msg = df.read(1206)
    data_pre, cut, data_post, time, angle = processLidarPacket(msg, 0.)
    print("first time {:.0f}".format(time))
    if sweeptime < time - 600: # hour off
        time -= 3600
    distance_in_packets = int(max((sweeptime - time) * 715 - 10, 0))
    df.seek(distance_in_packets * 1206)
    
    rotation = []
    angle = 0.
    savenext = False
    
    while True:
        msg = df.read(1206)
        if len(msg) != 1206:
            assert len(msg) == 0, "len {:d} time {:.0f}".format(len(msg), time)
            print("ended too early, time {:.0f}".format(time))
            break
        data_pre, cut, data_post, time, angle = processLidarPacket(msg, angle)
        rotation.append(data_pre)
        if cut: # rotation completed
            if savenext:
                assert time > sweeptime - .06 and time < sweeptime + .06
                data = np.concatenate(rotation, axis=0)
                return data, data_post, angle
            
            else:
                rotation = [data_post]
                if sweeptime < time + .1:
                    savenext = True
                assert time < sweeptime + .1

""" example of how to use functions
    filename = str
    nameFun = function taking count integer, giving str
    starttime = float in s,
    k = int
"""
def saveKRotations(filename, nameFun, starttime, k):
    with open(filename,'rb') as df:
        data, data_post, angle = singleRotation(df, starttime)
        data.tofile(nameFun(0))
        
        count = 1
        rotation = [data_post]
        
        while count < k:
            msg = df.read(1206)
            data_pre, cut, data_post, time, angle = processLidarPacket(msg, angle)
            rotation.append(data_pre)
            if cut:
                data = np.concatenate(rotation, axis=0)
                data.tofile(nameFun(count))
                rotation = [data_post]
                count += 1

def loadKrotations(filename,start_time,k,r_fps):
    start_time = start_time % 3600 # Velodyne returns only mm:ss
    list_points = []
    with open(filename,'rb') as df:
        data, data_post, angle = singleRotation(df,start_time)
        list_points.append(data)

        i_frame = 1
        count = 1
        rotation = [data_post]
        while count < k:
            msg = df.read(1206)
            data_pre, cut, data_post, time,angle = processLidarPacket(msg,angle)
            rotation.append(data_pre)
            if cut:
                data = np.concatenate(rotation,axis=0)
                rotation = [data_post]
                if i_frame % r_fps == 0:
                    list_points.append(data)
                    count += 1
                i_frame +=1
    return list_points
            

#def saveSingleRotation(filename, savename, sweeptime):
#    with open(filename,'rb') as df:
#    
#        msg = df.read(1206)
#        data_pre, cut, data_post, time, last_angle = processLidarPacket(msg, 0.)
#        print("first time {:.0f}".format(time))
##        msg2 = df.read(1206)
##        data_pre, cut, data_post, time2, last_angle = processLidarPacket(msg2, 0.)
##        print("difference {:.4f}".format(time2 - time))
#        if sweeptime < time - 600: # hour off
#            time -= 3600
#        distance_in_bits = int(max((sweeptime - time) / .0014 - 10, 0)) * 1206
#        #print distance_in_bits
#        df.seek(distance_in_bits)
#        
#        rotation = []
#        angle = 0.
#        firstcut = False
#        
#        while True:
#            msg = df.read(1206)
#            if len(msg) != 1206:
#                assert len(msg) == 0, "len {:d} time {:.0f}".format(len(msg), time)
#                print("ended too early, time {:.0f}".format(time))
#                break
#            data_pre, cut, data_post, time, angle = processLidarPacket(msg, angle)
#            rotation.append(data_pre)
#            if cut: # rotation completed
#                if firstcut:    
#                    data = np.concatenate(rotation, axis=0)
#                    print(data.shape)
#                    data.tofile(savename)
#                    #np.save(savename, data)
#                    break
#                
#                else:
#                    rotation = [data_post]
#                    if sweeptime < time + .1:
#                        firstcut = True
                        
                        
                        
                        
if __name__ == '__main__':
    assert len(sys.argv) == 4
    samename = lambda x: sys.argv[2]
    saveKRotations(sys.argv[1], samename, float(sys.argv[3]), 1)
#    saveSingleRotation("/media/motrom/CarstopData3/4_1_18/Accord lidar/velocalib1.dat",
#                       "lidartest.bin", 1073.)
