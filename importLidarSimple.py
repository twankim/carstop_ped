#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import struct

fullpackedstructure = ">" + ("H"+"L"*8+"B"*8)*50
angle_conversion = np.pi / 5200.
distance_conversion = 10.**-5
vertical_angles = np.array((-.318505,-.2692,-.218009,-.165195,-.111003,
                            -.0557982,0.,.0557982))
vertical_components = np.sin(vertical_angles)
horizontal_components = np.cos(vertical_angles)
        
def read(filename, starttime, endtime): # time in seconds from start of collection
    with open(filename, 'rb') as f:    
        predictions = []
        currtime = -1
        outarray = np.zeros((0,5))
        while True:
            
            # check for new time
            msg = f.read(16)
            if len(msg) < 16:
                print("eof!")
                break
            if msg[:7] == 'NEWTIME':
                currtime = float(msg[7:])
                prefix = ''
            else:
                prefix = msg
    
            msg = prefix + f.read(2100-len(prefix))
            if len(msg) < 2100: 
                print("eof2!")
                break
            
            # window in seconds, only gather data in this range
            if currtime < starttime: continue
        
            # gather HVIR format
            firinglist = struct.unpack(fullpackedstructure, msg)
            firingdata = [(firinglist[i*17]*angle_conversion,
                           firinglist[i*17+1+j] * distance_conversion,
                           firinglist[i*17+9+j], vertical_components[j],
                    horizontal_components[j]) for i in range(50) for j in range(1,8)]
            firingdata = np.array(firingdata, dtype=float)
            
            # look for completed rotation
            anglechange = np.where(np.diff(firingdata[:,0])<0)[0]
            if len(anglechange) > 1:
                raise ValueError("multiple rotation completions in a row")
            elif len(anglechange) == 1:
                cutoff = anglechange[0]+1
            else:
                if outarray.shape[0] > 0 and firingdata[0,0] < outarray[-1,0]:
                    cutoff = 0
                else:
                    cutoff = -1
                
            if cutoff >= 0:
                data = np.append(outarray, firingdata[:cutoff,:], axis=0)
                outarray = firingdata[cutoff:,:]
                
                ## here we actually work with the data
                # low power returns, can keep or remove
                data = data[data[:,2] > 0]
                # get horizontal and vertical distances
                data[:,3] *= data[:,1]
                data[:,4] *= data[:,1]
                # ignore points hitting the roof of the car...
                data = data[data[:,4] > 1., :]
                xyz = np.array((np.cos(data[:,0])*data[:,4], #X
                                np.sin(data[:,0])*data[:,4], #Y
                                data[:,3])).T #Z
                #assert not np.any(np.isnan(xyz))
                predictions += [(currtime, xyz.copy())]
                if currtime > endtime:
                    break
     
            else:
                outarray = np.append(outarray, firingdata, axis=0)
    return predictions