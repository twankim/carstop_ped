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

import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import manifold
from config.config import cfg
from skvideo.io import vread

commands = []
commands += ["click on top end of leftmost vertical line",
             "click on bottom of the same line"]
commands += ["click on top end of next vertical line",
             "click on bottom of the same line"] * (cfg.nvertlines - 1)
commands += ["click on left end of uppermost horizontal line",
             "click on right end of the same line"]
commands += ["click on left end of next horizontal line",
             "click on right end of the same line"] * (cfg.nhorzlines - 1)
commands += ["thanks, exit now!"]

def pixelToCamframe(points):
    points = np.array(points)
    points -= cfg.camera_center
    points[:,1] *= -1
    return points.astype(float) / cfg.camera_focal

def camframeToPixel(points):
    points = np.array(points) * cfg.camera_focal
    points[:,1] *= -1
    return points + cfg.camera_center
    
def inremovezone(point):
    return ((point[0] >= cfg.removezone[2]) and (point[0] <= cfg.removezone[3]) and
            (point[1] >= cfg.removezone[0]) and (point[1] <= cfg.removezone[1]))

class Clickevent:
    def __init__(self, commandlist, fig, scatter):
        self.scatter = scatter
        self.points = []
        self.count = 0
        self.commands = commandlist
        self.fig = fig
        print commandlist[0]
        
    def __call__(self,event):
        if event.inaxes:
            point = (int(event.xdata+.5), int(event.ydata+.5))
            if inremovezone(point):
                if self.count > 0:
                    self.count -= 1
                    del self.points[-1]
                    print("    last point removed!")
                    print(self.commands[self.count])
            elif self.count >= len(self.commands) - 1:
                print("   no more points to add...")
            else:
                self.count += 1
                self.points += [point]
                print("    {:d}, {:d} added".format(*point))
                print(self.commands[self.count])
            self.scatter.set_data([pointx[0] for pointx in self.points],
                                  [pointx[1] for pointx in self.points])
            self.fig.canvas.draw()
            plt.pause(.001)
        else:
            print("    Outside drawing area!")

def clickOnPoint(image, commandlist):
    pltsize = (image.shape[1]/cfg.mydpi, image.shape[0]/cfg.mydpi)
    fig = plt.figure("click on me", figsize=pltsize, dpi=cfg.mydpi)
    plt.axis('off')
    fig.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    fig.gca().xaxis.set_major_locator(NullLocator())
    fig.gca().yaxis.set_major_locator(NullLocator())

    scatterpoints, = fig.gca().plot([], [], 'go', markersize=6)    
    
    plt.imshow(image, aspect='equal')
    click = Clickevent(commandlist, fig, scatterpoints)
    plt.connect('button_press_event',click)
    plt.show()
    return click.points
    
def main(args):
    # if video, use first frame
    if args.isvideo:
        im = np.squeeze(vread(args.input,num_frames=1),axis=0)
    else:
        im = plt.imread(args.input)

    output = args.output

    ## now do grid stuff
    # (0,0) is bottom left
    grid = []
    for vline in range(cfg.nvertlines):
        grid += [(cfg.nhorzlines-1, vline), (0, vline)]
    for hline in range(cfg.nhorzlines):
        grid += [(cfg.nhorzlines-1-hline, 0), (cfg.nhorzlines-1-hline, cfg.nvertlines-1)]
    grid = np.array(grid, dtype=float) * cfg.gridlen
    street_length = (cfg.nvertlines-1)*cfg.gridlen
    height = (cfg.road_curve/street_length) * grid[:,1] * (street_length - grid[:,1])
    grid = np.append(grid, height[:,None], axis=1)

    ## ask user for lines
    #im = np.zeros((720,1280,3),dtype=np.uint8)
    im2 = im.copy()
    im2[cfg.removezone[0]:cfg.removezone[1], cfg.removezone[2]:cfg.removezone[3], :] = (200,0,0)
    points = clickOnPoint(im2, commands)
    assert len(points) == 2*(cfg.nhorzlines+cfg.nvertlines), "not all points were gathered"
    # convert to relative positions
    points = pixelToCamframe(points)


    ## find pose
    # need to start with a decent initial guess for position
    # otherwise optimization will definitely fail
    cfg.initial_camera_guess = np.array(cfg.initial_camera_guess)
    tmatrix_initial = np.zeros((4,4))
    tmatrix_initial[:3,:3] = manifold.expMat(cfg.initial_camera_guess[3:])
    tmatrix_initial[:3,3] = cfg.initial_camera_guess[:3]
    tmatrix_initial[3,3] = 1.
    invmatrix_initial = np.linalg.inv(tmatrix_initial)
    grid2 = grid.dot(invmatrix_initial[:3,:3].T) + invmatrix_initial[:3,3]
    #grid2 = manifold.unproject(grid, cfg.initial_camera_guess)
    initial_err = manifold.rms(manifold.pinhole(grid2), points)
    print("initial err: "+str(initial_err))
    camera_invpose, err = manifold.findCameraPose(grid2, points, plot=False)
    print("calibrated err: "+str(err))
    # print final transformation matrix
    tmatrix_inv = np.zeros((4,4))
    tmatrix_inv[:3,:3] = manifold.expMat(camera_invpose[3:])
    tmatrix_inv[:3,3] = camera_invpose[:3]
    tmatrix_inv[3,3] = 1.
    tmatrix = tmatrix_initial.dot(np.linalg.inv(tmatrix_inv))
    invtmatrix = tmatrix_inv.dot(invmatrix_initial)

    # print and save camera pose
    pose_string = '['+(('[{:.5f}'+', {:.5f}'*3+']\n')*4)[:-1]+']'
    pose_string = pose_string.format(*tmatrix.flatten().tolist())
    print("Camera pose matrix: ")
    print(pose_string)    
    with open(output, 'wb') as outputfile: outputfile.write(pose_string)

    ## plot result
    posed_grid = grid.dot(invtmatrix[:3,:3].T) + invtmatrix[:3,3]
    point_estimates = manifold.pinhole(posed_grid)
    point_estimates = camframeToPixel(point_estimates)
    im2 = im.copy()
    pltsize = (im2.shape[1]/cfg.mydpi, im2.shape[0]/cfg.mydpi)
    fig = plt.figure("result", figsize=pltsize, dpi=cfg.mydpi)
    plt.axis('off')
    fig.gca().set_axis_off()
    plt.subplots_adjust(top =1, bottom =0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    fig.gca().xaxis.set_major_locator(NullLocator())
    fig.gca().yaxis.set_major_locator(NullLocator())
    plt.imshow(im2, aspect='equal')
    for vline in range(cfg.nvertlines):
        line_estimate = point_estimates[vline*2:vline*2+2, :]
        plt.plot(line_estimate[:,0], line_estimate[:,1], 'g')
    for hline in range(cfg.nhorzlines):
        line_estimate = point_estimates[cfg.nvertlines*2+hline*2:cfg.nvertlines*2+hline*2+2,:]
        plt.plot(line_estimate[:,0], line_estimate[:,1], 'g')
    #plt.scatter(point_estimates[:,0], point_estimates[:,1], 6, 'g')
    plt.show()
    print 'ho'

def parse_args():
    def str2bool(v):
        return v.lower() in ('true', '1')

    parser = argparse.ArgumentParser(description=
                    'find Camera Pose using interactive session and a known grid')
    parser.add_argument('-isvideo', dest='isvideo',
                        help='Specify whether it is a video or an image',
                        default = True, type = str2bool)
    parser.add_argument('-input', dest='input',
                        help='Path to the input file (image/video)',
                        default = 'data/2017-10-22-075401.webm', type = str)
    parser.add_argument('-out', dest='output',
                        help='Path to the output file (save parameters)',
                        default = 'camerapose.txt', type = str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print "Called with args:"
    print args
    sys.exit(main(args))
