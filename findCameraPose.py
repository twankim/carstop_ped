#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 10/26/17
click on images to get points/lines of interest
"""

image = 'videoframe.png'
nvertlines = 10
nhorzlines = 5
gridlen = 2. * .3048
camera_center = (640, 360) # depends on image size
camera_focal = (919., 915.) # depends on camera, zoom, and image size
initial_camera_guess = [-226*.3048/12, 134*.3048/12, 81.*.3048/12, 0, 0, -.05]
mydpi = 96. # depends on monitor
removezone = (600,700,1000,1100)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import manifold

def pixelToCamframe(points):
    points = np.array(points)
    points -= camera_center
    points[:,1] *= -1
    return points.astype(float) / camera_focal

def camframeToPixel(points):
    points = np.array(points) * camera_focal
    points[:,1] *= -1
    return points + camera_center
    
def inRemoveZone(point):
    return ((point[0] >= removezone[2]) and (point[0] <= removezone[3]) and
            (point[1] >= removezone[0]) and (point[1] <= removezone[1]))

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
            if inRemoveZone(point):
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
    pltsize = (image.shape[0]/mydpi, image.shape[1]/mydpi)
    fig = plt.figure("click on me", figsize=pltsize, dpi=mydpi)
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
    
    
commands = []
commands += ["click on top end of leftmost vertical line",
             "click on bottom of the same line"]
commands += ["click on top end of next vertical line",
             "click on bottom of the same line"] * (nvertlines - 1)
commands += ["click on left end of uppermost horizontal line",
             "click on right end of the same line"]
commands += ["click on left end of next horizontal line",
             "click on right end of the same line"] * (nhorzlines - 1)
commands += ["thanks, exit now!"]
    
## now do grid stuff
# (0,0) is bottom left
grid = []
for vline in range(nvertlines):
    grid += [(nhorzlines-1, vline), (0, vline)]
for hline in range(nhorzlines):
    grid += [(nhorzlines-1-hline, 0), (nhorzlines-1-hline, nvertlines-1)]
grid = np.array(grid, dtype=float) * gridlen
grid = np.append(grid, np.zeros(((nvertlines+nhorzlines)*2,1)), axis=1)

## ask user for lines
#im = np.zeros((720,1280,3),dtype=np.uint8)
im = plt.imread(image)
im2 = im.copy()
im2[removezone[0]:removezone[1], removezone[2]:removezone[3], :] = (200,0,0)
points = clickOnPoint(im2, commands)
assert len(points) == 2*(nhorzlines+nvertlines), "not all points were gathered"
# convert to relative positions
points = pixelToCamframe(points)


## find pose
# need to start with a decent initial guess for position
# otherwise optimization will definitely fail
initial_camera_guess = np.array(initial_camera_guess)
tmatrix_initial = np.zeros((4,4))
tmatrix_initial[:3,:3] = manifold.expMat(initial_camera_guess[3:])
tmatrix_initial[:3,3] = initial_camera_guess[:3]
tmatrix_initial[3,3] = 1.
invmatrix_initial = np.linalg.inv(tmatrix_initial)
grid2 = grid.dot(invmatrix_initial[:3,:3].T) + invmatrix_initial[:3,3]
#grid2 = manifold.unproject(grid, initial_camera_guess)
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
print("Camera pose matrix: ")
print(np.round(tmatrix, decimals=5).tolist())

## plot result
posed_grid = grid.dot(invtmatrix[:3,:3].T) + invtmatrix[:3,3]
point_estimates = manifold.pinhole(posed_grid)
point_estimates = camframeToPixel(point_estimates)
im2 = im.copy()
pltsize = (im2.shape[0]/mydpi, im2.shape[1]/mydpi)
fig = plt.figure("result", figsize=pltsize, dpi=mydpi)
plt.axis('off')
fig.gca().set_axis_off()
plt.subplots_adjust(top =1, bottom =0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)
fig.gca().xaxis.set_major_locator(NullLocator())
fig.gca().yaxis.set_major_locator(NullLocator())
plt.imshow(im2, aspect='equal')
for vline in range(nvertlines):
    line_estimate = point_estimates[vline*2:vline*2+2, :]
    plt.plot(line_estimate[:,0], line_estimate[:,1], 'g')
for hline in range(nhorzlines):
    line_estimate = point_estimates[nvertlines*2+hline*2:nvertlines*2+hline*2+2,:]
    plt.plot(line_estimate[:,0], line_estimate[:,1], 'g')
#plt.scatter(point_estimates[:,0], point_estimates[:,1], 6, 'g')
plt.show()