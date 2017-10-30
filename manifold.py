#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 10/25/17
following https://pixhawk.org/_media/dev/know-how/jlblanco2010geometry3d_techrep.pdf
@author: motrom
"""
import numpy as np
import matplotlib.pyplot as plt

def skew_matrix(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def expMat(w):
    theta = np.sum(w**2)**.5
    if np.isclose(theta, 0): return np.eye(3)
    skew = skew_matrix(w)
    return np.eye(3) + np.sin(theta)/theta*skew +\
            (1-np.cos(theta))/theta**2*skew.dot(skew)
                 
def logMat(M):
    tr = M[0,0]+M[1,1]+M[2,2]
    w = np.array([M[2,1]-M[1,2],
                  M[0,2]-M[2,0],
                  M[1,0]-M[0,1]])
    if np.isclose(tr, 3) or np.isclose(tr, -1): return w / 2
    return w * np.arccos((tr-1)/2.) / (3+2*tr-tr**2)**.5
    
def project(x, w): # x = [n,3] array of points, w [6,] se3 vector
    return x.dot(expMat(w[3:]).T) + w[:3]
    
def unproject(x, w):
    return (x-w[:3]).dot(expMat(w[3:]).T)
    
def rms(x, y): return (np.sum((x-y)**2)/x.shape[0])**.5
    
def gradient(x, y):
    return np.append(np.mean(x-y,axis=0), np.mean(np.cross(y, x), axis=0))
    
def hessian_single(x, y):
    D = x.dot(y) * np.eye(3) - np.outer(x, y)
    return np.append(np.append(np.eye(3), -skew_matrix(y), axis=1),
                     np.append(skew_matrix(x), D, axis=1))
    
def hessian(x, y):
    A = np.tile(np.eye(3), (x.shape[0], 1, 1))
    C = np.array([skew_matrix(xi) for xi in x])
    B = -np.array([skew_matrix(yi) for yi in y])
    D = -np.einsum(x, [0,1], y, [0, 2], [0,1,2])
    D += A * np.einsum(x, [0,1], y, [0,1], [0])[:, None, None]
    full = np.append(np.append(A, B, axis=2), np.append(C, D, axis=2), axis=1)
    return np.mean(full, axis=0)
    
def pinhole(x):
    if x.ndim > 1:
        return x[:,1:]/x[:,0,None]
    return x[1:]/x[0]
    
def gradientCam(x, p):
    phat = pinhole(x)
    z = x[:,0]
    err = phat - p
    grad = np.zeros((x.shape[0], 6, 2))
    grad[:,1,0] = 1./z
    grad[:,2,1] = 1./z
    grad[:,0,:] = -phat/z[:,None]
    grad[:,4,0] = -phat[:,0]*phat[:,1]
    grad[:,4,1] = -1 - phat[:,1]**2
    grad[:,5,0] = 1 + phat[:,0]**2
    grad[:,5,1] = phat[:,0]*phat[:,1]
    grad[:,3,0] = -phat[:,1]
    grad[:,3,1] = phat[:,0]
    grad = np.einsum(grad, [0,1,2], err, [0,2], [0,1])
    return np.mean(grad, axis=0)
    
def findPose(x, y, plot=False):
    w = np.zeros((6,))
    x2 = project(x, w) # = x
    lasterr = rms(x2, y)
    errs = [lasterr]
    step = 1.
    for idx in range(10000):
        #grad = gradient(x2, y)
        grad = np.linalg.solve(hessian(x2, y), gradient(x2, y))
        w -= grad * step
        x2[:] = project(x, w)
        err = rms(x2, y)
        backtrackcount = 0
        while err > lasterr:
            backtrackcount += 1
            assert backtrackcount <= 10, idx
            step /= 2.
            w += grad * step
            x2[:] = project(x, w)
            err = rms(x2, y)
        errs += [err]
        if np.isclose(lasterr, err, rtol=1e-6):
            break
        lasterr = err
    if plot:
        plt.plot(errs)
    return w, err
    
def findCameraPose(x, y, plot=False):
    w = np.zeros((6,))
    x2 = project(x, w)
    lasterr = rms(pinhole(x2), y)
    errs = [lasterr]
    step = 1.
    for idx in range(10000):
        grad = gradientCam(x2, y)
        w -= grad*step
        x2[:] = project(x, w)
        err = rms(pinhole(x2), y)
        backtrackcount = 0
        while err > lasterr:
            backtrackcount += 1
            assert backtrackcount <= 20, idx
            step /= 2.
            w += grad * step
            x2[:] = project(x, w)
            err = rms(pinhole(x2), y)
        errs += [err]
        if np.isclose(lasterr, err, rtol=1e-6):
            break
        lasterr = err
    if plot:
        plt.figure()
        plt.plot(errs)
        plt.show()
    return w, err