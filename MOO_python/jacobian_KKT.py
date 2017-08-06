# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 22:51:50 2016

@author: wangronin
"""

import pdb
import numpy as np
from numpy import linspace, meshgrid, array

import matplotlib as mpl
#mpl.use('MacOSX')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpm2 import createProblem
from matplotlib import cm

P1 = createProblem(1, 2, 'random', 1, False, 'sphere')
P2 = createProblem(1, 2, 'random', 10, False, 'sphere')

#P2.peaks[1][0] = 0.1
#P2.peaks[1][1] = 0.8
#
#P1.peaks[0].height = 0.5
#P1.peaks[0].radius = 3
P1.peaks[0].shape = 0.5

#P2.peaks[0].radius = 0.5
P2.peaks[0].shape = 0.5
#
#P1.peaks[1].height = 0.7
#P1.peaks[1].radius = 2
#P1.peaks[1].shape = 2
#
#P1.peaks[1][0] = 0.4
#P1.peaks[1][1] = 0.4
#P1.peaks[1].height = 0.8
#P1.peaks[1].radius = 0.5
#P1.peaks[1].shape = 2
#
#P1.peaks[2].height = 0.7
#P1.peaks[2].radius = 3
#P1.peaks[2].shape = 2

dim = 2
x_lb = [-0.5] * dim
x_ub = [1.5] * dim

def jacobian_det(ax, grad1, grad2, x_lb, x_ub, title='f', n_per_axis=200):
    
    x1 = linspace(x_lb[0], x_ub[0], n_per_axis)
    x2 = linspace(x_lb[1], x_ub[1], n_per_axis)
    X1, X2 = meshgrid(x1, x2)
    
    f1_dx = array([grad1(p) for p in np.c_[X1.flatten(), X2.flatten()]])
    f2_dx = array([grad2(p) for p in np.c_[X1.flatten(), X2.flatten()]])
    f1_dx_norm = np.sqrt(np.sum(f1_dx ** 2.0, axis=1))
    f1_dx /= f1_dx_norm.reshape(-1, 1)
    f2_dx_norm = np.sqrt(np.sum(f2_dx ** 2.0, axis=1))
    f2_dx /= f2_dx_norm.reshape(-1, 1)
    
    Z = np.abs(array([np.linalg.det(np.c_[f1_dx[i], f2_dx[i]]) \
        for i in range(f1_dx.shape[0])]) )
    Z /= Z.max()
    Z = Z.reshape(-1, len(x1))
            
    ax.plot_surface(X1, X2, Z, rstride=1, cstride=1,
                    cmap=cm.Greys, linewidth=0, alpha=0.7)
                    
    ax.contour(X1, X2, Z, 30, colors='k', linewidths=1, offset=0)

# fitness function and their gradients
fitness = [P1.objectiveFunction, P2.objectiveFunction]

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.hold(True)

x = linspace(x_lb[0], x_ub[0], 100)
y = linspace(x_lb[1], x_ub[1], 100)
X, Y = meshgrid(x, y)

pos = array([X.flatten(), Y.flatten()]).T
Z1 = array([fitness[0](p) for p in pos]).reshape(-1, len(x))
Z2 = -array([fitness[1](p) for p in pos]).reshape(-1, len(x))

jacobian_det(ax, P1.obj_grad, P2.obj_grad, x_lb, x_ub)

#ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.Blues, linewidth=0, alpha=0.3)
#ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap=cm.Oranges, linewidth=0, alpha=0.2)

ax.contour(X, Y, Z1, 30, cmap=cm.Blues, linewidths=1, offset=1)
ax.contour(X, Y, Z2, 30, cmap=cm.Oranges, linewidths=1, offset=1)

#p1, p2 = P2.peaks[1], P1.peaks[1]
#
#k = (p2[1] - p1[1]) / (p2[0]- p1[0])
#c = p1[1] - k * p1[0]
#
#line_x = linspace(p1[0], p2[0], 2000)
#line_y =  k * line_x + c
#
#line_z = [max(-fitness[0](p), -fitness[1](p)) for p in c_[line_x, line_y]]
#
#ax.plot(line_x, line_y, line_z, color='m', lw=3, alpha=0.6) 
#
#ax.plot(line_x, line_y, ones(shape(line_x)), color='m', lw=3, alpha=0.6) 

#p1, p2 = P2.peaks[2], P1.peaks[1]
#
#k = (p2[1] - p1[1]) / (p2[0]- p1[0])
#c = p1[1] - k * p1[0]
#
#line_x = linspace(p1[0], p2[0], 2000)
#line_y =  k * line_x + c
#
#line_z = [max(-fitness[0](p), -fitness[1](p)) for p in c_[line_x, line_y]]
#
#ax.plot(line_x, line_y, line_z, color='c', lw=3, alpha=0.6) 
#
#ax.plot(line_x, line_y, ones(shape(line_x)), color='c', lw=3, alpha=0.6) 

ax.set_zlim([0, 1])
ax.set_zscale('linear')

ax.view_init(elev=23., azim=-90)


plt.show()