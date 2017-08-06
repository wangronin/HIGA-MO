# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:58:39 2016

@author: wangronin
"""

import pdb

import numpy as np
from numpy import linspace, meshgrid, array, zeros

from mpm2 import *

import matplotlib.pyplot as plt
from matplotlib import rcParams

class Gradient_Descent:
    
    def __init__(self, x0, dim, gradient, step_size):
        
        self.x = array(x0)
        self.dim = dim
        self.gradient = gradient
        self.itercount = 0
        self.step_size = step_size
        
    
    def step(self):
        
        gra = self.gradient(self.x)
        
        self.x -= self.step_size * gra
        
        self.itercount += 1
        
        return self.x
        
        

if __name__ == '__main__':
    
    # ------------------------------ matplotlib settings -----------------------------
    fig_width = 22
    fig_height = 22 * 9 / 16
    
    plt.ioff()
    rcParams['legend.numpoints'] = 1 
    rcParams['xtick.labelsize'] = 20
    rcParams['ytick.labelsize'] = 20
    rcParams['xtick.major.size'] = 7
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.size'] = 7
    rcParams['ytick.major.width'] = 1
    rcParams['axes.labelsize'] = 25
    rcParams['font.size'] = 30
    rcParams['lines.markersize'] = 8
        
    dim = 2
    maxiter = 151
    
    state = np.random.get_state()
    
    P = initProblem(1, dim, 'random', 1, False, 'ellipse')
    
    f = P.objectiveFunction
    gradient_f = P.obj_grad
    
    f = lambda x: x[0] ** 2.0 + 25.0 * x[1] ** 2.0
    
    gradient_f = lambda x: array([2.0 * x[0], 50.0 * x[1]])
    
    np.random.set_state(state)
    
    opt = P.getLocalOptima()[0]
    print opt
    
    
    n_per_axis = 200  # plot presicion
    x = linspace(-1, 1, n_per_axis)
    y = linspace(-1, 1, n_per_axis) 
    X, Y = meshgrid(x, y)
    
    f_sample = array([f(p) for p in array([X.flatten(), Y.flatten()]).T]).reshape(-1, len(x))
    
    x0 = [0.5, 0.5]
    
    print x0
    
    optimizer = Gradient_Descent(x0, dim, gradient_f, 0.03)
    
    trajectory = zeros((dim, maxiter))
    
    for i in range(maxiter):
        trajectory[:, i] = optimizer.step()
        
        
    fig0, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), 
           subplot_kw={'aspect': 'equal'}, dpi=100)
           
    ax.grid(True)
    
    x, y = trajectory[0, :], trajectory[1, :]
    
    ax.contour(X, Y, f_sample, 10, colors='k', linewidths=1)
    ax.plot(opt[0], opt[1], 'r.', ms=8, alpha=0.5)
#    ax.plot(x, y, 'b.', ms=3, alpha=0.3)
    ax.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', 
                   angles='xy', width=3, units='dots', scale=1, alpha=0.3, 
                   color='b', headwidth=2, headlength=4)
                   
    plt.tight_layout()
    
    plt.show()
    