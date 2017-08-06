# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:50:46 2015

Testing script for multi-objective Hyper-Volume Gradient algorithm

@author: wangronin

@Notes: 
    Dec 14 2015: The fps of matplotlib plotting is so low, which is roughly 
                 20 for this experiment
"""

import pdb
import time
import numpy as np
from numpy import array, zeros, ones, pi

from moo_gradient import MOO_HyperVolumeGradient
from naive_optimizer import MOO_naive
from hv import HyperVolume

from functools import partial

import matplotlib
#matplotlib.use('TKAgg') # MacOSX backend cause problem in dynamic plot...

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.animation as animation

#from PyGMO import problem
from mpm2 import *

from MOO_problem import problem as my_problem

import warnings


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
rcParams['axes.labelsize'] = 30
rcParams['font.size'] = 30

_color = ['r', 'g', 'b', 'c', 'k', 'm', 'y']

# ----------- define your own objective functions and their gradients here -------


# Gradient field calculation of ZDT fitness functions 
def ZDT_grad_f1(ind, x):
    dim = len(x)
    if ind in range(1, 5):     # ZDT1
        f1_grad = zeros(dim)
        f1_grad[0] = 1
    
    elif ind == 6:
        f1_grad = zeros(dim)
        f1_grad[0] = -16.0 * x[0] * np.exp(-4.0 * x[0]) * np.sin(6.0*pi*x[0]) ** 6.0 - \
            6.0 * (np.sin(6.0*pi*x[0]) ** 5.0) * np.cos(6.0*pi*x[0])* 6.0 * pi * np.exp(-4.0 * x[0])
    
    return f1_grad
            
def ZDT_grad_f2(ind, x):
    dim = len(x)
    s = zeros(dim)
    s[0] = 1
    if ind == 1:     # ZDT1
        f1 = x[0]
        g = 1.0 + 9.0 / (dim - 1.0) * np.sum(x[1:])
        f1_grad = zeros(dim)
        f1_grad[0] = 1
        g_grad = 9.0 / (dim - 1.0) * ones(dim)
        g_grad[0] = 0
        
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                f2_grad = (1.0 - np.sqrt(f1 / g)) * g_grad - 0.5 * g * (f1 / g) ** (-0.5) * \
                    (g * f1_grad - f1 * g_grad ) / (g ** 2.0)
            except Warning:
                f2_grad = zeros(dim)
                
    if ind == 3:
        f1 = x[0]
        g = 1.0 + 9.0 / (dim - 1.0) * np.sum(x[1:])
        f1_grad = zeros(dim)
        f1_grad[0] = 1
        g_grad = 9.0 / (dim - 1.0) * ones(dim)
        g_grad[0] = 0
        
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                f2_grad = (1.0 - (f1 / g) ** 2.0) * g_grad - 0.5 * g * (f1 / g) ** (-0.5) * \
                    (g * f1_grad - f1 * g_grad ) / (g ** 2.0) - \
                    (g * f1_grad - f1 * g_grad ) / (g ** 2.0) * np.sin(10 * pi * f1) -\
                    (f1 / g) * np.cos(10* pi*f1) * 10 * pi * s
            except Warning:
                f2_grad = zeros(dim)
                
    if ind == 2:
        f1 = x[0]
        g = 1.0 + 9.0 / (dim - 1.0) * np.sum(x[1:])
        f1_grad = zeros(dim)
        f1_grad[0] = 1
        g_grad = 9.0 / (dim - 1.0) * ones(dim)
        g_grad[0] = 0
        
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                f2_grad = (1.0 - (f1 / g) ** 2.0) * g_grad - 2 * g * (f1 / g) * \
                    (g * f1_grad - f1 * g_grad ) / (g ** 2.0)
            except Warning:
                f2_grad = zeros(dim)
                
    if ind == 4:
        f1 = x[0]
        g = 1.0 + 10.0 * (dim - 1.0) + np.sum(x[1:] ** 2.0 - 10 * np.cos(4.0*pi*x[1:]))
        f1_grad = s
        g_grad = zeros(dim)
        g_grad[1:] = 2.0 * x[1:] + 10.0 * 4.0 * pi * np.sin(4.0 * pi * x[1:])  
        
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                f2_grad = (1.0 - np.sqrt(f1 / g)) * g_grad - 0.5 * g * (f1 / g) ** (-0.5) * \
                    (g * f1_grad - f1 * g_grad ) / (g ** 2.0)
            except Warning:
                f2_grad = zeros(dim)
    
    elif ind == 6:
        f1 = 1.0 - np.exp(-4.0 * x[0]) * np.sin(6.0*pi*x[0]) ** 6.0
        g = 1.0 + 9.0 * (np.sum(x[1:]) / (dim - 1.0)) ** 0.25
        f1_grad = zeros(dim)
        f1_grad[0] = -16.0 * x[0] * np.exp(-4.0 * x[0]) * np.sin(6.0*pi*x[0]) ** 6.0 - \
            6.0 * np.sin(6.0*pi*x[0]) ** 5.0 * np.cos(6.0*pi*x[0])* 6.0 * pi * np.exp(-4.0 * x[0])
            
        g_grad = zeros(dim)
        g_grad[1:] = 9.0 * 0.25 * (np.sum(x[1:]) / (dim - 1.0)) ** (0.25 - 1) / (dim - 1.0)
        try:
            f2_grad = -2.0 * (f1 / g) * (g * f1_grad - f1 * g_grad ) / (g ** 2.0)
        except Warning:
            f2_grad = zeros(dim)
        
    return f2_grad


# ---------------------------- initialize the optimizer --------------------------
#np.random.seed(100)



dim = 2
mu = 25
maxiter = int(400)


# setup ZDT problems from PyGMO
#prob_ind = 1
#prob = problem.zdt(prob_ind, dim)
#fitness = prob.objfun
#grad = [partial(ZDT_grad_f1, prob_ind), partial(ZDT_grad_f2, prob_ind)]
#f_dim = prob.f_dimension
#x_lb = array(prob.lb)
#x_ub = array(prob.ub)
#ref = [1.1, 1.1]
#ref2 = [1.1, 1.1]

# setup my own problems
prob_ind = 3
prob = my_problem(prob_ind, dim, 0.5)
fitness = prob.objfun
grad = prob.objgrad
f_dim = prob.f_dimension
x_lb = array(prob.lb)
x_ub = array(prob.ub)
ref = [1.1, 1.1]
ref2 = [1.1, 1.1]


step_size = 0.005

sampling = 'grid'
opts = {'lb' : x_lb,
        'ub' : x_ub,
        'maxiter' : maxiter,
        'heuristic' : 'M2',
        'non_dominated_sorting' : False,
        'enable_dominated' : True}
        
optimizer = MOO_HyperVolumeGradient(dim, f_dim, mu, fitness, grad, ref=ref, opts=opts,
                                    step_size=step_size, sampling=sampling, 
                                    maximize=False, normalize=True)


# --------------------------- initialize the plot setttings ----------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

# adjust the subplots layouts
fig.subplots_adjust(left=0.04, bottom=0, right=0.99, top=1, wspace=0.1, 
                    hspace=0.1)
                    
plot_layers = opts['non_dominated_sorting']

for ax in (ax0, ax1):
    ax.grid(True)
    ax.hold(True)
    ax.set_aspect('equal')
    
    
fps_text = ax0.text(0.02, 0.95, '', transform=ax0.transAxes)  
hv_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

    
ax0.set_xlabel('$x_1$')
ax0.set_ylabel('$x_2$')
ax1.set_xlabel('$f_1$')
ax1.set_ylabel('$f_2$')

ax0.set_xlim(x_lb[0]-0.05, x_ub[0]+0.05)
ax0.set_ylim(x_lb[1]-0.05, x_ub[1]+0.05)
ax1.set_xlim([0, 1.1])
ax1.set_ylim([0, 1.1])

if not plot_layers:
    
    line00, line01 = ax0.plot([], [],  'or', [], [], 'ob', ms=8, mec='none', 
                              alpha=0.5)
    line10, line11 = ax1.plot([], [],  'or', [], [], 'ob', ms=8, mec='none', 
                              alpha=0.5)
    

                              
#    line00.set_clip_on(False)
#    line01.set_clip_on(False)
#    line10.set_clip_on(False)
#    line11.set_clip_on(False)


# ------------------------------- The animation ----------------------------------
# hyper-volume calculation
hv = HyperVolume(ref2)

delay = 10
toggle = True
t_start = time.time()
fps_movie = 15

def init():
    fps_text.set_text('')
    hv_text.set_text('')
    
    if not plot_layers:
        line00.set_data([], [])
#        line01.set_data([], [])
        line10.set_data([], [])
#        line11.set_data([], [])
        
        return line00, line10
        
    else:
        ax0.lines = []
        ax1.lines = []
        
        return []


def animate(ind):
    global t_start
        
    time.sleep(delay/1000.0)
    
    front_idx, fitness = optimizer.step()
    
    if optimizer.itercount % 10 == 0:
        fps = 10 / (time.time() - t_start)
        t_start = time.time()
#        fps_text.set_text('FPS: {}'.format(fps))
    
    print 'Iteration: {}'.format(optimizer.itercount)
    
    lines = []
    
    if plot_layers:
        ax0.lines = []
        ax1.lines = []
        
        fronts_idx = optimizer.fronts
        fronts = [fitness[:, idx] for idx in fronts_idx]
        x_eff = [optimizer.pop[:, idx] for idx in fronts_idx]
        
        for i, front in enumerate(fronts):
            if dim == 2:
                lines += ax0.plot(x_eff[i][0, :], x_eff[i][1, :], ls='none',
                                  marker='o', ms=5, mec='none', alpha=0.5,
                                  color=_color[i%len(_color)])
            lines += ax1.plot(front[0, :], front[1, :], ls='none',
                             marker='o', ms=5, mec='none', alpha=0.5,
                             color=_color[i%len(_color)])
                             
        volume = hv.compute(fronts[0].T)
        
    else:
        
        lines = ax0.lines + ax1.lines
        
        dominated_idx = list(set(range(optimizer.mu)) - set(front_idx))
        
        front = fitness[:, front_idx]
        dominated = fitness[:, dominated_idx]
        x_front = optimizer.pop[:, front_idx]
        x_dominated = optimizer.pop[:, dominated_idx]
        
        volume = hv.compute(front.T)
        
        if optimizer.itercount == 1:
            marker = 's'
            data = optimizer.pop
        else:
            marker = '+'
            data = optimizer.pop
        
        if dim == 2:
            line00.set_data(x_front[0, :], x_front[1, :])
#        line01.set_data(x_dominated[0, :], x_dominated[1, :])
            line01, = ax0.plot(x_dominated[0, :], x_dominated[1, :], ls='none',
                           marker=marker, ms=10, mec='0.4', mfc='0.2')
        else:
            line01, = ax0.plot([], [])
                           
        line10.set_data(front[0, :], front[1, :])
#        line11.set_data(dominated[0, :], dominated[1, :])
        line11, = ax1.plot(dominated[0, :], dominated[1, :], ls='none',
                           marker=marker, ms=10, mec='0.4', mfc='0.2')
                           
        
        lines += [line00, line01, line10, line11]
        
#    hv_text.set_text('HV = {}'.format(volume))
    if ind == 300:
        fig.savefig('{}-conv.eps'.format(opts['heuristic']))
    
    return lines + [hv_text, fps_text]
    

# calculate the actual interval between frames    
t0 = time.time()
animate(0)
t1 = time.time()
interval = 1. / fps_movie - (t1 - t0)

rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
ani = animation.FuncAnimation(fig, animate, frames=maxiter, interval=interval, 
                              blit=True, init_func=init)

# save the movie
#ani.save('test.mp4', fps=fps, extra_args=['-vcodec', 'libx264'], dpi=200)

plt.show()   
        
print optimizer.stop_list 
        
        
            
