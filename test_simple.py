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
from numpy import *
from platform import system

from moo_gradient import MOO_HyperVolumeGradient
from hv import HyperVolume

def f1(x):
   x1, x2 = x
   return x1 ** 2.0 + (x2 - 0.5) ** 2.0

def f2(x):
   x1, x2 = x
   return (x1 - 1)** 2.0 + (x2 - 0.5) ** 2.0

def f1_grad(x):
   x1, x2 = x
   return [2.0*x1, 2.0*(x2 - 0.5)]

def f2_grad(x):
   x1, x2 = x
   return [2.0*(x1 - 1), 2.0*(x2 - 0.5)]

# anther set of test functions
#def f1(x):
#    x1, x2 = x
#    return 1 - (x1 ** 2 + 1) * x2 ** 2
#
#def f2(x):
#    x1, x2 = x
#    return 1 - (x2 **2 + 1) * x1 ** 2
#
#def f1_grad(x):
#    x1, x2 = x
#    return [-(2 * x1 + 1) * x2 ** 2, -2.0 * (x1 ** 2 + 1) * x2 ]
#
#def f2_grad(x):
#    x1, x2 =x
#    return [-2.0 * (x2 ** 2 + 1) * x1, -(2 * x2 + 1) * x1 ** 2]

if system() == 'Darwin':
    # TKAgg backend is needed for MacOS
    import matplotlib
    matplotlib.use('TKAgg') 
    
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm

plt.ioff()
plt.style.use('ggplot')
rcParams['legend.numpoints'] = 1 
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['xtick.major.size'] = 7
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.size'] = 7
rcParams['ytick.major.width'] = 1
rcParams['axes.labelsize'] = 10
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'

_color = ['r', 'g', 'b', 'c', 'k', 'm', 'y']
np.random.seed(100)

# animation parameters
fig_width = 20
fig_height = fig_width * 9 / 16

delay = 0               # delays between frames
t_start = time.time()
fps = 100

dim = 2  
mu = 50  
maxiter = 500

fitness_func = [f1, f2]
fitness_grad = [f1_grad, f2_grad]
ref = [1, 1] 
step_size = 0.001
sampling = 'uniform' 
lb = [0, 0]
ub = [1, 1]
        
optimizer = MOO_HyperVolumeGradient(dim, 2, lb, ub, mu, fitness_func,
                                    fitness_grad, ref, step_size,
                                    sampling=sampling, maximize=False,
                                    maxiter=maxiter)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

# adjust the subplots layouts
fig.subplots_adjust(left=0.03, bottom=0.01, right=0.97, top=0.99, wspace=0.08, 
                    hspace=0.1)
                    
plot_layers = optimizer.dominated_steer == 'NDS'

if not plot_layers:
    line00, line01 = ax0.plot([], [],  'or', [], [], 'ob', ms=6, mec='none', 
                              alpha=0.5)
    line10, line11 = ax1.plot([], [],  'or', [], [], 'ob', ms=6, mec='none', 
                              alpha=0.5)
                              
fps_text = ax0.text(0.02, 0.95, '', transform=ax0.transAxes)  
hv_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

ax0.set_xlabel('$x_1$')
ax0.set_ylabel('$x_2$')
ax1.set_xlabel('$f_1$')
ax1.set_ylabel('$f_2$')
ax0.set_xlim([lb[0], ub[1]])
ax0.set_ylim([lb[1], ub[1]])
ax1.set_xlim([-0.1, ref[0] * 1.2])
ax1.set_ylim([-0.1, ref[1] * 1.2])

for ax in (ax0, ax1):
    ax.grid(True)
    ax.set_aspect('equal')

# hyper-volume calculation
hv = HyperVolume(ref)
pop_x_traject = {}

def init():
    fps_text.set_text('')
    hv_text.set_text('')
    
    if not plot_layers:
        line00.set_data([], [])
        line01.set_data([], [])
        line10.set_data([], [])
        line11.set_data([], [])
        return line00, line01, line10, line11
    else:
        ax0.lines = []
        ax1.lines = []
        return []

def animate(ind):
    global t_start
    
    time.sleep(delay/1000.0)
    front_idx, fitness = optimizer.step()

    print 'iteration {}'.format(optimizer.itercount)
    lines = []
    
    if optimizer.itercount % 10 == 0:
        fps = 10 / (time.time() - t_start)
        t_start = time.time()
        fps_text.set_text('FPS: {}'.format(fps))
        
    if plot_layers:
        ax0.lines = []
        ax1.lines = []
        
        fronts_idx = optimizer.fronts
        fronts = [fitness[:, idx] for idx in fronts_idx]
        x_eff = [optimizer.pop[:, idx] for idx in fronts_idx]
        
        for idx in range(mu):
            if pop_x_traject.has_key(idx):
                pop_x_traject[idx] = pop_x_traject[idx] + optimizer.pop[:, idx].tolist()
            else:
                pop_x_traject[idx] = optimizer.pop[:, idx].tolist()
        
        for i, front in enumerate(fronts):
            if dim == 1:
                lines += ax0.plot(x_eff[i][0, :], np.zeros(x_eff[i].shape[1]), ls='none',
                                  marker='o', ms=5, mec='none', alpha=1,
                                  color=_color[i%len(_color)])
            if dim == 2:
                lines += ax0.plot(x_eff[i][0, :], x_eff[i][1, :], ls='none',
                                  marker='o', ms=5, mec='none', alpha=1,
                                  color=_color[i%len(_color)])
            lines += ax1.plot(front[0, :], front[1, :], ls='none',
                             marker='o', ms=5, mec='none', alpha=1,
                             color=_color[i%len(_color)])
                             
        volume = hv.compute(fronts[0].T)
    else:
        dominated_idx = list(set(range(optimizer.mu)) - set(front_idx))
        
        front = fitness[:, front_idx]
        dominated = fitness[:, dominated_idx]
        x_front = optimizer.pop[:, front_idx]
        x_dominated = optimizer.pop[:, dominated_idx]
        
        volume = hv.compute(front.T)
        
        line00.set_data(x_front[0, :], x_front[1, :])
        line01.set_data(x_dominated[0, :], x_dominated[1, :])
        line10.set_data(front[0, :], front[1, :])
        line11.set_data(dominated[0, :], dominated[1, :])
        lines += [line00, line01, line10, line11]
    hv_text.set_text('HV = {}'.format(volume))
    return lines + [hv_text, fps_text]

ani = animation.FuncAnimation(fig, animate, frames=maxiter, interval=50, 
                              blit=True, init_func=init)

if 11 < 2:
    plt.rcParams['animation.ffmpeg_path'] = u'/usr/local/bin/ffmpeg'
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Hao Wang'), bitrate=1800)
    ani.save('test.mp4', writer=writer, dpi=100)

else:
    plt.show()

#fig = plt.figure(figsize=(12, 9))
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X, Y, fitness, rstride=1, cstride=1, cmap=cm.Blues, linewidth=0, alpha=0.3)

# fig2, ax2 = plt.subplots(1, 1, figsize=(9, 7),
#                    subplot_kw={'aspect': 'equal'}, dpi=100)
# ax.hold(True)

# CS = ax2.contour(X, Y, F, 80, cmap=cm.plasma_r, 
#                 linewidths=1)
# ax2.plot(linspace(0, 1, 100), linspace(0, 1, 100), 'k--', lw=1)

# fig2.colorbar(CS, ax=ax2)

# ax2.set_xlabel('$x^{(1)}$')
# ax2.set_ylabel('$x^{(2)}$')

# #for idx in range(self.mu):
# x = np.array(pop_x_traject[0])
# y = np.array(pop_x_traject[1])
# ax2.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy',
#                angles='xy', width=2, units='dots', scale=1, alpha=0.3,
#                color='k', headwidth=4, headlength=6)
               
# plt.tight_layout()

# plt.show()
            
