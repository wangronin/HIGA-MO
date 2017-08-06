# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 23:08:26 2016

@author: wangronin
"""

import pdb
import numpy as np
from numpy import array, linspace, meshgrid

import matplotlib as mpl
#mpl.use('MacOSX')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as colors

from mpm2 import createProblem

# plot settings
plt.ioff()
fig_width = 16
fig_height = fig_width * 9 / 16
_color = ['r', 'm', 'b', 'g', 'c']

rcParams['font.size'] = 22
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['legend.numpoints'] = 1 
rcParams['xtick.labelsize'] = 17
rcParams['ytick.labelsize'] = 17
rcParams['xtick.major.size'] = 7
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.size'] = 7
rcParams['ytick.major.width'] = 1


def biobj_contour_plot(ax, objfun, x_lb, x_ub, n_per_axis=200):

    # plot the contour lines of objective functions
    x = linspace(x_lb[0], x_ub[0], n_per_axis)
    y = linspace(x_lb[1], x_ub[1], n_per_axis)
    X, Y = meshgrid(x, y)

    if isinstance(objfun, (list, tuple)):
        samples = array([[objfun[0](p), objfun[1](p)] for p in array([X.flatten(), \
            Y.flatten()]).T])

    elif hasattr(objfun, '__call__'):
        samples = array([objfun(p) for p in array([X.flatten(), \
            Y.flatten()]).T])

    f1_sample = samples[:, 0].reshape(-1, len(x))
    f2_sample = samples[:, 1].reshape(-1, len(x))

    CS0 = ax.contour(X, Y, f1_sample, 20, colors='k', linewidths=1)
    CS1 = ax.contour(X, Y, -f2_sample, 20, colors='k', linewidths=1)
#    ax.set_title('solid: $f_1$, dashed: $f_2$', fontsize=17, y=1.05)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    return CS0, CS1


def efficient_set(prob1, prob2, peak1, peak2, is_ridge=False):
    
    a0 = 1e-20
    aN = 5e3
    N = 2e3
    s = (aN / a0) ** (1.0 / (N-1))
    k = [a0 * s ** i for i in np.arange(0, N)]
    
    D1, D2 = prob1.peaks[peak1].D, prob2.peaks[peak2].D
    
    c1 = array(prob1.peaks[peak1])
    c2 = array(prob2.peaks[peak2])
    
    efficient, pseudo = [], []
    for i, kk in enumerate(k):
        res = c1 - np.dot(c1 - c2, np.dot(np.linalg.inv(D1 / kk + D2), D2).T)
        if is_ridge:
            a = prob1.getActivePeak(res)
            b = prob2.getActivePeak(res)
            if np.all(a == c1) and np.all(b == c2):
                efficient += [res.tolist()]
            else:
                pseudo += [res.tolist()]
        
    return array(efficient).T, array(pseudo).T
    
    
def plot_pareto_front(ax, f1, f2, efficient, linestyles, color):
    
    x = [f1(_) for _ in efficient.T]
    y = [f2(_) for _ in efficient.T]
    
    ax.plot(x, y, ls=linestyles, color=color, alpha=0.9, lw=4)
  
    
def plot_MPM2_ridge(ax, prob, x_lb, x_ub, n_per_axis=550):
    
    x = linspace(x_lb[0], x_ub[0], n_per_axis)
    y = linspace(x_lb[1], x_ub[1], n_per_axis) 
    X, Y = meshgrid(x, y)
    
    points = np.c_[X.flatten(), Y.flatten()]
    
    Z = np.zeros(points.shape[0])
    for i, p in enumerate(points):
        peak_values = [prob.g(p, peak) for peak in prob.peaks]
        a, b = np.sort(peak_values)[-2:]
        if np.isclose(a, b, 7.*1e-3):
            Z[i] = 1.
    Z = Z.reshape(-1, len(x))

    ax.contour(X, Y, Z, 1, colors='k', linewidths=4, linestyles='solid')
    
    
def plot_contour_gradient(ax, f, grad, x_lb, x_ub, title='f', n_per_axis=200):
    
    fig = ax.figure
    
    x = linspace(x_lb[0], x_ub[0], n_per_axis)
    y = linspace(x_lb[1], x_ub[1], n_per_axis) 
    X, Y = meshgrid(x, y)
    
    fitness = array([f(p) for p in np.c_[X.flatten(), Y.flatten()]]).reshape(-1, len(x))
    ax.contour(X, Y, fitness, 25, colors='k', linewidths=1)
    
    # calculate function gradients   
    x1 = linspace(x_lb[0], x_ub[0], np.floor(n_per_axis / 10))
    x2 = linspace(x_lb[1], x_ub[1], np.floor(n_per_axis / 10)) 
    X1, X2 = meshgrid(x1, x2)     
                       
    dx = array([grad(p) for p in np.c_[X1.flatten(), X2.flatten()]])
    dx_norm = np.sqrt(np.sum(dx ** 2.0, axis=1))
    dx /= dx_norm.reshape(-1, 1)
    dx1 = dx[:, 0].reshape(-1, len(x1))
    dx2 = dx[:, 1].reshape(-1, len(x1))
    
    CS = ax.quiver(X1, X2, dx1, dx2, dx_norm, cmap=plt.cm.gist_rainbow, 
                    norm=colors.LogNorm(vmin=dx_norm.min(), vmax=dx_norm.max()),
                    headlength=5)
    
    fig.colorbar(CS, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True)
    ax.set_title(title, y=1.05)
    ax.set_xlim(x_lb[0], x_ub[0])
    ax.set_ylim(x_lb[1], x_ub[1])
    

# Problem setup
dim = 2
P1 = createProblem(1, dim, 'random', 1, True, 'ellipse')
P2 = createProblem(3, dim, 'random', 5, True, 'ellipse')

f1 = P1.objectiveFunction
f1_grad = P1.obj_grad
f2 = P2.objectiveFunction
f2_grad = P2.obj_grad

x_lb = [0] * dim
x_ub = [1] * dim

# Plot the contour lines together with the gradient field
fig0, (ax0, ax1) = plt.subplots(1, 2, figsize=(fig_width, fig_height), 
                                subplot_kw={'aspect': 'equal'}, dpi=100)
                                    
#plot_MPM2_ridge(ax1, P2, x_lb, x_ub)
plot_contour_gradient(ax0, f1, f1_grad, x_lb, x_ub, title='$f_1$', n_per_axis=250)
plot_contour_gradient(ax1, f2, f2_grad, x_lb, x_ub, title='$f_2$', n_per_axis=250)
                                    

if 1 < 2:
    # Plot the Pareto efficient set
    fig2, (ax0, ax1) = plt.subplots(1, 2, figsize=(fig_width, fig_height), 
                             subplot_kw={'aspect': 'equal'}, dpi=100)
    
    ax0.hold(True)
    ax1.hold(True)
    
    biobj_contour_plot(ax0, (f1, f2), x_lb, x_ub, n_per_axis=200)
    plot_MPM2_ridge(ax0, P2, x_lb, x_ub)
    
    n_peak1 = len(P1.peaks)
    n_peak2 = len(P2.peaks)
    
    for i in range(n_peak1):
        for j in range(i, n_peak2):
            efficient, pseudo = efficient_set(P1, P2, i, j, True)
            color_ = plt.cm.Set1((i+j)/(1.0*n_peak1 + n_peak2))
            
            if len(efficient) != 0:
                ax0.plot(efficient[0, :], efficient[1, :], ls='-', color=color_, 
                         lw=5, alpha=1.)
                plot_pareto_front(ax1, f1, f2, efficient, '-', color_)
                         
            if len(pseudo) != 0:
                ax0.plot(pseudo[0, :], pseudo[1, :], ls='--', color=color_, 
                         lw=4, alpha=1.)
                plot_pareto_front(ax1, f1, f2, pseudo, '--', color_)
    
    ax1.grid(True)
    ax1.set_xlabel('$f_1$')
    ax1.set_ylabel('$f_2$')
    ax1.set_ylim([0, 1])

plt.show()


