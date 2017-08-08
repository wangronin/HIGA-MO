# -*- coding: utf-8 -*-
"""
Created on Thur Mar 3 16:50:46 2015

Testing script for multi-objective Hyper-Volume Gradient algorithm

@author: wangronin

@Notes:
    Dec 14 2015: The fps of matplotlib plotting is so low, which is roughly
                 20 for this experiment
"""

import pdb
import sys, os, time

import numpy as np
from numpy import array, linspace, meshgrid, zeros

from naive_optimizer import MOO_naive
from moo_gradient import MOO_HyperVolumeGradient
from mo_problem import simple
from hv import HyperVolume

from pandas import DataFrame, read_csv
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages

from mpm2 import createProblem
from copy import deepcopy

# ------------------------------ matplotlib settings -----------------------------
fig_width = 22
fig_height = 22 * 9 / 16

plt.ioff()
rcParams['legend.numpoints'] = 1
rcParams['xtick.labelsize'] = 30
rcParams['ytick.labelsize'] = 30
rcParams['xtick.major.size'] = 10
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.size'] = 10
rcParams['ytick.major.width'] = 1
rcParams['axes.labelsize'] = 30
rcParams['font.size'] = 30
rcParams['lines.markersize'] = 11
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'


_tab = ' ' * 4

# generate colors
colors = [plt.cm.Set1(i) for i in linspace(0, 1, 11)]
#colors = ['r', 'g', 'm', 'c', 'b']


# ---------------------------- function definitions --------------------------------
def setup_MPM2_problem(dim, n_peaks1, n_peaks2, seed1, seed2, shape1='ellipse',
                       shape2='ellipse', verbose=False):

    # setup MPM2 functions
    # call signature: npeaks, dimensions, topology, random seed, rotated, peak shape
    P1 = createProblem(n_peaks1, dim, 'random', seed1, True, shape1)
    P2 = createProblem(n_peaks2, dim, 'random', seed2, True, shape2)

    # fitness function and their gradients
    fitness = [P1.objectiveFunction, P2.objectiveFunction]
    grad = [P1.obj_grad, P2.obj_grad]

    f_dim = 2

    if verbose:
        # print local optima information
        print 'F1 local optima:',
        print P1.getLocalOptima()
        print

        print 'F2 local optima:',
        print P2.getLocalOptima()
        print

    # search domain: box constraints
    x_lb = array([0] * dim)
    x_ub = array([1] * dim)

    # reference points
    ref = [1.1, 1.1]

    return fitness, grad, f_dim, x_lb, x_ub, ref


def setup_ZDT_problem(dim, prob_index):
    from mo_problem import zdt

    prob = zdt(prob_index, dim)
    x_lb = array(prob.lb)
    x_ub = array(prob.ub)

    ref = [11, 11] if prob_index == 6 else [1.1, 1.1]

    return prob.objfun, prob.objfun_dx, prob.f_dim, x_lb, x_ub, ref


def setup_problem(dim, index):

    # setup MPM2 functions
    p = simple(index, dim)

    fitness = p.objfun
    grad = p.objgrad
    f_dim = 2

    # box constraints
    x_lb = p.lb
    x_ub = p.ub

    ref = [1.1, 1.1]

    return fitness, grad, f_dim, x_lb, x_ub, ref


def setup_HIGA_optimizer(dim, f_dim, mu, fitness, grad, x_lb, x_ub, ref,
                         step_size, maxiter):

    optimizer = MOO_HyperVolumeGradient(dim, 2, x_lb, x_ub, mu, fitness,
                                        grad, ref, step_size, sampling=sampling, 
                                        maximize=False, maxiter=maxiter, 
                                        steer_dominated='NDS', normalize=True)

    return optimizer


def setup_naive_optimizer(dim, f_dim, mu, fitness, x_lb, x_ub, ref,
                       step_size, maxiter):

    opts = {
            'lb' : x_lb,
            'ub' : x_ub,
            }

    optimizer = MOO_naive(dim, f_dim, mu, fitness, opts=opts, step_size=step_size,
                           sampling='lhs', maxiter=maxiter, maximize=False)

    return optimizer


def setup_biobj_plot(x_lb, x_ub, ref, is_statistics=False):

    n_subplot = 3 if is_statistics else 2
    fig, axes = plt.subplots(1, n_subplot, figsize=(fig_width, fig_height))

    # adjust the subplots layouts
    fig.subplots_adjust(left=0.04, bottom=0.1, right=0.99, top=0.9, wspace=0.15,
                        hspace=0.1)

    for ax in axes:
        ax.grid(True)
        ax.hold(True)
        ax.set_aspect('equal')

    hv_text = axes[1].text(0.02, 0.963, '', transform=axes[1].transAxes,
                           fontsize=18)

    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[1].set_xlabel('$f_1$')
    axes[1].set_ylabel('$f_2$')

    r = x_ub - x_lb
    ratio = r[0] / r[1]
    axes[0].set_aspect(aspect=str(ratio))
#    axes[0].set_xlim([x_lb[0], x_ub[0]])
#    axes[0].set_ylim([x_lb[1], x_ub[1]])
    axes[0].set_xlim(x_lb[0]-0.05, x_ub[0]+0.05)
    axes[0].set_ylim(x_lb[1]-0.05, x_ub[1]+0.05)
    axes[1].set_xlim([0, ref[0]+0.05])
    axes[1].set_ylim([0, ref[1]+0.05])

    if n_subplot == 3:
         axes[2].set_xlabel('iteration')
         axes[2].set_ylabel('hypervolume')

    # plot the reference point
    axes[1].plot(*ref, color='k', ls='none', marker='.', ms=15)
    axes[1].hlines(ref[1], 0, ref[0], linestyles='dashed')
    axes[1].vlines(ref[0], 0, ref[1], linestyles='dashed')
    axes[1].text(ref[0]+0.01, ref[1]+0.01, r'$\mathbf{r}$')

    return fig, axes, hv_text


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

    CS0 = ax.contour(X, Y, f1_sample, 30, colors='k', linewidths=1)
    CS1 = ax.contour(X, Y, -f2_sample, 30, colors='k', linewidths=1)
    ax.set_title('solid: $f_1$, dashed: $f_2$', fontsize=17)

    return CS0, CS1


def run_optimizer(optimizer, maxiter, mu, ref):

    # hyper-volume calculation
    hv = HyperVolume(ref)

    # track the population
    pop_track = DataFrame(zeros((maxiter, 2 * mu)), \
        columns=['x^{}_{}'.format(i, j) for i in range(1, mu+1) for j in [1, 2]])
    pop_f_track = DataFrame(zeros((maxiter, 2 * mu)), \
        columns=['f^{}_{}'.format(i, j) for i in range(1, mu+1) for j in [1, 2]])
    dominance_track = DataFrame(zeros((maxiter, mu), dtype='int'),
                                columns=range(1, mu+1))

    hv_track = zeros(maxiter)
    hist_sigma = zeros((mu, maxiter))

    # recording the trajectories of dominated points
    pop_x_traject = {}
    pop_f_traject = {}

    for i in range(maxiter):
        # invoke the optimizer by step
        ND_idx, _ = optimizer.step()
        pop = deepcopy(optimizer.pop)
        fitness = deepcopy(_)
        fronts_idx = optimizer.fronts

        # compute the hypervolume indicator
        PF = fitness[:, fronts_idx[0]]
        hv_track[i] = hv.compute(PF.T)

        # tracking the whole population
        pop_track.loc[i] = pop.reshape(1, -1, order='F')
        pop_f_track.loc[i] = fitness.reshape(1, -1, order='F')
        for j, f in enumerate(fronts_idx):
            dominance_track.iloc[i, f] = j



        # record the trajectory of dominated points
        for j, idx in enumerate(range(optimizer.mu)):
            if pop_f_traject.has_key(idx):
                if any(pop_f_traject[idx][-1, :] != fitness[:, j]):
                    pop_f_traject[idx] = np.vstack((pop_f_traject[idx], fitness[:, j]))
            else:
                pop_f_traject[idx] = np.atleast_2d(fitness[:, j])

            if pop_x_traject.has_key(idx):
                if any(pop_x_traject[idx][-1, :] != pop[:, j]):
                    pop_x_traject[idx] = np.vstack((pop_x_traject[idx], pop[:, j]))
            else:
                pop_x_traject[idx] = np.atleast_2d(pop[:, j])

    return pop_track, pop_f_track, dominance_track, pop_x_traject, pop_f_traject, \
        hv_track, hist_sigma


def test_one_problem(k, problem, algorithm, **kwargs):

    # ------------------------------- animation fucntions ------------------------
    def init():
        """
        Note that this function create the basic frame upon which the animation takes
        place.
        """
        # plot the initial approximation set in objective space
#        for key, item in pop_x_traject.iteritems():
#            ax0.plot(item[0, 0], item[0, 1], ls='none', marker='s', ms=5, color='k',
#                     mec='none', alpha=0.7)

        # plot the initial approximation set in decision space
    #    for key, item in D_f_traject.iteritems():
    #        ax1.plot(item[0, 0], item[0, 1], ls='none', marker='s', ms=5, color='k',
    #                 mec='none', alpha=0.7)


        # create trajectories in the decision space
        for key, item in pop_x_traject.iteritems():
            lines_trajectory_x[key] = ax0.plot([], [], ls='-', lw=2,
                                               color='b', alpha=0.4)[0]

        # create trajectories in the objective space
        for key, item in pop_f_traject.iteritems():
            lines_trajectory_f[key] = ax1.plot([], [], ls='-', color='b',
                                               alpha=0.3)[0]

        # create all the search points
        for i in range(mu):
            points_x[i] = ax0.plot([], [], 'o',  mec='none', alpha=0.6)[0]
            points_f[i] = ax1.plot([], [], 'o',  mec='none', alpha=0.6)[0]

        line_hv.set_data([], [])

        return []

    def animate(ind):
        """
        The object returned by this function is going to be updated after each
        frame.
        """
        # delays...
        itercount = ind + 1

        if ind == 0:
            animate.t_start = time.time()

        if itercount % 20 == 0:
            print 'fps: ', 20 / (time.time() - animate.t_start)
            animate.t_start = time.time()

        pop = pop_track.loc[ind].values.reshape(-1, mu, order='F')
        fitness = pop_f_track.loc[ind].values.reshape(-1, mu, order='F')
        dom = array(dominance_track.loc[ind])

        # plot the non-dominated layers (anti-chains)
        for i in range(mu):
            rank = dom[i]
            points_x[i].set_data(*pop[:, i])
            points_f[i].set_data(*fitness[:, i])
            points_x[i].set_color(colors[rank%len(colors)])
            points_f[i].set_color(colors[rank%len(colors)])

        # update the trajectory in the decision space
        for key, item in pop_x_traject.iteritems():
            rank = dom[key]
            if item.shape[0] >= itercount:
                lines_trajectory_x[key].set_color(colors[rank%len(colors)])
                if itercount < trace_len_x:
                    lines_trajectory_x[key].set_data(item[0:itercount, 0],
                        item[0:itercount, 1])
                else:
                    lines_trajectory_x[key].set_data(item[itercount-trace_len_x:itercount, 0],
                    item[itercount-trace_len_x:itercount, 1])

        # update the trajectory in the objective space
        for key, item in pop_f_traject.iteritems():
            rank = dom[key]
            if item.shape[0] >= itercount:
                lines_trajectory_f[key].set_color(colors[rank%len(colors)])
                if itercount < trace_len_f:
                    lines_trajectory_f[key].set_data(item[0:itercount, 0],
                        item[0:itercount, 1])
                else:
                    lines_trajectory_f[key].set_data(item[itercount-trace_len_f:itercount, 0],
                        item[itercount-trace_len_f:itercount, 1])

        # update the hypervolume indicator value
        hv_text.set_text('HV = {}'.format(hv_track[ind]))
        line_hv.set_data(np.arange(ind), hv_track[:ind])

        return [hv_text, line_hv] + \
            [l for l in lines_trajectory_x.itervalues()] + \
            [l for l in lines_trajectory_f.itervalues()] + \
            [l for l in points_f.itervalues()] + \
            [l for l in points_x.itervalues()]


    # ------------------------------- animation end ------------------------------
    dim = kwargs['dim']
    step_size = kwargs['step_size']
    maxiter = kwargs['maxiter']
    mu = kwargs['mu']
    path = kwargs['path']
    is_animated = kwargs['is_animated']
    fps_movie = kwargs['fps_movie']
    save_animation = kwargs['save_animation']
    n_peaks1, n_peaks2, seed1, seed2 = map(kwargs.get, ('n_peaks1', 'n_peaks2',
                                                        'seed1', 'seed2'))

    data_dir = os.path.join(path, 'csv')
    movie_dir = os.path.join(path, 'movie')

    try:
        os.makedirs(data_dir)
        os.makedirs(movie_dir)
    except OSError:
        pass

    trace_len_x = 70
    trace_len_f = 20

    print '{}testing on MPM2 problem {}, n1:{}, n2:{}, s1:{}, s2:{}'.format(_tab,
        k+1, n_peaks1, n_peaks2, seed1, seed2)

    # setup problem
    if problem == 'MPM2':
        objfun, grad, f_dim, x_lb, x_ub, ref = setup_MPM2_problem(dim, n_peaks1,
            n_peaks2, seed1, seed2)
    elif problem == 'ZDT':
        objfun, grad, f_dim, x_lb, x_ub, ref = setup_ZDT_problem(dim,
                                                                 kwargs['zdt.index'])

    # setup MO-Optimization algorithm
    if algorithm == 'HIGA':
        optimizer = setup_HIGA_optimizer(dim, f_dim, mu, objfun, grad, x_lb,
                                         x_ub, ref, step_size, maxiter)
    elif algorithm == 'naive':
        optimizer = setup_naive_optimizer(dim, f_dim, mu, objfun, x_lb, x_ub, ref,
            step_size, maxiter)

    # run the optimizer for once
    _ = run_optimizer(optimizer, maxiter, mu, ref)
    pop_track, pop_f_track, dominance_track, pop_x_traject, pop_f_traject, \
        hv_track, sigma = _

    # save the population tracking information to csv
    pop_track.to_csv('{}/csv/decision_vector-{}.csv'.format(path, k+1), index=False)
    pop_f_track.to_csv('{}/csv/objective_vectors-{}.csv'.format(path, k+1), index=False)
    dominance_track.to_csv('{}/csv/dominance-{}.csv'.format(path, k+1), index=False)

    print '{}testing done...'.format(_tab)

    # ------------------------- make the animation -------------------------------
    # optional: generate the animation
    if is_animated:
        # setup the figure for plotting
        fig, (ax0, ax1, ax2), hv_text = setup_biobj_plot(x_lb, x_ub, ref, True)

        fig.suptitle('npeaks1: {} -- npeaks2: {} -- seed1: {} -- seed2: {}'.format(n_peaks1, \
            n_peaks2, seed1, seed2))

        # plot the contour lines of objective functions
        biobj_contour_plot(ax0, objfun, x_lb, x_ub)

        lines_trajectory_x = {}
        lines_trajectory_f = {}
        points_x = {}
        points_f = {}

        line_hv, = ax2.plot([], [], 'k-')

        # setup the aspect ratio such that the physical length of the axis is equal
        ratio = (1.0*maxiter) / (max(hv_track)*1.05 - min(hv_track)*0.95)
        ax2.set_aspect(aspect=str(ratio))
        ax2.set_xlim([0, maxiter])
        ax2.set_ylim([min(hv_track)*0.95, max(hv_track)*1.05])

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps_movie, metadata=dict(artist='Hao Wang'),
                        extra_args=['-vcodec',
                                    'libx264',
                                    '-pix_fmt',  # -pix_fmt: for MaxOS quicktime
                                    'yuv420p'])
        ani = animation.FuncAnimation(fig, animate, frames=maxiter, interval=10,
                                      blit=True, init_func=init, repeat=True)

        # save the movie
        if not save_animation:
            plt.show()
        elif save_animation:
            # Only .mov format can be played by QuickTime on OSX...
            print 'Saving the animation..., which might take a while.'
            ani.save('{}/movie/trajectory{}.mov'.format(path, k+1), writer=writer, dpi=100)
            print 'done...'

    # ------------------------- plot the final results ------------------------=--
    # setup the figure for plotting
    fig, (ax0, ax1), hv_text = setup_biobj_plot(x_lb, x_ub, ref)

#    fig.suptitle("npeaks1: {} -- npeaks2: {} -- seed1: {} -- seed2: {}".format(n_peaks1, \
#        n_peaks2, seed1, seed2))

    # plot the contour lines of objective functions
    biobj_contour_plot(ax0, objfun, x_lb, x_ub)

    fitness = optimizer.fitness
    front = fitness[:, optimizer.pareto_front]

    # plot the final approximation set
#    hv_text.set_text('HV = {}'.format(hv_track[-1]))

    # plot all layers found...
    for i, front_idx in enumerate(optimizer.fronts):
        front = fitness[:, front_idx]
        x_front = optimizer.pop[:, front_idx]

        ax0.plot(x_front[0, :], x_front[1, :], ls='none', marker='o', mec='none',
                 alpha=1, color=colors[i%len(colors)])
        ax1.plot(front[0, :], front[1, :], ls='none', marker='o', mec='none',
                 alpha=1, color=colors[i%len(colors)])

    # trajectories in the decision space
    for key, item in pop_x_traject.iteritems():
        item = np.atleast_2d(item)
        # stop the trajectory plotting just before a point becomes non-dominated
        end = next((i for i, r in enumerate(dominance_track.iloc[:, key]) if r == 0),
                   None)

        x, y = (item[:, 0], item[:, 1]) if end is None else (item[:end+1, 0],
                item[:end+1, 1])

        # plot the trajectory every 10 steps because the average steps are two
        # small to visualize
        x = np.r_[x[:-1:10], x[-1]]
        y = np.r_[y[:-1:10], y[-1]]

        # start location
#        ax0.plot(item[0, 0], item[0, 1], ls='none', marker='s', color='k',
#                 mec='none', alpha=0.5)

        # quiver plots
        ax0.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy',
                   angles='xy', width=1.8, units='dots', scale=1, alpha=0.3,
                   color='k', headwidth=4, headlength=6)

    # trajectories in the objective space
    for key, item in pop_f_traject.iteritems():
        item = np.atleast_2d(item)
        # stop the trajectory plotting just before a point becomes non-dominated
        end = next((i for i, r in enumerate(dominance_track.iloc[:, key]) if r == 0),
                   None)

        x, y = (item[:, 0], item[:, 1]) if end is None else (item[:, 0], item[:, 1])

        # start location is not plotted for MPM2 problem. Too crowded initialization
#        ax1.plot(item[0, 0], item[0, 1], ls='none', marker='s', ms=5, color='k',
#                 mec='none', alpha=0.6)
        # quiver plots
        ax1.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy',
                   angles='xy', width=1.6, units='dots', scale=1, alpha=0.2,
                   color='k', headwidth=3, headlength=4)

    return fig


def test_slave(**kwargs):

    is_MPI = False
    if not bool(kwargs):
        is_MPI = True
        from mpi4py import MPI
        # Get the master communicator
        comm = MPI.Comm.Get_parent()
        print 1
        prob_idx = comm.scatter(None, root=0)
        print 1
        df, problem, algorithm, opts, random_seed = comm.bcast(None, root=0)
    else:
        prob_idx, df, problem, algorithm, opts, random_seed = \
            map(kwargs.get, ('prob_idx', 'df', 'problem', 'algorithm',
                             'opts', 'random_seed'))

    fig = {}

    # the actual computation are here...
    for i in prob_idx: 
        np.random.seed(random_seed)   # seed random seed before each experiment
        par = df.loc[i]
        fig[i] = test_one_problem(i, problem, algorithm, n_peaks1=par.n1, \
            n_peaks2=par.n2, seed1=par.s1, seed2=par.s2, **opts)

    if is_MPI:
        # synchronization...
        comm.Barrier()

        # return performance metrics
        comm.gather(fig, root=0)

        # free all slave processes
        comm.Disconnect()
    else:
        return fig


if len(sys.argv) == 2 and sys.argv[1] == '-slave':
    test_slave()
    sys.exit()


if __name__ == '__main__':

    random_seed = 700     # for reproducible results
    algorithm = 'HIGA'
    problem_str = 'MPM2'

    opts = {
            'mu': 20,                     # size of the approximation set
            'maxiter': int(500),          # iteration budget
            'step_size': 0.01,           # initial step-size
            'fps_movie': 40,              # animation fps
            'dim': 2,                     # decision space dimension
            'is_animated': False,          # dynamic plot
            'save_animation': True,       # show animation
            'path': './results/{}/'.format(algorithm)    # data storage folder
            }

    n_process = 1    # 1 for sequential execution, >1 for parallel execution
                         # requires MPI installation

    df = read_csv('./setup.csv')
    n_problems = df.shape[0]

    figures = {}

    if n_process > 1:
        from mpi4py import MPI

        binsize = n_problems / n_process
        reminder = n_problems % n_process
        index = [[i + j*n_process for j in range(binsize+int(i<reminder))] \
            for i in range(n_process)]

        # parallel execution
        print 'Running in parallel...'
        # Spawning processes
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=['./test_MPM2.py', '-slave'],
                                   maxprocs=n_process)

        # scatter the register for historical information
        comm.scatter(index, root=MPI.ROOT)
        comm.bcast([df, problem_str, algorithm, opts, random_seed], root=MPI.ROOT)

        # Synchronization while the children process are performing
        # heavy computations...
        comm.Barrier()

        # Gether the fitted model from the childrenn process
        # Note that 'None' is only valid in master-slave working mode
        results = comm.gather(None, root=MPI.ROOT)

        for r in results:
            figures.update(r)

        # free all slave processes
        comm.Disconnect()

        print 'done...'
        print

    elif n_process == 1:
        # parallel execution
        print 'Running sequentially...'
        figures = test_slave(problem=problem_str, algorithm=algorithm,
                             df=df, prob_idx=range(n_problems),
                             random_seed=random_seed, opts=opts)

        print 'done...'
        print

    print 'Plotting to pdf...'

    if 11 < 2:
        # save figures to file
        with PdfPages(opts['path'] + 'results.pdf') as pdf:
            for key in sorted(figures):
                pdf.savefig(figures[key])

            # set the file's metadata
            d = pdf.infodict()
            d['Title'] = 'Hypervolume indicator Gradient Ascent Algorithm on MPM2 problem'
            d['Author'] = 'Hao Wang'
            d['Subject'] = 'Final Pareto front and the trajectories of dominated points'
            d['Keywords'] = 'PdfPages multipage keywords author title subject'
            d['CreationDate'] = datetime.today()
    else:
        for i, key in enumerate(sorted(figures)):
            fig = figures[key]
            par = '_'.join(map(str, df.loc[i]))
            fig.savefig(opts['path'] + par + '.png')


    print 'done...'
    print
