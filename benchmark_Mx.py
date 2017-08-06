# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:31:42 2016

@author: Hao Wang

@Notes:
    ------------------------------------------------------------------------------
    Benchmark the hypervolume gradient ascent algorithm and compare to other 
    multi-objective optimization algorithms
    ------------------------------------------------------------------------------
    16 March 2016: merge the slave file with the master file
"""

import pdb
import sys
import numpy as np
from numpy import zeros, array

from moo_gradient import MOO_HyperVolumeGradient

from mpi4py import MPI
    
import matplotlib.pyplot as plt
from matplotlib import rcParams

from MOO_problem import problem as my_problem

import cPickle as cp

from hv import HyperVolume

# ------------------------------ fix the unpicklable problem --------------------
import copy_reg
import types


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


# --------------------------- set up matplotlib ----------------------------------
rcParams['legend.numpoints'] = 1 
rcParams['xtick.labelsize'] = 25
rcParams['ytick.labelsize'] = 25
rcParams['xtick.major.size'] = 7
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.size'] = 7
rcParams['ytick.major.width'] = 1
rcParams['axes.labelsize'] = 25

_color = ['r', 'g', 'b', 'c', 'k', 'm', 'y']

fig_width = 22
fig_height = 22 * 9 / 16

_table = ' ' * 3


def slave():
    """This function performs the actual heavy computation
    """
    # Get the master communicator
    comm = MPI.Comm.Get_parent()
    
    optimizer, prob = comm.bcast(None, root=0)
    
    n_approximation = 1000
    ref = prob.ref
    pareto_f2 = prob.pareto_front()
    hv = HyperVolume(ref)
    
    f1 = np.linspace(0, 1, n_approximation)
    f2 = pareto_f2(f1)
    pareto_front = np.vstack([f1, f2]).T
    
    # run the algorithm
    optimizer.optimize()
        
    front_idx = optimizer.pareto_front
    front = optimizer.fitness[:, front_idx].T
    
    # Compute the performance metrics
    volume = hv.compute(front)
    n_point = len(front_idx)
    
    convergence = 0
    for p in front:
        dis = np.sqrt(np.sum((pareto_front - p) ** 2.0, axis=1))
        convergence += np.min(dis)
    
    convergence /= len(front_idx)
    
    # synchronization...
    comm.Barrier()
    
    output = {
             'hv': volume,
             'convergence': convergence,
             'n_point': n_point
             }
    
    # return performance metrics
    comm.gather(output, root=0)
    
    # free all slave processes
    comm.Disconnect()
    

argv = sys.argv

if len(argv) == 2 and argv[1] == '-slave':
    slave()
    sys.exit()
    
# --------------------------- settings for benchmark -----------------------------
# maximal iterations
maxiter = int(2e2)

# initial step size
#step_size = 0.008

# independent runs
runs = 8

# problem dimensions
dims = [2, 10, 50]
n_dim = len(dims)

# problem set
problem_str = [r'GSP convex $\alpha=0.75$', r'GSP linear $\alpha=0.5$', 
            r'GSP concave $\alpha=0.25$']
problems = [0.75, 0.5, 0.25]
n_problem = len(problems)

# algorithms to be tested
algorithms = ['M{}'.format(i) for i in range(1, 6)] + ['ns']  # benchmark M1-M5 
n_algorithm = len(algorithms)

# reference point
ref = [11, 11]

# performance metric declaration
hv = zeros((n_dim, n_problem, n_algorithm, runs))
convergence = zeros((n_dim, n_problem, n_algorithm, runs))
    
for i, d in enumerate(dims):
    print 'on {} dimension...'.format(d)
    
    # approximation set size
#    pop_size = int(np.ceil(np.sqrt(20.0 * d)) + 10)
    pop_size = int(50)
    
    for j, prob_ind in enumerate(problems):
        print '{}optimizing problem {}...'.format(_table, prob_ind)
        
        # initialize the problem
        # my own problem set
        prob = my_problem(5, d, alpha=prob_ind)
        
        fitness = prob.objfun
        grad = prob.objgrad
        f_dim = prob.f_dimension
        x_lb = array(prob.lb)
        x_ub = array(prob.ub)
         
        for k, algo in enumerate(algorithms):
            print '{}testing {}...'.format(_table * 2, algo)
            
            if algo == 'ns':
                is_ns = True
                algo = 'M1'
            else:
                is_ns = False
                
            # create a instance of our gradient-based Bi-objective algorithm 
            opts = {'lb' : x_lb,
                    'ub' : x_ub,
                    'maxiter' : np.inf,
                    'heuristic' : algo,
                    'maxiter': maxiter,
                    'non_dominated_sorting' : is_ns,
                    'enable_dominated' : True}
            
            # initial step-size setting
            step_size = 0.0025 * np.max(x_ub - x_lb) * np.sqrt(d) / np.sqrt(dims[0])
            
            optimizer = MOO_HyperVolumeGradient(d, f_dim, pop_size, fitness,
                                                grad, ref=ref, opts=opts, 
                                                step_size=step_size,
                                                sampling='unif', maximize=False,
                                                normalize=True)

            # --------------------------- run algorithms -------------------------
            # parallel execution
            print '{}in parallel...'.format(_table * 3)
            # Spawning processes to test ES algorithms
            comm = MPI.COMM_SELF.Spawn(sys.executable, args=['./benchmark_Mx.py', '-slave'], 
                                       maxprocs=runs)
        
            # scatter the register for historical information
            comm.bcast([optimizer, prob], root=MPI.ROOT)
                        
            # Synchronization while the children process are performing 
            # heavy computations...
            comm.Barrier()
                
            # Gether the fitted model from the childrenn process
            # Note that 'None' is only valid in master-slave working mode
            results = comm.gather(None, root=MPI.ROOT)
            
            # register the running outputs
            hv[i, j, k, :] = array([r['hv'] for r in results])
            convergence[i, j, k, :] = array([r['convergence'] for r in results])
            tmp = array([r['n_point'] for r in results])
            print '{}hv mean: {}, std: {}'.format(_table * 4, 
                np.mean(hv[i, j, k, :]), np.std(hv[i, j, k, :]))
            print '{}convergence mean: {}, std: {}'.format(_table * 4, 
                np.mean(convergence[i, j, k, :]), np.std(convergence[i, j, k, :]))
            print '{}Avg. point number: {}, std: {}'.format(_table * 4, 
                np.mean(tmp), np.std(tmp))
            
            # free all slave processes
            comm.Disconnect()
            print '{}Done...'.format(_table * 3)    
        
        mean_hv_per_problem = np.mean(hv[i, j, :, :], axis=1)
        best_per_problem = np.nonzero(mean_hv_per_problem == max(mean_hv_per_problem))[0]
        mean_cov_per_problem = np.mean(convergence[i, j, :, :], axis=1)
        best_per_problem2 = np.nonzero(mean_cov_per_problem == min(mean_cov_per_problem))[0]
        print '-' * 80
        print 'On dim {}, in terms of hv: the best algorithm on problem {} is {}'.format(prob_ind, 
            d, algorithms[best_per_problem])
        print 'On dim {}, in terms of conv: the best algorithm on problem {} is {}'.format(prob_ind, 
            d, algorithms[best_per_problem2])
        print '-' * 80
        
        
# save the raw benchmark data
with open('./result.dat', 'w') as f_data:
    cp.dump({'dim': dims,
             'maxiter': maxiter,
             'runs': runs,
             'problem': problem_str,
             'algorithm': algorithms,
             'ref.algorithm': ref,
             'ref.eval': prob.ref,
             'pop_size': pop_size,
             'hv': hv,
             'convergence': convergence
             }, f_data)

        
# ------------------------ write the result to latex table -----------------------
print 'creating Latex tables...'
hv_mean = np.mean(hv, axis=-1)
hv_std = np.std(hv, axis=-1)

convergence_mean = np.mean(convergence, axis=-1)
convergence_std = np.std(convergence, axis=-1)

hv_sort = np.argsort(hv_mean, axis=-1)[:, :, ::-1]
hv_rank = np.empty(np.shape(hv_mean), dtype='int')

for i, block in enumerate(hv_sort):
    for j, row in enumerate(block):
        hv_rank[i, j][row] = range(1, hv_rank.shape[-1]+1)
        
convergence_sort = np.argsort(convergence_mean, axis=-1)
convergence_rank = np.empty(np.shape(convergence_mean), dtype='int')

for i, block in enumerate(convergence_sort):
    for j, row in enumerate(block):
        convergence_rank[i, j][row] = range(1, convergence_rank.shape[-1]+1)

with open('./result.tex', 'w') as f:    
    for k, d in enumerate(dims):
        f.write('\\begin{table}[!h]\n')
        f.write('\caption{{Results in {}-D}}\n'.format(d))
        f.write('\label{{tab:res-{}d}}\n'.format(d))
        f.write('\centering\n')
        f.write('\\begin{tabular}[h]{l|l||r|r|c||r|r|c}\n')
        f.write('Test- &  & \multicolumn{3}{c||}{Convergence measure} &' + \
            '\multicolumn{3}{c}{$\mathcal{S}$ metric}\\\\ \cline{3-8}\n')
        f.write('function & Algorithm & Average & Std. dev. ' + \
            '& Rank & Average & Std. dev. & Rank \\\\ \n')
        f.write('\hline \hline \n')
        for i, prob in enumerate(problem_str):
            prob_str = prob.split(' ')
            
            for j, algo in enumerate(algorithms):
                __str = prob_str[j] if j < len(prob_str) else ''
                
                if convergence_rank[k, i, j] == 1:
                    __str += '& {:s} & \\textbf{{{:.8f}}} & {:.4e} & {:d}'
                else:
                    __str += '& {:s} & {:.8f} & {:.4e} & {:d}'
                    
                if hv_rank[k, i, j] == 1:
                    __str +=  '& \\textbf{{{:.8f}}} & {:.4e} & {:d} \\\\ \n'
                else:
                    __str += '& {:.8f} & {:.4e} & {:d} \\\\ \n'
                    
                __str = __str.format(algo, convergence_mean[k, i, j], 
                                     convergence_std[k, i, j], 
                                     convergence_rank[k, i, j], hv_mean[k, i, j], 
                                     hv_std[k, i, j], hv_rank[k, i, j])
                    
                if j == len(algorithms) - 1:
                    __str += '\hline \n'
                f.write(__str)
                
        f.write('\end{tabular} \n')
        f.write('\end{table} \n')
        f.write('\n')

print 'completed...'


def plot_PF():
    pass
    