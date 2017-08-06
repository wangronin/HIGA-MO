# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:31:42 2016

@author: Hao Wang

@Notes:
    ------------------------------------------------------------------------------
    Benchmark the hypervolume gradient ascent algorithm and compare to other 
    multi-objective optimization algorithms
"""

import pdb
import sys
import numpy as np
from numpy import zeros

from numpy import array
from moo_gradient import MOO_HyperVolumeGradient
from naive_optimizer import MOO_naive
import mo_problem as MOP
from hv import HyperVolume

#from PyGMO import algorithm, population, problem

from mpi4py import MPI

_table = ' ' * 3

# ------------------------------ fix the unpicklable problem --------------------
import copy_reg
import types


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


def ERT(run_time_array, performance, target=1e-8):
    total_run_time = 1.0 * np.sum(run_time_array)
    n_successful = len(performance < target)
    return total_run_time / n_successful
    
    
#==============================================================================
# MPI slave that does most of the computations
def mpi_slave():

    comm = MPI.Comm.Get_parent()
    
    d, pop_size, maxiter, algo, prob, prob2, ref, target = comm.bcast(None, root=0)
    
    x_lb = array(prob.lb)
    x_ub = array(prob.ub)
    hv = HyperVolume(ref)
    
    # initialize algorithms
    if algo in ['HIGA-MO', "Lara's direction", 'Gap-filling']:
        non_dominated_sorting = True if algo == 'HIGA-MO' else False
        if algo == "Lara's direction":
            heuristic = 'M3'
        elif algo == 'Gap-filling':
            heuristic = 'M5'
        else:
            heuristic = 'M1'
            
        opts = {
                'lb' : x_lb,
                'ub' : x_ub,
                'heuristic' : heuristic,
                'maxiter': maxiter,
                'non_dominated_sorting' : non_dominated_sorting,
                'performance_metric' : prob.convergence_metric,
                'target' : target,
                'enable_dominated' : True
                }
            
        # TODO: initial set-size setting
        step_size = 0.001 * np.max(x_ub - x_lb)
        optimizer = MOO_HyperVolumeGradient(d, prob.f_dim, pop_size,
                                            prob.objfun, prob.objfun_dx, ref, 
                                            opts, step_size=step_size,
                                            sampling='unif', maximize=False,
                                            normalize=True)
    elif algo == 'SLS':
        # Naive Stochastic Local search (the one used in MELA project)
        # multi-objective group hill-climbing
        opts = {
                'lb' : x_lb,
                'ub' : x_ub,
                'performance_metric' : prob.convergence_metric,
                'target' : target
                }
        
        step_size = 0.01 * np.max(x_ub - x_lb)
        optimizer = MOO_naive(d, prob.f_dim, pop_size, prob.objfun, opts=opts, 
                              step_size=step_size, sampling='unif', 
                              maxiter=maxiter, maximize=False)
    
    elif algo == 'NSGA-II':
        # NSGA-II optimizer
        pop = population(prob2, pop_size)
        optimizer = algorithm.nsga_II(gen=maxiter)
        
    elif algo == 'SMS-EMOA':
        # SMS-EMOA optimizer
        pop = population(prob2, pop_size)
        optimizer = algorithm.sms_emoa(gen=maxiter*pop_size, sel_m=1)
        
    elif algo == 'SPEA2':
        # SMS-EMOA optimizer
        pop = population(prob2, pop_size)
        optimizer = algorithm.spea2(gen=maxiter)
        
    # Optimizer is Running...
    if algo in ['HIGA-MO', "Lara's direction", 'Gap-filling', 'SLS']:
        optimizer.optimize()
        
        front_idx = optimizer.pareto_front
        front = optimizer.fitness[:, front_idx].T
        
    else:
        pop = optimizer.evolve(pop)
        front_idx = pop.compute_pareto_fronts()[0]
        front = array([list(pop[i].cur_f) for i in front_idx])
    
    # compute the convergence metric
    convergence = prob.convergence_metric(front)
        
    # compute the hypervolume indicator
    volume = hv.compute(front)
    
    # the number of points on the non-dominated set
    n_point = len(front_idx)
    
    # synchronization...
    comm.Barrier()
    
    output = {
             'hv' : volume,
             'convergence' : convergence,
             'itercount' : optimizer.itercount,
             'n_point' : n_point,
             }
    
    comm.gather(output, root=0)
    comm.Disconnect()
#==============================================================================

  
argv = sys.argv
if len(argv) == 2 and argv[1] == '-slave':
    mpi_slave()
    sys.exit()


#==============================================================================
# settings for benchmark
dims = [2]
runs = 15
maxiter = int(1e5)
algorithms = ['HIGA-MO', "Lara's direction", 'Gap-filling', 'NSGA-II', 'SMS-EMOA', 'SPEA2']  
#algorithms = ['HIGA-MO', 'Lara\' direction', 'Gap-filling']  
algorithms = ['HIGA-MO', 'SLS']  
#problems = range(1, 5) + [6]   # ZDT1 to ZDT4 and ZDT6
problems = [1] # double sphere problem
target = 0.01
#==============================================================================

#==============================================================================
# 
n_dim = len(dims)
n_algorithm = len(algorithms)
n_problem = len(problems)
problem_str = ['ZDT' + str(i) for i in problems]
#==============================================================================

#==============================================================================
# performance metrics declaration
hv = zeros((n_dim, n_problem, n_algorithm, runs))
convergence = zeros((n_dim, n_problem, n_algorithm, runs))
itercount = zeros((n_dim, n_problem, n_algorithm, runs))
expected_run_time = zeros((n_dim, n_problem, n_algorithm))
#==============================================================================
    
for i, d in enumerate(dims):
    print 'on {} dimension...'.format(d)
    
    pop_size = 50
    
    for j, prob_ind in enumerate(problems):
        print '{}optimizing ZDT{} problem...'.format(_table, prob_ind)
        
        # initialize the problem
#        d = 10 if prob_ind in [4, 6] else 30
        prob = MOP.simple(prob_ind, d)
        prob2 = None
#        prob2 = problem.zdt(prob_ind, d)
        
        ref = [11, 11] if prob_ind == 6 else [1.1, 1.1]
        
        for k, algo in enumerate(algorithms):
            print '{}testing {}...'.format(_table * 2, algo)

            
            # --------------------------- run algorithms -------------------------
            # parallel execution
            print '{}in parallel...'.format(_table * 3)
            # Spawning processes to test ES algorithms
            comm = MPI.COMM_SELF.Spawn(sys.executable, 
                                       args=['./benchmark.py', '-slave'], 
                                       maxprocs=runs)
                                       
            # scatter the register for historical information
            comm.bcast([d, pop_size, maxiter, algo, prob, prob2, ref, target], 
                       root=MPI.ROOT)
                        
            # Synchronization while the children process are performing 
            # heavy computations...
            comm.Barrier()
                
            # Gether the fitted model from the childrenn process
            # Note that 'None' is only valid in master-slave working mode
            results = comm.gather(None, root=MPI.ROOT)
            
            # register the running outputs
            hv[i, j, k, :] = array([r['hv'] for r in results])
            convergence[i, j, k, :] = array([r['convergence'] for r in results])
            itercount[i, j, k, :] = array([r['itercount'] for r in results])
            n_point = array([r['n_point'] for r in results])
            expected_run_time[i, j, k] = ERT(itercount[i, j, k, :], 
                convergence[i, j, k, :], target)
            
            print expected_run_time[i, j, k]
            
            print '{}hv mean: {}, std: {}'.format(_table * 4, 
                np.mean(hv[i, j, k, :]), np.std(hv[i, j, k, :]))
                
            print '{}convergence mean: {}, std: {}'.format(_table * 4, 
                np.mean(convergence[i, j, k, :]), np.std(convergence[i, j, k, :]))
                
            print '{}itercount mean: {}, std: {}'.format(_table * 4, 
                np.mean(itercount[i, j, k, :]), np.std(itercount[i, j, k, :]))
                
            print '{}Avg. point number: {}, std: {}'.format(_table * 4, 
                np.mean(n_point), np.std(n_point))
            
            # free all slave processes
            comm.Disconnect()
            print '{}Done...'.format(_table * 3)
            
        mean_hv_per_problem = np.mean(hv[i, j, :, :], axis=1)
        best_per_problem = np.nonzero(mean_hv_per_problem == max(mean_hv_per_problem))[0][0]
        mean_cov_per_problem = np.mean(convergence[i, j, :, :], axis=1)
        best_per_problem2 = np.nonzero(mean_cov_per_problem == min(mean_cov_per_problem))[0][0]
        
        print '-' * 80
        print 'On dim {}, in terms of hv: the best algorithm on problem {} is {}'.format(d, 
            prob_ind, algorithms[best_per_problem])
        print 'On dim {}, in terms of conv: the best algorithm on problem {} is {}'.format(d, 
            prob_ind, algorithms[best_per_problem2])
        print '-' * 80
            
            
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
            
